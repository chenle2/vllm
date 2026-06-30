# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler, _SchedulingContext
from vllm.v1.request import Request

logger = init_logger(__name__)

WRS_A = 0.4
WRS_B = 0.6
MAX_QUEUES = 4
T_BASE = 10.0
SLACK_FACTOR = 40.0
PHYSICAL_C = 0.000176
DEFAULT_T_REFRESH = 30.0
DEFAULT_STARVATION_URGENCY_THRESHOLD = 2.0
DEFAULT_STARVATION_PROMPT_CAP = 800
DEFAULT_MAX_CONCURRENT_DECODES = 36

# 默认值来自 slora Chameleon v13 artifact。``agent`` 是第一版 vLLM 迁移的
# 目标场景；下面的环境变量可以覆盖运行时参数，而不需要扩展公开的
# SchedulerConfig 配置面。
CHAMELEON_DB = {
    "real": {
        "max_input": 1021.0,
        "max_output": 977.0,
        "cut_offs": [0.1168, 0.2596, 0.4462],
    },
    "agent": {
        "max_input": 4988.0,
        "max_output": 4802.0,
        "cut_offs": [0.0857, 0.1299, 0.1915],
    },
}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s.", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s.", name, value, default)
        return default


def _new_fcfs_queue(requests: Iterable[Request] = ()) -> FCFSRequestQueue:
    queue = FCFSRequestQueue()
    for request in requests:
        queue.add_request(request)
    return queue


class ChameleonScheduler(Scheduler):
    """使用 Chameleon v13 waiting 请求排序的 vLLM v1 scheduler。

    running 请求、KV 分配、LoRA admission、encoder budget 和抢占仍由基础
    Scheduler 负责。Chameleon 只决定 eligible waiting 请求的顺序，并且只对
    实际 admission 成功的请求做 DRR 记账。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        scenario = os.environ.get("VLLM_CHAMELEON_SCENARIO", "agent")
        cfg = CHAMELEON_DB.get(scenario, CHAMELEON_DB["agent"])
        if scenario not in CHAMELEON_DB:
            logger.warning("Unknown Chameleon scenario %r; using agent.", scenario)

        self.chameleon_scenario = scenario
        self.chameleon_global_max_in = cfg["max_input"]
        self.chameleon_global_max_out = cfg["max_output"]
        self.chameleon_cut_offs = list(cfg["cut_offs"])
        self.chameleon_pq_tokens = [10000] * MAX_QUEUES
        self.chameleon_deficits = [0.0] * MAX_QUEUES
        self.chameleon_last_update_wall_time = time.time()
        self.chameleon_t_refresh = _env_float(
            "VLLM_CHAMELEON_T_REFRESH", DEFAULT_T_REFRESH
        )
        self.chameleon_max_concurrent_decodes = max(
            1,
            _env_int(
                "VLLM_CHAMELEON_MAX_CONCURRENT_DECODES",
                DEFAULT_MAX_CONCURRENT_DECODES,
            ),
        )
        self.chameleon_starvation_threshold = _env_float(
            "VLLM_CHAMELEON_STARVATION_URGENCY_THRESHOLD",
            DEFAULT_STARVATION_URGENCY_THRESHOLD,
        )
        self.chameleon_starvation_prompt_cap = _env_int(
            "VLLM_CHAMELEON_STARVATION_PROMPT_CAP",
            DEFAULT_STARVATION_PROMPT_CAP,
        )

        if self.scheduler_config.async_scheduling:
            logger.warning(
                "ChameleonScheduler is intended for async_scheduling=False; "
                "continuing with the base scheduler mechanics."
            )

    def _get_wrs(self, request: Request) -> float:
        input_norm = request.num_prompt_tokens / self.chameleon_global_max_in
        output_norm = request.max_tokens / self.chameleon_global_max_out
        return WRS_A * input_norm + WRS_B * output_norm

    def _bucket_index(self, request: Request) -> int:
        wrs = self._get_wrs(request)
        for idx, cut_off in enumerate(self.chameleon_cut_offs):
            if wrs < cut_off:
                return idx
        return MAX_QUEUES - 1

    @staticmethod
    def _request_cost(request: Request) -> int:
        return request.num_prompt_tokens + request.max_tokens

    def _ttft_slo(self, request: Request) -> float:
        return T_BASE + max(PHYSICAL_C * request.num_prompt_tokens * SLACK_FACTOR, 0.05)

    def _urgency(self, request: Request, current_wall_time: float) -> float:
        wait_time = max(0.0, current_wall_time - request.arrival_time)
        return wait_time / max(self._ttft_slo(request), 0.01)

    def _calculate_hrrn_score(
        self, request: Request, current_wall_time: float
    ) -> float:
        wait_time = max(0.0, current_wall_time - request.arrival_time)
        service_time_est = (request.num_prompt_tokens * PHYSICAL_C) + (
            request.max_tokens * 0.05
        )
        urgency = self._urgency(request, current_wall_time)
        virtual_wait = wait_time * (urgency**2) if urgency > 1.0 else wait_time
        return (virtual_wait + service_time_est) / max(service_time_est, 1e-6)

    @staticmethod
    def _one_dimensional_kmeans_cutoffs(values: list[float]) -> list[float]:
        # 保持迁移版本无额外依赖：slora 使用 sklearn.KMeans，但 vLLM 不应
        # 仅为了启用这个 scheduler 就依赖 sklearn。
        if len(values) < MAX_QUEUES:
            return []

        sorted_values = sorted(values)
        centers = [
            sorted_values[
                min(len(sorted_values) - 1, i * len(sorted_values) // MAX_QUEUES)
            ]
            for i in range(MAX_QUEUES)
        ]
        for _ in range(8):
            clusters = [[] for _ in range(MAX_QUEUES)]
            for value in sorted_values:
                idx = min(
                    range(MAX_QUEUES),
                    key=lambda i: (abs(value - centers[i]), i),
                )
                clusters[idx].append(value)
            new_centers = [
                sum(cluster) / len(cluster) if cluster else centers[i]
                for i, cluster in enumerate(clusters)
            ]
            if all(abs(new_centers[i] - centers[i]) < 1e-9 for i in range(MAX_QUEUES)):
                break
            centers = new_centers

        centers = sorted(centers)
        return [(centers[i] + centers[i + 1]) / 2 for i in range(MAX_QUEUES - 1)]

    def _update_adaptive_parameters(
        self, current_wall_time: float, backlog: list[Request]
    ) -> None:
        if len(backlog) < 10:
            return

        wrs_values = [self._get_wrs(request) for request in backlog]
        cutoffs = self._one_dimensional_kmeans_cutoffs(wrs_values)
        if cutoffs:
            self.chameleon_cut_offs = cutoffs

        bins: list[list[Request]] = [[] for _ in range(MAX_QUEUES)]
        for request in backlog:
            bins[self._bucket_index(request)].append(request)

        time_window = max(current_wall_time - self.chameleon_last_update_wall_time, 1.0)
        new_quotas = []
        for bucket in bins:
            if not bucket:
                new_quotas.append(1000)
                continue
            costs = [self._request_cost(request) for request in bucket]
            max_cost = max(costs)
            mean_cost = sum(costs) / len(costs)
            variance = (
                sum((cost - mean_cost) ** 2 for cost in costs) / len(costs)
                if len(costs) > 1
                else 0.0
            )
            lam = len(bucket) / time_window
            quantum = 5 * (max_cost + (lam * variance * 0.001))
            new_quotas.append(max(int(quantum), 1000))

        self.chameleon_pq_tokens = new_quotas
        self.chameleon_last_update_wall_time = current_wall_time

    def _grant_drr_deficits(self, backlog: list[Request]) -> None:
        non_empty = {self._bucket_index(request) for request in backlog}
        for idx in range(MAX_QUEUES):
            if idx in non_empty:
                self.chameleon_deficits[idx] = min(
                    self.chameleon_deficits[idx] + self.chameleon_pq_tokens[idx],
                    self.chameleon_pq_tokens[idx] * 3.0,
                )
            else:
                self.chameleon_deficits[idx] = 0.0

    def _ordered_candidates(
        self, backlog: list[Request], current_wall_time: float
    ) -> tuple[list[Request], dict[str, int]]:
        buckets: list[list[Request]] = [[] for _ in range(MAX_QUEUES)]
        bucket_by_id: dict[str, int] = {}
        for request in backlog:
            idx = self._bucket_index(request)
            buckets[idx].append(request)
            bucket_by_id[request.request_id] = idx

        candidate_ids: set[str] = set()
        ordered: list[Request] = []
        # 排序时只消耗一份影子副本。真实 DRR deficit 只有在 vLLM 成功
        # admission 请求之后才扣费。
        available_deficits = list(self.chameleon_deficits)

        def maybe_add(request: Request) -> None:
            if request.request_id in candidate_ids:
                return
            idx = bucket_by_id[request.request_id]
            cost = self._request_cost(request)
            if cost > available_deficits[idx]:
                return
            available_deficits[idx] -= cost
            candidate_ids.add(request.request_id)
            ordered.append(request)

        # 饥饿预通道对应 v13：严重超时的小 prompt 先获得机会，但仍然必须
        # 有足够的 DRR credit。
        starved = [
            request
            for request in backlog
            if request.num_prompt_tokens < self.chameleon_starvation_prompt_cap
            and self._urgency(request, current_wall_time)
            > self.chameleon_starvation_threshold
        ]
        starved.sort(
            key=lambda request: self._urgency(request, current_wall_time),
            reverse=True,
        )
        for request in starved:
            maybe_add(request)

        # 正常通道按高 WRS 到低 WRS 扫描分桶，并在每个桶内用 mHRRN 偏向
        # SLO urgency 正在升高的请求。
        for idx in range(MAX_QUEUES - 1, -1, -1):
            bucket = sorted(
                buckets[idx],
                key=lambda request: self._calculate_hrrn_score(
                    request, current_wall_time
                ),
                reverse=True,
            )
            for request in bucket:
                maybe_add(request)

        return ordered, bucket_by_id

    def _schedule_waiting_requests(self, ctx: _SchedulingContext) -> None:
        if len(self.running) >= self.chameleon_max_concurrent_decodes:
            return

        current_wall_time = time.time()
        original_waiting = list(self.waiting)
        original_skipped = list(self.skipped_waiting)
        backlog = original_waiting + original_skipped
        if not backlog:
            return

        if (
            current_wall_time - self.chameleon_last_update_wall_time
            >= self.chameleon_t_refresh
        ):
            self._update_adaptive_parameters(current_wall_time, backlog)
        self._grant_drr_deficits(backlog)

        candidates, bucket_by_id = self._ordered_candidates(
            backlog, current_wall_time
        )
        if not candidates:
            if not ctx.defer_prefills:
                self.prefill_capacity_bound = bool(self.waiting)
            return

        candidate_ids = {request.request_id for request in candidates}
        origin_by_id = {request.request_id: "waiting" for request in original_waiting}
        origin_by_id.update(
            {request.request_id: "skipped" for request in original_skipped}
        )
        waiting_remainder = [
            request
            for request in original_waiting
            if request.request_id not in candidate_ids
        ]
        skipped_remainder = [
            request
            for request in original_skipped
            if request.request_id not in candidate_ids
        ]

        before_scheduled_ids = set(ctx.num_scheduled_tokens)
        # 通过临时替换 vLLM 的 waiting 队列来复用基础 admission 循环。
        # 这样 KV-cache、LoRA、encoder、chunked-prefill、spec decode 和
        # connector 检查都仍然只保留在一处实现。
        old_waiting = self.waiting
        old_skipped_waiting = self.skipped_waiting
        try:
            self.waiting = _new_fcfs_queue(candidates)
            self.skipped_waiting = _new_fcfs_queue()
            super()._schedule_waiting_requests(ctx)
            remaining_candidates = list(self.waiting)
            skipped_candidates = list(self.skipped_waiting)
        finally:
            self.waiting = old_waiting
            self.skipped_waiting = old_skipped_waiting

        new_scheduled_ids = set(ctx.num_scheduled_tokens) - before_scheduled_ids
        for request_id in new_scheduled_ids:
            idx = bucket_by_id.get(request_id)
            request = self.requests.get(request_id)
            if idx is not None and request is not None:
                self.chameleon_deficits[idx] -= self._request_cost(request)

        # 恢复没有 admission 成功的候选请求，并保留它们原本来自普通
        # waiting 队列还是 skipped_waiting 队列的信息。
        waiting_restored = [
            request
            for request in remaining_candidates
            if origin_by_id.get(request.request_id) == "waiting"
        ]
        waiting_restored.extend(waiting_remainder)

        skipped_restored = list(skipped_candidates)
        skipped_restored.extend(
            request
            for request in remaining_candidates
            if origin_by_id.get(request.request_id) == "skipped"
        )
        skipped_restored.extend(skipped_remainder)

        self.waiting = _new_fcfs_queue(waiting_restored)
        self.skipped_waiting = _new_fcfs_queue(skipped_restored)
        if not ctx.defer_prefills:
            self.prefill_capacity_bound = bool(self.waiting)
