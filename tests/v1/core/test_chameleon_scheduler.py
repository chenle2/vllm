# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.chameleon_scheduler import ChameleonScheduler
from vllm.v1.core.sched.scheduler import _SchedulingContext

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def create_chameleon_scheduler(**kwargs) -> ChameleonScheduler:
    return create_scheduler(
        scheduler_cls=ChameleonScheduler,
        async_scheduling=False,
        **kwargs,
    )


def test_wrs_bucket_assignment_orders_small_before_large():
    scheduler = create_chameleon_scheduler()
    small = create_requests(1, num_tokens=64, max_tokens=64, req_ids=["small"])[0]
    large = create_requests(1, num_tokens=4096, max_tokens=4096, req_ids=["large"])[0]

    assert scheduler._get_wrs(small) < scheduler._get_wrs(large)
    assert scheduler._bucket_index(small) < scheduler._bucket_index(large)


def test_hrrn_score_increases_with_wait_time_and_urgency():
    scheduler = create_chameleon_scheduler()
    request = create_requests(1, num_tokens=128, max_tokens=32)[0]
    now = time.time()

    request.arrival_time = now - 1.0
    low_score = scheduler._calculate_hrrn_score(request, now)
    request.arrival_time = now - 100.0
    high_score = scheduler._calculate_hrrn_score(request, now)

    assert high_score > low_score


def test_starvation_prepass_admits_overdue_short_request_first():
    scheduler = create_chameleon_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=10000,
        max_model_len=10000,
    )
    short = create_requests(1, num_tokens=64, max_tokens=16, req_ids=["short"])[0]
    large = create_requests(1, num_tokens=4096, max_tokens=4096, req_ids=["large"])[0]
    now = time.time()
    short.arrival_time = now - 1000.0
    large.arrival_time = now

    scheduler.add_request(large)
    scheduler.add_request(short)

    output = scheduler.schedule()

    assert [req.req_id for req in output.scheduled_new_reqs] == ["short"]
    assert [req.request_id for req in scheduler.running] == ["short"]
    assert len(scheduler.waiting) == 1


def test_drr_deficit_not_charged_when_kv_admission_fails(monkeypatch):
    scheduler = create_chameleon_scheduler(max_num_batched_tokens=1024)
    request = create_requests(1, num_tokens=32, max_tokens=16, req_ids=["blocked"])[0]
    scheduler.add_request(request)
    scheduler.chameleon_pq_tokens = [1000] * 4
    scheduler.chameleon_deficits = [0.0] * 4
    scheduler.chameleon_t_refresh = 10_000.0
    bucket = scheduler._bucket_index(request)

    monkeypatch.setattr(
        scheduler.kv_cache_manager, "allocate_slots", Mock(return_value=None)
    )

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {}
    assert scheduler.chameleon_deficits[bucket] == 1000
    assert len(scheduler.waiting) == 1


def test_lora_blocked_request_does_not_consume_drr_deficit():
    scheduler = create_chameleon_scheduler(max_num_batched_tokens=1024)
    request = create_requests(
        1, num_tokens=32, max_tokens=16, req_ids=["lora-blocked"]
    )[0]
    request.lora_request = Mock(lora_int_id=2)
    scheduler.lora_config = Mock(max_loras=1)
    scheduler.add_request(request)
    scheduler.chameleon_pq_tokens = [1000] * 4
    scheduler.chameleon_deficits = [0.0] * 4
    scheduler.chameleon_t_refresh = 10_000.0
    bucket = scheduler._bucket_index(request)

    ctx = _SchedulingContext(
        scheduled_new_reqs=[],
        scheduled_resumed_reqs=[],
        scheduled_running_reqs=[],
        req_to_new_blocks={},
        num_scheduled_tokens={},
        token_budget=1024,
        scheduled_loras={1},
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        encoder_compute_budget=scheduler.max_num_encoder_input_tokens,
        scheduled_timestamp=time.monotonic(),
        defer_prefills=False,
        prefill_scheduled=False,
    )

    scheduler._schedule_waiting_requests(ctx)

    assert ctx.num_scheduled_tokens == {}
    assert scheduler.chameleon_deficits[bucket] == 1000
    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 1
