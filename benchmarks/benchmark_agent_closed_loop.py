# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the VTC SWE-bench/agent closed-loop workload against vLLM.

The VTC artifact stores the agent workload as a pickle shaped like:

    [adapter_dirs, tasks]

where each task has an arrival_time and a serial list of turns.  This
benchmark preserves that closed-loop dependency: turns within the same task are
issued one after another, while different tasks run concurrently according to
their original arrival times.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class AgentResponse:
    # 记录单次 agent turn 请求的关键指标，最后会被 asdict 写入结果文件。
    adapter_dir: str
    prompt_len: int
    output_len: int
    request_latency: float
    first_token_latency: float
    req_time: float
    req_id: str
    success: bool
    status: int = 0
    error: str = ""
    generated_events: int = 0
    generated_bytes: int = 0


def dummy_prompt(prompt_len: int) -> str:
    # benchmark 只关心输入长度和调度形态，这里用重复 token 近似 prompt。
    return "Hello " * prompt_len


def percentile(values: list[float], pct: float) -> float:
    # 使用线性插值计算百分位，避免样本较少时只能落在离散点上。
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def normalize_server_url(server: str, endpoint: str) -> str:
    # 统一处理 server 和 endpoint 两侧的斜杠，避免拼出双斜杠或漏斜杠。
    return server.rstrip("/") + "/" + endpoint.lstrip("/")


def build_payload(
    *,
    backend: str,
    model: str,
    prompt: str,
    output_len: int,
    ignore_eos: bool,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    # completions 和 chat completions 的请求体结构不同，但都使用流式输出。
    if backend == "openai-chat":
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_completion_tokens": output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    if ignore_eos:
        # 压测时可忽略 EOS，让服务尽量生成到 max_tokens/max_completion_tokens。
        payload["ignore_eos"] = True
    # 允许调用方透传 vLLM/OpenAI 兼容接口支持的额外字段，例如 LoRA 参数。
    payload.update(extra_body)
    return payload


def extract_text_from_sse_payload(data: dict[str, Any], backend: str) -> str:
    # 从 OpenAI 兼容的 SSE 增量响应中抽取本次 chunk 产生的文本。
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if backend == "openai-chat":
        return choice.get("delta", {}).get("content") or ""
    return choice.get("text") or ""


async def iter_sse_data(response: aiohttp.ClientResponse):
    # OpenAI 流式接口使用 server-sent events，每个事件以空行分隔。
    buffer = ""
    async for chunk in response.content.iter_any():
        buffer += chunk.decode("utf-8", errors="ignore")
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            for line in raw_event.splitlines():
                line = line.strip()
                # 跳过空行、注释行以及非 data 字段，只向上层返回 payload 内容。
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                yield line.removeprefix("data:").strip()
    # 某些服务端断开时可能留下未以空行收尾的最后一段 data。
    if buffer.strip().startswith("data:"):
        yield buffer.strip().removeprefix("data:").strip()


async def send_openai_request(
    *,
    session: aiohttp.ClientSession,
    url: str,
    backend: str,
    model: str,
    adapter_dir: str,
    req_time: float,
    req_id: str,
    prompt_len: int,
    output_len: int,
    cutoff_tokens: int,
    trigger_mode: str,
    ignore_eos: bool,
    extra_body: dict[str, Any],
    trigger_event: asyncio.Event,
) -> AgentResponse:
    # 单次 OpenAI 兼容请求：发送 prompt、消费 SSE，并记录延迟与输出规模。
    prompt = dummy_prompt(prompt_len)
    payload = build_payload(
        backend=backend,
        model=model,
        prompt=prompt,
        output_len=output_len,
        ignore_eos=ignore_eos,
        extra_body=extra_body,
    )
    headers = {
        "Authorization": "Bearer EMPTY",
        "Content-Type": "application/json",
        "User-Agent": "Agent Closed Loop Benchmark",
        "x-request-id": req_id,
    }

    start = time.perf_counter()
    # 首 token 延迟用收到第一个非空文本 chunk 的时间近似。
    first_token_latency: float | None = None
    generated_events = 0
    generated_bytes = 0
    status = 0
    error = ""

    try:
        async with session.post(url, headers=headers, json=payload) as response:
            status = response.status
            if response.status != 200:
                error = await response.text()
                return AgentResponse(
                    adapter_dir=adapter_dir,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    request_latency=time.perf_counter() - start,
                    first_token_latency=-1,
                    req_time=req_time,
                    req_id=req_id,
                    success=False,
                    status=status,
                    error=error,
                )

            async for event_data in iter_sse_data(response):
                if event_data == "[DONE]":
                    break
                try:
                    data = json.loads(event_data)
                except json.JSONDecodeError:
                    continue
                text = extract_text_from_sse_payload(data, backend)
                if text:
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - start
                    generated_events += 1
                    generated_bytes += len(text.encode("utf-8"))

                    # cutoff_tokens 用来释放同一 task 的下一段模拟 IO/turn。
                    # events 模式按 chunk 数近似，bytes 模式按 UTF-8 字节数粗略换算。
                    if trigger_mode == "events":
                        should_trigger = generated_events >= cutoff_tokens
                    else:
                        should_trigger = generated_bytes >= cutoff_tokens * 4
                    if should_trigger and not trigger_event.is_set():
                        trigger_event.set()
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures.
        error = repr(exc)
        return AgentResponse(
            adapter_dir=adapter_dir,
            prompt_len=prompt_len,
            output_len=output_len,
            request_latency=time.perf_counter() - start,
            first_token_latency=-1,
            req_time=req_time,
            req_id=req_id,
            success=False,
            status=status,
            error=error,
            generated_events=generated_events,
            generated_bytes=generated_bytes,
        )
    finally:
        # 不论成功、失败或异常，都要释放等待方，避免闭环任务永久阻塞。
        if not trigger_event.is_set():
            trigger_event.set()

    latency = time.perf_counter() - start
    return AgentResponse(
        adapter_dir=adapter_dir,
        prompt_len=prompt_len,
        output_len=output_len,
        request_latency=latency,
        first_token_latency=first_token_latency if first_token_latency is not None else -1,
        req_time=req_time,
        req_id=req_id,
        success=first_token_latency is not None,
        status=status,
        error=error,
        generated_events=generated_events,
        generated_bytes=generated_bytes,
    )


async def run_agent_task(
    *,
    session: aiohttp.ClientSession,
    url: str,
    backend: str,
    model: str,
    model_from_adapter: bool,
    global_start_time: float,
    task: dict[str, Any],
    trigger_mode: str,
    ignore_eos: bool,
    extra_body: dict[str, Any],
    responses: list[AgentResponse],
) -> None:
    # 按 trace 中的 arrival_time 恢复不同 task 的到达时间。
    await asyncio.sleep(max(0.0, global_start_time + task["arrival_time"] - time.time()))

    # 同一个 task 内的 turns 保持闭环依赖：上一轮达到 cutoff 或结束后才进入下一轮。
    for idx, turn in enumerate(task["turns"]):
        dispatch_time = time.time() - global_start_time
        req_id = f"{task['task_id']}_turn_{idx}"
        trigger_event = asyncio.Event()
        adapter_dir = task.get("adapter_dir") or ""
        request_model = adapter_dir if model_from_adapter and adapter_dir else model

        request_task = asyncio.create_task(
            send_openai_request(
                session=session,
                url=url,
                backend=backend,
                model=request_model,
                adapter_dir=adapter_dir,
                req_time=dispatch_time,
                req_id=req_id,
                prompt_len=int(turn["prompt_len"]),
                output_len=int(turn["output_len"]),
                cutoff_tokens=int(turn.get("cutoff_tokens", turn["output_len"])),
                trigger_mode=trigger_mode,
                ignore_eos=ignore_eos,
                extra_body=extra_body,
                trigger_event=trigger_event,
            )
        )

        await trigger_event.wait()
        if turn.get("has_io") and turn.get("io_time", 0) > 0:
            # 模拟 agent 在模型输出后执行工具调用或环境交互的耗时。
            await asyncio.sleep(float(turn["io_time"]))

        responses.append(await request_task)


async def replay_trace(args: argparse.Namespace, tasks: list[dict[str, Any]]):
    # 一个 task 对应一个 asyncio 任务；task 内部串行，task 之间并发。
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.max_connections)
    responses: list[AgentResponse] = []
    url = normalize_server_url(args.server, args.endpoint)
    extra_body = json.loads(args.extra_body) if args.extra_body else {}

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, trust_env=True
    ) as session:
        start_time = time.time()
        # 并发启动所有 trace task，由 run_agent_task 自己 sleep 到原始到达时间。
        task_handles = [
            asyncio.create_task(
                run_agent_task(
                    session=session,
                    url=url,
                    backend=args.backend,
                    model=args.model,
                    model_from_adapter=args.model_from_adapter,
                    global_start_time=start_time,
                    task=task,
                    trigger_mode=args.trigger_mode,
                    ignore_eos=args.ignore_eos,
                    extra_body=extra_body,
                    responses=responses,
                )
            )
            for task in tasks
        ]
        await asyncio.gather(*task_handles)
        total_time = time.time() - start_time
    return responses, total_time


def summarize(
    *,
    responses: list[AgentResponse],
    total_time: float,
    trace_path: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # 只用成功请求计算延迟分布，失败请求仍保留在 responses 明细里。
    successes = [response for response in responses if response.success]
    latencies = [response.request_latency for response in successes]
    ttfts = [response.first_token_latency for response in successes]

    per_adapter: dict[str, list[float]] = {}
    for response in successes:
        # adapter_dir 维度用于观察不同 LoRA/adapter 的平均请求延迟。
        per_adapter.setdefault(response.adapter_dir, []).append(response.request_latency)

    summary = {
        "config": {
            "backend": args.backend,
            "server": args.server,
            "endpoint": args.endpoint,
            "model": args.model,
            "trace": trace_path,
            "trigger_mode": args.trigger_mode,
            "model_from_adapter": args.model_from_adapter,
        },
        "result": {
            "total_time": total_time,
            "num_requests": len(responses),
            "num_success": len(successes),
            "num_failed": len(responses) - len(successes),
            "throughput": len(successes) / total_time if total_time > 0 else 0.0,
            "avg_latency": statistics.mean(latencies) if latencies else 0.0,
            "p50_latency": percentile(latencies, 50),
            "p90_latency": percentile(latencies, 90),
            "avg_first_token_latency": statistics.mean(ttfts) if ttfts else 0.0,
            "p50_first_token_latency": percentile(ttfts, 50),
            "p90_first_token_latency": percentile(ttfts, 90),
            "per_adapter_avg_latency": {
                adapter: statistics.mean(values)
                for adapter, values in sorted(per_adapter.items())
            },
            "responses": [asdict(response) for response in responses],
        },
    }
    return summary


def load_tasks(trace_path: Path, limit_tasks: int | None) -> tuple[list[Any], list[dict[str, Any]]]:
    # VTC trace 的 pickle 约定为 [adapter_dirs, tasks]。
    with trace_path.open("rb") as file:
        obj = pickle.load(file)
    if not isinstance(obj, list) or len(obj) != 2:
        raise ValueError("Expected trace pickle to contain [adapter_dirs, tasks].")
    adapter_dirs, tasks = obj
    if limit_tasks is not None:
        tasks = tasks[:limit_tasks]
    return adapter_dirs, tasks


def parse_args() -> argparse.Namespace:
    # CLI 参数保持接近 OpenAI 兼容 benchmark 的常见用法。
    parser = argparse.ArgumentParser(
        description="Replay VTC's closed-loop agent workload against vLLM."
    )
    parser.add_argument(
        "--trace",
        default="../VTC-artifact-llama3/agent_closed_loop.pkl",
        help="Path to VTC agent_closed_loop.pkl.",
    )
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument(
        "--backend",
        choices=["openai-completions", "openai-chat"],
        default="openai-completions",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-from-adapter",
        action="store_true",
        help="Use each task's adapter_dir as the OpenAI model name for LoRA runs.",
    )
    parser.add_argument("--output", default="agent_closed_loop_results.jsonl")
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument(
        "--trigger-mode",
        choices=["events", "bytes"],
        default="events",
        help="How to approximate cutoff_tokens for releasing simulated IO.",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON object merged into every OpenAI request body.",
    )
    parser.add_argument("--timeout", type=float, default=3 * 3600)
    parser.add_argument("--max-connections", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace)
    _, tasks = load_tasks(trace_path, args.limit_tasks)

    # 先打印 trace 规模，便于确认本次压测是否加载了预期数据。
    all_turns = [turn for task in tasks for turn in task["turns"]]
    print(
        "Loaded agent trace:",
        f"tasks={len(tasks)}",
        f"turns={len(all_turns)}",
        f"max_prompt_len={max(turn['prompt_len'] for turn in all_turns)}",
        f"max_output_len={max(turn['output_len'] for turn in all_turns)}",
    )

    started = time.time()
    responses, total_time = asyncio.run(replay_trace(args, tasks))
    summary = summarize(
        responses=responses,
        total_time=total_time,
        trace_path=str(trace_path),
        args=args,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as file:
        # 追加 JSONL，方便同一个输出文件保存多组压测结果。
        file.write(json.dumps(summary) + "\n")

    result = summary["result"]
    print(f"Wall time: {time.time() - started:.2f} s")
    print(f"Total time: {result['total_time']:.2f} s")
    print(f"Requests: {result['num_success']}/{result['num_requests']} succeeded")
    print(f"Throughput: {result['throughput']:.2f} requests/s")
    print(f"Average latency: {result['avg_latency']:.2f} s")
    print(f"P90 latency: {result['p90_latency']:.2f} s")
    print(f"Average TTFT: {result['avg_first_token_latency']:.2f} s")
    print(f"P90 TTFT: {result['p90_first_token_latency']:.2f} s")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
