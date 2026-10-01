#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate and benchmark the Zenoh Remote Inference API.

This script supports:
  1. Standalone loopback evaluation (runs both server and client benchmarks).
  2. Running as a dedicated Zenoh inference server.
  3. Running as a dedicated Zenoh inference client.

Usage examples:
  # 1. Quick local benchmark with synthetic observation & mock model:
  uv run python examples/runtime/zenoh_inference.py --mode loopback

  # 2. Run benchmark with custom chunk size and request count:
  uv run python examples/runtime/zenoh_inference.py --mode loopback --requests 50 --chunk-size 32

  # 3. Dedicated server mode (listens on Zenoh key / endpoint):
  uv run python examples/runtime/zenoh_inference.py --mode server --endpoint tcp/127.0.0.1:7447

  # 4. Dedicated client mode:
  uv run python examples/runtime/zenoh_inference.py --mode client --endpoint tcp/127.0.0.1:7447
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any

import numpy as np

from physicalai.runtime.execution.queue import ChunkedActionQueue
from physicalai.runtime.execution.zenoh_remote_execution import (
    ZenohRemoteExecution,
    ZenohRemoteInferenceServer,
)


class SyntheticInferenceModel:
    """Synthetic model for latency and throughput evaluation without heavy weights."""

    def __init__(self, action_dim: int = 6, chunk_size: int = 16, simulated_inference_s: float = 0.0) -> None:
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.simulated_inference_s = simulated_inference_s
        self.reset_count = 0
        self.predict_count = 0

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        if self.simulated_inference_s > 0:
            time.sleep(self.simulated_inference_s)
        self.predict_count += 1
        # Generate predictable synthetic trajectory
        t = np.linspace(0, 1, self.chunk_size, dtype=np.float32)[:, None]
        base = np.arange(self.action_dim, dtype=np.float32)[None, :]
        return np.sin(t + base)

    def reset(self) -> None:
        self.reset_count += 1


def make_sample_observation(image_h: int = 224, image_w: int = 224, joint_dim: int = 6) -> dict[str, Any]:
    """Generate sample observation containing RGB image and joint positions."""
    return {
        "overhead": np.random.randint(0, 256, (image_h, image_w, 3), dtype=np.uint8),
        "joint_positions": np.random.randn(joint_dim).astype(np.float32),
    }


def run_server(key_expr: str, endpoint: str | None, simulated_latency_ms: float, chunk_size: int) -> None:
    print(f"[Server] Starting Zenoh Remote Inference Server on key '{key_expr}'...")
    if endpoint:
        print(f"[Server] Listening on endpoint: {endpoint}")
    model = SyntheticInferenceModel(
        chunk_size=chunk_size,
        simulated_inference_s=simulated_latency_ms / 1000.0,
    )
    server = ZenohRemoteInferenceServer(
        model=model,  # type: ignore[arg-type]
        key_expr=key_expr,
        listen_endpoint=endpoint,
    )
    try:
        print("[Server] Server is running. Press Ctrl+C to terminate.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
    finally:
        server.stop()
        print("[Server] Stopped.")


def run_client(
    key_expr: str,
    endpoint: str | None,
    num_requests: int,
    warmup_requests: int,
) -> None:
    print(f"[Client] Connecting to Zenoh key '{key_expr}'...")
    if endpoint:
        print(f"[Client] Target endpoint: {endpoint}")

    execution = ZenohRemoteExecution(
        endpoint=endpoint,
        key_expr=key_expr,
        request_timeout_s=10.0,
    )
    queue = ChunkedActionQueue()

    # Initialize execution
    dummy_model = SyntheticInferenceModel()
    execution.start(dummy_model, queue)  # type: ignore[arg-type]

    sample_obs = make_sample_observation()

    try:
        print(f"[Client] Running warmup ({warmup_requests} request(s))...")
        for _ in range(warmup_requests):
            execution.warmup(sample_obs)
        print(f"[Client] Warmup successful. Chunk size: {execution.chunk_size}")

        print(f"[Client] Resetting remote policy...")
        execution.reset(reset_model=True)

        print(f"[Client] Benchmarking {num_requests} inference requests...")
        latencies_ms: list[float] = []

        for i in range(num_requests):
            # Consume queue items to simulate robot popping actions
            while queue.remaining > 0:
                queue.pop()

            start_t = time.perf_counter()
            execution.maybe_request(sample_obs)

            # Wait for inference thread to push the new chunk
            deadline = time.perf_counter() + 5.0
            while queue.remaining == 0:
                if time.perf_counter() > deadline:
                    raise TimeoutError(f"Request {i + 1} timed out waiting for action chunk")
                time.sleep(0.0005)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(elapsed_ms)

        # Print statistics
        latencies = np.array(latencies_ms)
        print("\n" + "=" * 50)
        print(" Zenoh Remote Inference Benchmark Results")
        print("=" * 50)
        print(f"Total Requests  : {num_requests}")
        print(f"Total Time      : {latencies.sum() / 1000.0:.3f} s")
        print(f"Throughput      : {num_requests / (latencies.sum() / 1000.0):.2f} req/s")
        print(f"Latency Mean    : {latencies.mean():.2f} ms")
        print(f"Latency Median  : {np.median(latencies):.2f} ms")
        print(f"Latency Min     : {latencies.min():.2f} ms")
        print(f"Latency Max     : {latencies.max():.2f} ms")
        print(f"Latency P95     : {np.percentile(latencies, 95):.2f} ms")
        print(f"Latency P99     : {np.percentile(latencies, 99):.2f} ms")
        print("=" * 50 + "\n")

    finally:
        execution.stop()
        print("[Client] Disconnected.")


def run_loopback(
    key_expr: str,
    requests: int,
    warmup: int,
    chunk_size: int,
    simulated_latency_ms: float,
) -> None:
    print(f"[Loopback] Setting up in-process Zenoh server & client on key '{key_expr}'...")
    model = SyntheticInferenceModel(chunk_size=chunk_size, simulated_inference_s=simulated_latency_ms / 1000.0)
    server = ZenohRemoteInferenceServer(model=model, key_expr=key_expr)  # type: ignore[arg-type]

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Small delay for queryable registration
    time.sleep(0.2)

    try:
        run_client(
            key_expr=key_expr,
            endpoint=None,
            num_requests=requests,
            warmup_requests=warmup,
        )
    finally:
        print("[Loopback] Stopping server...")
        server.stop()
        server_thread.join(timeout=3.0)
        print("[Loopback] Completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Zenoh Remote Inference API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["loopback", "server", "client"],
        default="loopback",
        help="Evaluation mode: loopback (both in one process), server, or client",
    )
    parser.add_argument(
        "--key",
        default="physicalai/inference/eval",
        help="Zenoh key expression for inference query/reply",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Zenoh network endpoint (e.g., tcp/127.0.0.1:7447 or udp/127.0.0.1:7447)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=25,
        help="Number of inference requests to benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup requests",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk size returned by the model",
    )
    parser.add_argument(
        "--simulated-latency-ms",
        type=float,
        default=0.0,
        help="Simulated server-side model inference time in milliseconds",
    )

    args = parser.parse_args()

    if args.mode == "server":
        run_server(
            key_expr=args.key,
            endpoint=args.endpoint,
            simulated_latency_ms=args.simulated_latency_ms,
            chunk_size=args.chunk_size,
        )
    elif args.mode == "client":
        run_client(
            key_expr=args.key,
            endpoint=args.endpoint,
            num_requests=args.requests,
            warmup_requests=args.warmup,
        )
    else:
        run_loopback(
            key_expr=args.key,
            requests=args.requests,
            warmup=args.warmup,
            chunk_size=args.chunk_size,
            simulated_latency_ms=args.simulated_latency_ms,
        )


if __name__ == "__main__":
    main()
