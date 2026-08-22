"""Optional load test for the FastAPI inference service.

This measures *serving* latency and throughput. It says nothing about model
quality - keep it mentally separate from everything in artifacts/evaluation/.

    uvicorn src.api:app --port 8000            # terminal 1
    locust -f performance/locustfile.py --host http://localhost:8000   # terminal 2

Headless:

    locust -f performance/locustfile.py --host http://localhost:8000 \
        --headless --users 5 --spawn-rate 1 --run-time 1m
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Generic prompts - no personal data is embedded in this repo's load test.
PROMPTS = [
    "What is his full name?",
    "Where did he study?",
    "What does he do for work?",
    "What are his technical skills?",
    "What projects has he worked on?",
    "What are his hobbies?",
    "Which company employs him?",
    "What did he study at university?",
]


class InferenceUser(HttpUser):
    """One simulated client hitting /generate."""

    wait_time = between(1, 3)

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="GET /health")

    @task(9)
    def generate(self) -> None:
        payload = {"prompt": random.choice(PROMPTS), "max_new_tokens": 64, "temperature": 0.0}
        with self.client.post(
            "/generate", json=payload, name="POST /generate", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            body = response.json()
            if not body.get("response"):
                response.failure("empty response field")
            else:
                response.success()
