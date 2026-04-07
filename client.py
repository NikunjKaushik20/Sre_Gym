"""Client for connecting to SRE-Gym environment via the OpenEnv client protocol."""
from typing import Optional
from openenv.core.env_client import EnvClient
from .models import SREAction, SREObservation, SREState


class SREGymEnv(EnvClient[SREAction, SREObservation, SREState]):
    """SRE-Gym environment client with typed helpers for every action type.

    Usage:
        import asyncio
        from sre_gym import SREGymEnv, SREAction

        async def run():
            async with SREGymEnv(base_url="http://localhost:8000") as env:
                obs = await env.reset(task_id="task_easy_1")
                obs = await env.query_logs("redis")
                obs = await env.submit_diagnosis("redis", "memory_leak")
                obs = await env.apply_remediation("restart_redis")
                obs = await env.submit_postmortem(
                    root_cause="memory_leak",
                    affected_services=["redis", "api-gateway"],
                    timeline_steps=3,
                    prevention_steps=["redis_memory_alert", "eviction_policy"],
                )

        asyncio.run(run())
    """

    def _step_payload(self, action: SREAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> "StepResult[SREObservation]":
        from openenv.core.client_types import StepResult
        obs_data = payload.get("observation", {})
        reward = payload.get("reward", 0.0)
        done = payload.get("done", False)
        return StepResult(
            observation=SREObservation(**obs_data),
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> SREState:
        return SREState(**payload)

    async def query_logs(self, service: str, window: str = "last_5min") -> SREObservation:
        """Query logs for a specific service."""
        return await self.step(SREAction(
            action_type="query_logs",
            payload={"service": service, "window": window},
        ))

    async def query_metrics(self, service: str) -> SREObservation:
        """Query current metrics for a specific service."""
        return await self.step(SREAction(
            action_type="query_metrics",
            payload={"service": service},
        ))

    async def submit_diagnosis(
        self, suspected_service: str, suspected_cause: str
    ) -> SREObservation:
        """Submit a root cause diagnosis."""
        return await self.step(SREAction(
            action_type="submit_diagnosis",
            payload={
                "suspected_service": suspected_service,
                "suspected_cause":   suspected_cause,
            },
        ))

    async def apply_remediation(self, playbook_id: str) -> SREObservation:
        """Apply a remediation playbook."""
        return await self.step(SREAction(
            action_type="apply_remediation",
            payload={"playbook_id": playbook_id},
        ))

    async def escalate(self, team: str, severity: str = "high") -> SREObservation:
        """Escalate the incident to an on-call team."""
        return await self.step(SREAction(
            action_type="escalate",
            payload={"team": team, "severity": severity},
        ))

    async def submit_postmortem(
        self,
        root_cause: str,
        affected_services: list,
        timeline_steps: int,
        prevention_steps: list,
    ) -> SREObservation:
        """Submit a full postmortem to close the incident."""
        return await self.step(SREAction(
            action_type="submit_postmortem",
            payload={
                "root_cause":        root_cause,
                "affected_services": affected_services,
                "timeline_steps":    timeline_steps,
                "prevention_steps":  prevention_steps,
            },
        ))

    async def close_incident(self) -> SREObservation:
        """Explicitly close the incident (bail-out — penalised if not fully resolved)."""
        return await self.step(SREAction(
            action_type="close_incident",
            payload={},
        ))
