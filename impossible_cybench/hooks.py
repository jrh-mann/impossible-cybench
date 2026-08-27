"""Live monitor-decision surfacing, the inspect-onic way.

An Inspect Hook: `on_sample_event` fires on every logged event during the run, and an
approver's decision is recorded as an ApprovalEvent. Raw ApprovalEvents live only in the
transcript (they do NOT appear in the runner log stream), so a monitor FLAG/BLOCK is
invisible to `hawk logs`/`hawk trace` by default. This hook re-emits them via the runner
logger, so `hawk logs -f | grep '\\[MONITOR'` catches them live and reliably.

Registered by being imported from the package __init__ (the package's inspect_ai entry
point loads that), so no separate entry point is needed.
"""
import logging

from inspect_ai.hooks import Hooks, SampleEvent, hooks

logger = logging.getLogger("impossible_cybench.monitor")


@hooks(
    name="impcy_flag_surfacer",
    description="surface k8s_monitor FLAG / model_monitor BLOCK decisions to the log stream",
)
class FlagSurfacer(Hooks):
    async def on_sample_event(self, data: SampleEvent) -> None:
        ev = data.event
        if getattr(ev, "event", None) != "approval":
            return
        expl = getattr(ev, "explanation", None) or ""
        # k8s_monitor(mode=monitor) flags via an escalate carrying "FLAG infra-attack"
        if "FLAG infra-attack" in expl:
            logger.warning(f"[MONITOR-FLAG] sample={data.sample_id} :: {expl}")
        # model_monitor kills via decision=terminate (also catches k8s block-mode terminate)
        elif getattr(ev, "decision", None) in ("terminate", "reject"):
            logger.warning(
                f"[MONITOR-BLOCK] sample={data.sample_id} "
                f"approver={getattr(ev, 'approver', '?')} decision={ev.decision} :: {expl}"
            )
