"""Job-Shop dispatch rule: critical-path urgency + machine-load-aware selection."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE = 120.0
_START = _time.monotonic()


def _machine_load(machine_options, time):
    """Estimate total work currently queued on the given machine options.

    We approximate contention by summing the durations of all ready ops that
    could use the same machine (a proxy for load). Lower = less congested.
    """
    loads = {}
    for mo in machine_options:
        for m in mo:
            loads[m] = loads.get(m, 0.0)
    return loads


def dispatch(ready_operations, time):
    now = _time.monotonic()
    if now > _DEADLINE - 0.5:
        # Safety margin: fall back to the simplest valid choice.
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Precompute per-machine aggregate load (proxy for contention).
    # Sum of durations of all ready ops that can use each machine.
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_load[m] = machine_load.get(m, 0.0) + d

    best_idx = 0
    best_key = None

    for i, op in enumerate(ready_operations):
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0.0)
        mo = op["machine_options"]

        # Critical path proxy: remaining_work (sum of durations of this job's
        # remaining ops). Tie-break by remaining_ops (more ops = longer path).
        # Urgency = remaining_work + small * remaining_ops.
        urgency = remaining_work + 0.1 * remaining_ops

        # Choose the machine that minimizes estimated completion time:
        # duration + (load on that machine / (n-1) as a contention proxy).
        # We divide load by (n-1) to normalize per-op contribution.
        best_est = None
        for m, d in mo.items():
            load = machine_load.get(m, 0.0)
            # contention proxy: load from OTHER ops on this machine
            contention = (load - d) / max(n - 1, 1)
            est = d + 0.5 * contention
            if best_est is None or est < best_est:
                best_est = est

        # Combine: prioritize high urgency, and among similar urgency,
        # prefer the op whose best machine is least congested.
        # key = (urgency descending, best_est ascending)
        # We'll compare tuples: higher urgency better, lower est better.
        key = (urgency, -best_est)

        if best_key is None or key > best_key:
            best_key = key
            best_idx = i

    # Check deadline again before returning.
    if _time.monotonic() > _DEADLINE - 0.2:
        return 0

    return int(best_idx)
# EVOLVE-BLOCK-END
