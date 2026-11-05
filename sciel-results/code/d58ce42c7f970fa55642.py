"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


# Deadline guard: track start time to ensure we finish within budget
_DISPATCH_START = None
_DEADLINE_MARGIN = 10.0  # seconds before the 120s limit


def _ensure_deadline():
    """Check if we're approaching the deadline; return True if we should return immediately."""
    global _DISPATCH_START
    if _DISPATCH_START is None:
        _DISPATCH_START = _time.time()
    elapsed = _time.time() - _DISPATCH_START
    return elapsed > (120.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # If only one operation, return it
    if len(ready_operations) == 1:
        return 0

    # Deadline guard: if we're running out of time, fall back to a simple rule
    if _ensure_deadline():
        # Simple fallback: pick operation with max remaining_work
        max_work = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_work:
                max_work = rw
                best_idx = i
        return best_idx

    n = len(ready_operations)
    scores = np.zeros(n, dtype=np.float64)

    # Pre-compute machine contention: for each machine, count how many ready ops
    # target it and sum their durations. This gives a proxy for machine load.
    machine_load = {}  # machine -> (count, total_duration)
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine, duration in machine_options.items():
            if machine not in machine_load:
                machine_load[machine] = [0, 0.0]
            machine_load[machine][0] += 1
            machine_load[machine][1] += duration

    for i, op in enumerate(ready_operations):
        # Get machine options: {machine: duration}
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0)

        if not machine_options:
            # Fallback: use remaining_work
            scores[i] = remaining_work
            continue

        durations = list(machine_options.values())
        max_duration = max(durations)
        min_duration = min(durations)

        # Constraint ratio: how much more constrained is this op vs its best alternative
        constraint_ratio = max_duration / (min_duration + 1e-9)

        # Machine contention score: for each machine this op can run on,
        # compute a "pressure" metric. Higher pressure means the machine is
        # heavily loaded, so scheduling an op there reduces fragmentation.
        # We use: (count * avg_duration) as a proxy for total machine busy time
        machine_pressure = 0.0
        for machine, duration in machine_options.items():
            if machine in machine_load:
                count, total_dur = machine_load[machine]
                # Pressure: how busy is this machine relative to this op's duration?
                # A machine with many ops and high total duration is congested
                machine_pressure = max(machine_pressure, count * (total_dur / (count + 1e-9)))

        # Normalize machine_pressure to be comparable across ops
        # Use a soft normalization: pressure / (pressure + 1) to bound it
        pressure_factor = machine_pressure / (machine_pressure + max_duration + 1e-9)

        # Combine scores:
        # 1. remaining_work: LRPT component (high remaining work = high priority)
        # 2. constraint_ratio: ops with severe bottleneck should go first
        # 3. pressure_factor: ops targeting congested machines should go first
        #    (reduces idle time on busy machines)
        # 4. Shortest duration bonus: if an op has a short duration on a congested
        #    machine, it's a good "filler" to keep that machine busy

        # Weighted combination
        w_work = 1.0
        w_constraint = 0.3
        w_pressure = 0.4

        scores[i] = (
            w_work * remaining_work
            + w_constraint * constraint_ratio * remaining_work
            + w_pressure * pressure_factor * remaining_work
        )

    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
