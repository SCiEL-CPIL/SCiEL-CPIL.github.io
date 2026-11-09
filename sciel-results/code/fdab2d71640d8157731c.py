"""Job-Shop Scheduling dispatcher with Greedy Lookahead Simulation and multi-criteria scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill


def _deadline_exceeded():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a fast simple rule
    if _deadline_exceeded():
        # Fast fallback: pick operation with smallest min duration (SPT)
        min_durs = []
        for op in ready_operations:
            mo = op.get("machine_options", {})
            if mo:
                min_durs.append(min(mo.values()))
            else:
                min_durs.append(1.0)
        return int(np.argmin(min_durs))

    # Extract features
    remaining_work = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    max_durations = np.empty(n, dtype=np.float64)
    avg_durations = np.empty(n, dtype=np.float64)
    machine_options_list = []

    for i, op in enumerate(ready_operations):
        remaining_work[i] = op.get("remaining_work", 0.0)
        remaining_ops[i] = op.get("remaining_ops", 1.0)
        mo = op.get("machine_options", {})
        if mo:
            durs = list(mo.values())
            min_durations[i] = min(durs)
            max_durations[i] = max(durs)
            avg_durations[i] = sum(durs) / len(durs)
            machine_options_list.append(mo)
        else:
            min_durations[i] = 1.0
            max_durations[i] = 1.0
            avg_durations[i] = 1.0
            machine_options_list.append({0: 1.0})

    # --- Greedy Lookahead Simulation: minimize projected makespan ---
    # For each candidate operation i, simulate scheduling it (on its best machine),
    # then estimate the makespan of the remaining ready set.
    # Select the operation that minimizes the projected future horizon.

    all_machines = set()
    for mo in machine_options_list:
        all_machines.update(mo.keys())

    if not all_machines:
        all_machines = {0}

    # Build machine load estimates
    machine_load = {m: 0.0 for m in all_machines}
    for mo in machine_options_list:
        for m, dur in mo.items():
            machine_load[m] += dur

    # For each operation, pre-compute the best machine (least loaded)
    op_best_machine = [None] * n
    op_best_load = np.empty(n, dtype=np.float64)

    for i in range(n):
        mo = machine_options_list[i]
        best_m = None
        best_load = float('inf')
        for m, dur in mo.items():
            if machine_load[m] < best_load:
                best_load = machine_load[m]
                best_m = m
        op_best_machine[i] = best_m
        op_best_load[i] = best_load

    # For each candidate i, simulate scheduling it and compute projected makespan
    projected_makespans = np.empty(n, dtype=np.float64)

    for i in range(n):
        m_i = op_best_machine[i]
        dur_i = machine_options_list[i].get(m_i, 1.0)

        # After scheduling i, update machine loads
        # The load on m_i decreases by dur_i (since i is now scheduled)
        new_machine_load = dict(machine_load)
        new_machine_load[m_i] -= dur_i
        if new_machine_load[m_i] < 0:
            new_machine_load[m_i] = 0.0

        # Estimate the makespan of the remaining ready set
        # For each remaining operation j, estimate its completion time
        # using a simple list scheduling heuristic
        max_completion = 0.0

        # Track estimated availability time for each machine
        machine_avail = {m: 0.0 for m in all_machines}

        # Sort remaining operations by some priority (e.g., remaining work descending)
        remaining_indices = [j for j in range(n) if j != i]

        # Use a simple greedy list scheduling: process operations in order of
        # decreasing remaining work, assign each to its least-loaded machine
        remaining_indices_sorted = sorted(
            remaining_indices,
            key=lambda j: -remaining_work[j]
        )

        for j in remaining_indices_sorted:
            mo_j = machine_options_list[j]
            # Find the machine with minimum estimated completion time for op j
            best_m_j = None
            best_finish = float('inf')
            for m, dur_j in mo_j.items():
                # This op can start after machine m is available
                start_time = machine_avail[m]
                finish_time = start_time + dur_j
                if finish_time < best_finish:
                    best_finish = finish_time
                    best_m_j = m

            if best_m_j is not None:
                machine_avail[best_m_j] = best_finish
                if best_finish > max_completion:
                    max_completion = best_finish

        # Projected makespan includes the time already elapsed plus the projected future
        # But since we're comparing relative values, we can just use max_completion
        # Add a small bias for the operation we just scheduled (its duration)
        projected_makespans[i] = max_completion + dur_i * 0.1

    # --- Secondary criteria to break ties and add robustness ---
    def _norm(arr):
        mn = arr.min()
        mx = arr.max()
        if mx <= mn + 1e-12:
            return np.full_like(arr, 0.5, dtype=np.float64)
        return (arr - mn) / (mx - mn)

    # WSPT: work per unit time on the operation (urgency signal)
    with np.errstate(divide='ignore', invalid='ignore'):
        wspt = remaining_work / np.maximum(min_durations, 1e-6)
    n_wspt = _norm(wspt)  # higher = more urgent

    # Closeness to completion: fewer remaining ops = closer to done
    n_close = 1.0 - _norm(remaining_ops)

    # SPT: shorter min duration preferred
    n_spt = 1.0 - _norm(min_durations)

    # Normalize projected makespans (lower is better, so invert)
    n_proj = 1.0 - _norm(projected_makespans)

    # Weighted combination: prioritize projected makespan minimization, then urgency
    w_proj = 0.55
    w_wspt = 0.20
    w_close = 0.15
    w_spt = 0.10

    score = (w_proj * n_proj +
             w_wspt * n_wspt +
             w_close * n_close +
             w_spt * n_spt)

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
