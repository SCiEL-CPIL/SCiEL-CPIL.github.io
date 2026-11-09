"""Job-Shop Scheduling dispatcher with dynamic bottleneck identification."""
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

    # Deadline guard: fast fallback
    if _deadline_exceeded():
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

    # --- Dynamic bottleneck identification strategy ---
    # For each ready operation, consider assigning it to its least-loaded machine.
    # The "load" of a machine is estimated by the average duration of operations
    # that can use that machine among all ready operations.
    # We want to maximize the minimum slack time across all ready operations.
    #
    # Slack for an operation i assigned to machine m:
    #   slack_i = (time when machine m would be free) - (current time)
    # We estimate machine free time by summing the durations of all ready ops
    # that would use that machine (in some order). To simplify, we use the
    # total work on each machine as a proxy for when it becomes free.
    #
    # Strategy: For each operation, find the machine that minimizes its
    # completion time (i.e., machine with least accumulated load + shortest duration).
    # Then compute the slack for all operations. Pick the operation that, when
    # scheduled, maximizes the minimum slack across all ready operations.

    # Build machine load estimates
    # machine_load[m] = sum of min_durations of all ready ops that can use machine m
    # Actually, use the specific duration for that machine if available
    all_machines = set()
    for mo in machine_options_list:
        all_machines.update(mo.keys())

    if not all_machines:
        all_machines = {0}

    machine_load = {m: 0.0 for m in all_machines}
    for i, mo in enumerate(machine_options_list):
        for m in mo:
            machine_load[m] += mo[m]

    # For each operation, find the best machine (least loaded + shortest duration)
    # and compute its estimated completion time
    op_completion_times = np.empty(n, dtype=np.float64)
    op_best_machines = [None] * n

    for i in range(n):
        mo = machine_options_list[i]
        best_m = None
        best_ct = float('inf')
        for m, dur in mo.items():
            ct = machine_load[m] + dur  # estimated completion if scheduled now
            if ct < best_ct:
                best_ct = ct
                best_m = m
        op_completion_times[i] = best_ct
        op_best_machines[i] = best_m

    # Now, for each candidate operation to schedule next, compute the minimum
    # slack across all OTHER ready operations after this one is removed.
    # Removing operation i reduces machine_load of its best machine by its duration.
    # This may increase slack for other operations on that machine.

    # Pre-compute: for each operation, what is its current estimated completion time?
    # After scheduling op i, the machine load changes, affecting other ops' completion times.

    # Simpler approach: score each operation by how much it improves the worst-case
    # completion time among all ready operations.

    # For each candidate i:
    #   - Remove i from consideration
    #   - Recompute completion times for remaining ops with updated machine loads
    #   - Find max completion time among remaining
    #   - The operation that minimizes this max completion time is best

    # But this is O(n^2 * machines) which might be slow. Let's optimize.

    # Alternative: Use a proxy score.
    # The operation that, when scheduled, most reduces the critical path should be chosen.
    # Critical path proxy: max over all operations of (machine_load[best_machine] + duration)

    # For each candidate i, estimate the reduction in critical path:
    #   If we schedule i on machine m_i, machine_load[m_i] decreases by dur_i.
    #   This affects all other operations that use m_i.
    #   The new critical path = max over j != i of (new_machine_load[best_m_j] + dur_j)

    # Let's compute this efficiently.

    # First, compute current critical path
    current_max_ct = float(np.max(op_completion_times))

    # For each candidate, estimate new critical path
    scores = np.empty(n, dtype=np.float64)

    for i in range(n):
        m_i = op_best_machines[i]
        dur_i = machine_options_list[i].get(m_i, 1.0)

        # New machine load for m_i
        new_load_m_i = machine_load[m_i] - dur_i

        # For each other operation j, compute new completion time
        max_ct = 0.0
        for j in range(n):
            if j == i:
                continue
            m_j = op_best_machines[j]
            dur_j = machine_options_list[j].get(m_j, 1.0)
            if m_j == m_i:
                new_ct = new_load_m_i + dur_j
            else:
                new_ct = op_completion_times[j]
            if new_ct > max_ct:
                max_ct = new_ct

        # Also consider the completion time of i itself (it's now being scheduled)
        # i's completion is at time + dur_i (assuming machine is free now)
        # But in our model, i's completion is dur_i (if machine is free at time 0)
        # Actually, the makespan contribution is max(max_ct, current_time + dur_i)
        # Since we're comparing relative scores, use max_ct as the proxy.

        # Lower max_ct is better
        scores[i] = -max_ct

    best_idx = int(np.argmax(scores))
    return best_idx
