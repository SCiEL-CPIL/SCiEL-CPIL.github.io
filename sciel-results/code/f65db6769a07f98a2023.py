"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Phase-adaptive dispatch with machine-contention-aware routing.
    
    Structural idea: The schedule has two regimes. Early on, many jobs have
    large remaining work — prioritize the job with the most remaining work
    (LRPT / critical-path urgency) so the longest chain is never starved.
    Late on, remaining work is small and the queue is short — prioritize
    the operation that finishes fastest (SRPT / throughput) to clear the
    tail and reduce makespan.
    
    Machine selection: instead of always picking the shortest-duration
    machine, apply a soft congestion penalty based on how many other
    ready operations are also eligible for that machine. This avoids
    piling all operations onto one "fast" machine and balances load.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Pre-compute per-operation features
    rws = []
    min_durs = []
    machine_options_list = []
    remaining_ops_counts = []

    # Machine contention: count how many ready ops are eligible for each machine
    machine_usage_counts = {}

    for op in ready_operations:
        mw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})

        rws.append(mw)
        remaining_ops_counts.append(ro)
        machine_options_list.append(mo)

        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            min_durs.append(mo[best_m])
            for m in mo.keys():
                machine_usage_counts[m] = machine_usage_counts.get(m, 0) + 1
        else:
            min_durs.append(0)

    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Phase indicator: how much work is left in the system relative to the max.
    # Early: avg_rw / max_rw is high (jobs are similar, lots of work remaining)
    # Late:  avg_rw / max_rw is low  (jobs have diverged, little work remaining)
    avg_rw = sum(rws) / n
    max_rw = max(rws) if rws else 1.0
    min_rw = min(rws) if rws else 0.0

    if max_rw > 0:
        phase_ratio = avg_rw / max_rw
    else:
        phase_ratio = 1.0

    # phase_ratio near 1.0  -> early phase (use LRPT: weight remaining_work heavily)
    # phase_ratio near 0.0  -> late phase  (use SRPT: weight 1/duration heavily)
    # Blend: interpolate between the two regimes
    # alpha = 1.0 means "urgency" (LRPT), alpha = 0.0 means "throughput" (SRPT)
    alpha = phase_ratio  # 1.0 early, 0.0 late

    best_score = -1.0
    best_index = 0
    best_machine = None

    for i in range(n):
        rw = rws[i]
        ro = remaining_ops_counts[i]
        mo = machine_options_list[i]

        if not mo:
            continue

        # --- Machine selection with congestion penalty ---
        best_m_for_op = None
        best_machine_cost = float('inf')

        for m, d in mo.items():
            cnt = machine_usage_counts.get(m, 0)
            # Soft penalty: for cnt=1, penalty~0; for cnt=2, ~0.33; saturates toward 0.5
            # This gently discourages choosing a machine that many other ops also want
            penalty = 0.3 * cnt / (1.0 + 0.5 * cnt)
            cost = d + penalty
            if cost < best_machine_cost:
                best_machine_cost = cost
                best_m_for_op = m

        if best_m_for_op is None:
            continue

        min_dur = mo[best_m_for_op]
        if min_dur <= 0:
            min_dur = 1e-9

        # --- Scoring: blend LRPT and SRPT ---
        # LRPT component: remaining work (urgency, critical path)
        lrpt_score = rw

        # SRPT component: inverse of duration (throughput, clear the queue)
        # Normalize by adding 1 to avoid division by zero and keep scale reasonable
        srpt_score = 1.0 / (min_dur + 1.0)

        # Scale srpt_score to be comparable to lrpt_score
        # Use max_rw as a reference scale so srpt_score is in similar magnitude
        srpt_scaled = srpt_score * (max_rw + 1.0)

        # Blend
        score = alpha * lrpt_score + (1.0 - alpha) * srpt_scaled

        # Small tie-breaker: jobs with more remaining ops have longer chains
        score += 0.05 * ro

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    return (best_index, best_machine)
# EVOLVE-BLOCK-END
