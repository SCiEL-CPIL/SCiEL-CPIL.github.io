"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE = 110.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def _softmin(values, temperature=1.0):
    """Soft-min using log-sum-exp trick for numerical stability."""
    if not values:
        return float('inf')
    v = list(values)
    m = min(v)
    if m <= 0:
        m = 1e-9
    # Soft-min: -T * log(sum(exp(-x/T)))
    exps = [math.exp(-(x - m) / temperature) for x in v]
    s = sum(exps)
    if s <= 0:
        return m
    return m - temperature * math.log(s)


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.

    Strategy:
    1. Urgency: combine remaining work with a critical-path proxy (remaining ops)
       to prioritize jobs that are both long and deep in their sequence.
    2. Machine selection: use a smooth soft-min surrogate over candidate machines,
       where each machine's cost is duration + W * other_load. This avoids the
       nonsmoothness of hard argmin and provides a better gradient signal.
    3. Score = urgency / soft_min_cost, selecting the pair with highest score.

    The smooth surrogate helps by not being overly sensitive to small differences
    between near-tied machines, leading to more balanced load distribution.
    """
    if _elapsed() > _DEADLINE:
        # Hard safety fallback: Shortest Processing Time (SPT)
        if not ready_operations:
            return 0
        best_i = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
        return best_i

    if not ready_operations:
        return 0
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    n = len(ready_operations)

    # Precompute remaining work, remaining ops, and machine options for all ready ops
    op_data = []
    for op in ready_operations:
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1.0))
        opts = op.get("machine_options", {})
        op_data.append((rw, ro, opts))

    # Precompute machine load: total committed duration of all ready ops per machine.
    machine_load = {}
    for rw, ro, opts in op_data:
        for m, d in opts.items():
            machine_load[m] = machine_load.get(m, 0.0) + float(d)

    # Congestion weight: how strongly we penalize machines that other ready ops
    # also need. Higher => avoid bottlenecks more aggressively.
    W = 0.5

    # Soft-min temperature: lower = closer to hard min, higher = smoother
    T = 2.0

    # Weight for critical-path (remaining ops) in urgency
    ALPHA = 0.4

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    for i in range(n):
        rw, ro, opts = op_data[i]
        if rw <= 0 and ro <= 0:
            continue

        # Compute urgency: blend of remaining work and a critical-path proxy
        # Use max(rw, ro * avg_duration) to capture both total work and depth
        avg_d = rw / max(ro, 1.0)
        cp_proxy = ro * avg_d  # this equals rw, so use a different formulation
        # Better: urgency = rw + ALPHA * ro * min_duration_proxy
        # where min_duration_proxy approximates the shortest possible completion
        min_dur = min(opts.values()) if opts else 1.0
        urgency = rw + ALPHA * ro * min_dur

        if urgency <= 0:
            continue

        # Collect costs for all eligible machines
        machine_costs = []
        for m, d in opts.items():
            d = float(d)
            if d == 0:
                continue
            load = machine_load.get(m, d)
            other_load = load - d  # committed load of *other* ready ops on m
            # Use a convex combination: emphasize duration, penalize congestion
            cost = d * (1.0 + W * other_load / max(d, 1e-9))
            if cost <= 0:
                cost = 1e-9
            machine_costs.append((m, cost))

        if not machine_costs:
            continue

        # Use soft-min for smooth machine selection cost
        costs_only = [c for _, c in machine_costs]
        soft_cost = _softmin(costs_only, temperature=T)
        if soft_cost <= 0:
            soft_cost = 1e-9

        # For the actual machine choice, pick the argmin (hard selection)
        # but use soft cost in the score for smoothness
        best_m_for_op = min(machine_costs, key=lambda x: x[1])[0]

        # Score: urgency / soft effective cost. Higher is better.
        # Add a small time-awareness: prefer operations that can start sooner
        # relative to the current simulation time to reduce idle gaps
        time_factor = 1.0 + 0.1 * (time / max(rw, 1.0))
        score = (urgency / soft_cost) * time_factor

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Fallback if no valid score was found
    if best_machine is None:
        best_index = 0
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: opts[k])
        else:
            best_machine = None

    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
