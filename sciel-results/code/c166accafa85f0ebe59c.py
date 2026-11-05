"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE = 110.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.

    Strategy:
    1. Urgency: prioritize jobs with the most remaining work (avoids starving
       long jobs and keeps the schedule progressing).
    2. Machine selection: for the chosen operation, pick the eligible machine
       that minimizes  duration + W * backlog,  where backlog is the committed
       load of the *other* ready operations that can also use that machine.
       This balances short processing time against avoiding machine bottlenecks.
    3. Tie-breaking: prefer the machine with the least total ready-op
       contention (spreads load across machines).

    The combination favors finishing urgent work on the least-loaded machine,
    which empirically reduces makespan in flexible job-shop settings.
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

    # Precompute remaining work and machine options for all ready ops
    op_data = []
    for op in ready_operations:
        rw = float(op.get("remaining_work", 0.0))
        opts = op.get("machine_options", {})
        op_data.append((rw, opts))

    # Precompute machine load: total committed duration of all ready ops per machine.
    machine_load = {}
    for rw, opts in op_data:
        for m, d in opts.items():
            machine_load[m] = machine_load.get(m, 0.0) + float(d)

    # Congestion weight: how strongly we penalize machines that other ready ops
    # also need. Higher => avoid bottlenecks more aggressively.
    W = 0.5

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    for i in range(n):
        rw, opts = op_data[i]
        if rw <= 0:
            continue

        # Find the best machine for this operation
        best_m_for_op = None
        best_pair_cost = float('inf')
        best_pair_tie = float('inf')

        for m, d in opts.items():
            d = float(d)
            if d == 0:
                continue

            load = machine_load.get(m, d)
            other_load = load - d  # committed load of *other* ready ops on m

            cost = d + W * other_load
            if cost <= 0:
                cost = 1e-9

            # Tie-break: prefer machine with least total ready-op contention
            tie = load

            if cost < best_pair_cost or (cost == best_pair_cost and tie < best_pair_tie):
                best_pair_cost = cost
                best_pair_tie = tie
                best_m_for_op = m

        if best_m_for_op is None:
            continue

        # Score: urgency / effective cost. Higher is better.
        score = rw / best_pair_cost

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
