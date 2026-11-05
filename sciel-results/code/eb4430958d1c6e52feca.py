"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE_MARGIN = 5  # seconds before the 150s kill; be conservative


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Strategy:
    1. Primary: Machine-load balancing. Prefer operations whose best machine
       is the least constrained (fewest ready ops can use it) and has the
       lowest aggregate load.
    2. Secondary: Remaining work (LRPT) to prioritize critical path jobs.
    3. Tertiary: Shortest minimum duration (SPT) to release resources quickly.
    4. Machine: Machine with shortest duration for the chosen operation.

    Return: (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op)
        return (0, best_m)

    # Deadline guard: if we are too close to the limit, return a quick heuristic
    if _elapsed() > 145:
        op = ready_operations[0]
        best_m, _ = _best_machine_for_op(op)
        return (0, best_m)

    # Aggregate machine statistics across ready ops
    machine_load = {}       # total duration if all ready ops used this machine
    machine_count = {}      # number of ready ops that can use this machine
    machine_min_dur = {}    # min duration per machine across ready ops

    for op in ready_operations:
        durations = op["machine_options"]
        if not durations:
            continue
        for m, d in durations.items():
            machine_load[m] = machine_load.get(m, 0.0) + d
            machine_count[m] = machine_count.get(m, 0) + 1
            if m not in machine_min_dur or d < machine_min_dur[m]:
                machine_min_dur[m] = d

    scores = []
    best_machines = []

    for op in ready_operations:
        rw = op["remaining_work"]
        durations = op["machine_options"]
        if not durations:
            scores.append(-1)
            best_machines.append(None)
            continue

        # For this operation, find the best machine (shortest duration)
        best_m, best_d = _best_machine_for_op(op)

        # Machine constraint score: fewer ops can use this machine => more constrained
        # We want to prioritize ops on constrained machines to prevent bottlenecks
        m_count = machine_count.get(best_m, 1)
        m_load = machine_load.get(best_m, 0.0)
        m_min = machine_min_dur.get(best_m, best_d)

        # Normalize machine load by number of ops that can use it
        # Lower normalized load => less loaded machine => better to assign here
        normalized_load = m_load / max(m_count, 1)

        # Machine constraint penalty: higher count means less constrained
        # We penalize ops whose best machine is already heavily loaded or has many options
        machine_constraint_score = m_count * 1000.0 + normalized_load * 10.0

        # Soft-min duration for this op (smooth surrogate)
        vals = np.array(list(durations.values()), dtype=float)
        T = 1.0
        neg = -vals / T
        neg_shift = neg - np.max(neg)
        logsumexp_neg = np.max(neg) + np.log(np.sum(np.exp(neg_shift)))
        soft_min_dur = -T * logsumexp_neg

        # Combined score:
        # - Higher remaining work is good (LRPT)
        # - Lower machine constraint score is good (balance load)
        # - Lower soft-min duration is good (SPT)
        # We want to maximize the score.
        # Machine constraint is a penalty, so subtract it.
        score = rw * 1e6 - machine_constraint_score - soft_min_dur

        scores.append(score)
        best_machines.append(best_m)

    # Find index with max score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]

    if best_machine is None:
        return best_idx
    return (best_idx, best_machine)


def _best_machine_for_op(op):
    """
    For a given operation, select the machine that minimizes its duration.
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    best_machine = min(durations, key=lambda m: durations[m])
    return best_machine, durations[best_machine]
# EVOLVE-BLOCK-END
