"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard: 150s wall-clock limit, keep safety margin
_DEADLINE_START = None
_DEADLINE_LIMIT = 145.0  # seconds, safety margin from 150s

def _check_deadline():
    """Return True if we should stop searching and return best-so-far."""
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = _time.monotonic()
    return (_time.monotonic() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and optionally which machine.
    
    Uses a regret-based look-ahead heuristic:
    - For each ready operation, compute the "regret" = difference between
      the best and second-best machine assignment costs (duration + estimated
      future machine load via remaining_work).
    - Prioritize operations with high regret (machine choice matters most)
      and high urgency (shortest best-machine duration).
    - Return (index, machine) to also select the best machine.
    """
    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    if n_ops == 1:
        op = ready_operations[0]
        best_mach = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_mach)

    # Collect data for each ready operation
    best_costs = []
    second_costs = []
    best_machines = []
    remaining_works = []
    remaining_ops = []
    job_ids = []

    for op in ready_operations:
        mo = op["machine_options"]
        # Sort machines by duration
        sorted_machines = sorted(mo.keys(), key=lambda m: mo[m])
        best_m = sorted_machines[0]
        best_c = mo[best_m]
        second_c = mo[sorted_machines[1]] if len(sorted_machines) > 1 else best_c + 1.0

        best_costs.append(best_c)
        second_costs.append(second_c)
        best_machines.append(best_m)
        remaining_works.append(op.get("remaining_work", 0.0))
        remaining_ops.append(op.get("remaining_ops", 0))
        job_ids.append(op.get("job", 0))

    best_costs = np.array(best_costs, dtype=np.float64)
    second_costs = np.array(second_costs, dtype=np.float64)
    remaining_works = np.array(remaining_works, dtype=np.float64)
    remaining_ops = np.array(remaining_ops, dtype=np.float64)

    # Regret: how much worse the second-best choice is compared to best
    regret = second_costs - best_costs

    # Congestion estimate: use remaining_work as proxy for machine load.
    # Operations with more remaining work on their best machine are more urgent.
    # We estimate future load on the best machine by remaining_work (total work left).
    # A simple congestion penalty: if many ops share similar best machines,
    # we penalize to spread load. But we don't have machine-level queue info,
    # so we use a heuristic: shorter best_cost ops are less congesting.

    # Urgency score: shorter best duration is better (less blocking)
    # Combine: high regret + short best duration + high remaining work
    # Normalize components
    if n_ops > 1:
        regret_max = np.max(regret)
        regret_min = np.min(regret)
        regret_range = regret_max - regret_min
        if regret_range < 1e-9:
            regret_range = 1.0
        regret_norm = (regret - regret_min) / regret_range

        best_cost_max = np.max(best_costs)
        best_cost_min = np.min(best_costs)
        best_cost_range = best_cost_max - best_cost_min
        if best_cost_range < 1e-9:
            best_cost_range = 1.0
        # Invert: shorter is better
        urgency_norm = 1.0 - (best_costs - best_cost_min) / best_cost_range

        rw_max = np.max(remaining_works)
        rw_min = np.min(remaining_works)
        rw_range = rw_max - rw_min
        if rw_range < 1e-9:
            rw_range = 1.0
        rw_norm = (remaining_works - rw_min) / rw_range

        # Composite score: weight regret heavily, then urgency, then remaining work
        # Regret tells us "this decision is critical" -> schedule it now
        # Urgency tells us "this op finishes fast" -> good to start early
        # Remaining work tells us "this job has a lot left" -> prioritize long jobs
        score = 0.45 * regret_norm + 0.35 * urgency_norm + 0.20 * rw_norm
    else:
        score = np.array([0.0])

    # Deadline guard: if we're close to the limit, just pick the top score
    if _check_deadline():
        idx = int(np.argmax(score))
        return (idx, best_machines[idx])

    idx = int(np.argmax(score))
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
