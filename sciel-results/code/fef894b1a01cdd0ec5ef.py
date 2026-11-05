"""Job-shop dispatch rule: Ratio-based critical path with soft-min machine assignment."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def _critical_path_estimate(ready_operations, op_index):
    """
    Estimate the critical path length for the job containing ready_operations[op_index].
    Uses sum of minimum durations of remaining operations as a proxy for longest path.
    """
    op = ready_operations[op_index]
    job = op.get("job")
    
    # Sum minimum possible durations for all remaining ops in this job
    min_duration_sum = 0.0
    for i, other_op in enumerate(ready_operations):
        if other_op.get("job") == job:
            mo = other_op.get("machine_options", {})
            if mo:
                min_d = min(mo.values())
                min_duration_sum += min_d
    
    # Also factor in remaining_work as a secondary signal
    rw = float(op.get("remaining_work", 0) or 0)
    
    # Combine: primarily use critical path estimate, with work volume as tiebreaker
    return min_duration_sum + 0.1 * rw


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Strategy:
    1. Primary signal: Ratio of remaining work to estimated completion cost
       (RW / softmin_cost). This prioritizes jobs that are both deep AND
       have feasible machine options, balancing depth against feasibility.
    2. Soft-min over machine options with contention-aware cost gives a smooth
       surrogate for the nonsmooth min-duration selection.
    3. Small urgency bonus for jobs with fewer remaining operations to break
       ties and encourage parallelism.
    4. Returns (index, chosen_machine) for the selected operation.
    """
    if not ready_operations:
        return 0
    if _deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Build machine contention counts: how many ready ops can use each machine
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    # Pre-compute critical path estimates for each job
    job_cp_estimates = {}
    for i, op in enumerate(ready_operations):
        job = op.get("job")
        if job not in job_cp_estimates:
            job_cp_estimates[job] = _critical_path_estimate(ready_operations, i)

    best_idx = 0
    best_machine = None
    best_score = float("-inf")

    for i in range(n):
        op = ready_operations[i]
        job = op.get("job")
        rw = float(op.get("remaining_work", 0) or 0)
        rem_ops = op.get("remaining_ops", 0) or 0

        mo = op.get("machine_options", {})
        if not mo:
            # Fallback: use critical path estimate
            cp_est = job_cp_estimates.get(job, rw)
            score = cp_est
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = None
            continue

        # Build contention-adjusted costs for each machine option
        costs = []
        for m, d in mo.items():
            c = machine_counts.get(m, 1)
            # Cost model: duration scaled by contention factor
            cost = d * (1.0 + 0.4 * (c - 1))
            costs.append(cost)

        # Find the minimum cost machine
        min_cost_val = float("inf")
        chosen_m = None
        for m, d in mo.items():
            c = machine_counts.get(m, 1)
            cost = d * (1.0 + 0.4 * (c - 1))
            if cost < min_cost_val:
                min_cost_val = cost
                chosen_m = m

        # Primary signal: critical path estimate (higher = more urgent)
        cp_est = job_cp_estimates.get(job, rw)
        
        # Secondary: efficiency ratio (remaining work / best machine cost)
        if min_cost_val > 0:
            efficiency = rw / min_cost_val
        else:
            efficiency = rw

        # Combine: 70% critical path urgency, 30% efficiency
        score = 0.7 * cp_est + 0.3 * efficiency

        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = chosen_m

    if best_machine is not None:
        return (int(best_idx), best_machine)
    else:
        return int(best_idx)

# EVOLVE-BLOCK-END
