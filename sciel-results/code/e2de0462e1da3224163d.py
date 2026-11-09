import numpy as np
import time as _time

_START = _time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

def _projected_makespan(ready_ops, current_time, machine_loads, job_remaining_work, job_remaining_ops):
    """
    Estimate projected makespan given current state.
    machine_loads: dict machine -> total work assigned so far (including current_time offset)
    job_remaining_work: dict job -> remaining work
    job_remaining_ops: dict job -> remaining ops count
    """
    # Lower bound: max machine load (when each machine finishes its queued work)
    if machine_loads:
        max_machine_load = max(machine_loads.values())
    else:
        max_machine_load = current_time

    # Lower bound: for each job, the earliest it can finish is current_time + remaining_work
    # but also constrained by number of ops (sequential)
    max_job_finish = current_time
    for job, rw in job_remaining_work.items():
        if rw > 0:
            # Job needs rw more work, done sequentially, so at least rw time
            finish = current_time + rw
            if finish > max_job_finish:
                max_job_finish = finish

    # A simple makespan estimate: max of machine load and job finish times
    # But machine loads already account for time, so combine
    # Actually machine_loads should be absolute completion times
    return max(max_machine_load, max_job_finish)

def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using greedy lookahead simulation.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _time.time() > _DEADLINE - 5.0:
        # Fallback: pick highest remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Build current state estimates
    # We don't have direct access to machine loads from the simulator,
    # so we estimate from the ready operations and their machine options.
    
    # For each candidate, simulate its completion and estimate resulting makespan
    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        # Pick the machine with minimum duration for this op (best case)
        machine_options = op["machine_options"]
        if not machine_options:
            # No machine options? Skip or assign worst
            continue
        best_machine = min(machine_options, key=lambda m: machine_options[m])
        duration = machine_options[best_machine]
        completion_time = time + duration

        # After this op completes:
        # - This job's remaining work decreases by duration
        # - This job's remaining ops decreases by 1
        # - The machine gets this duration of load
        
        # Estimate projected makespan
        # Machine loads: assume current time is baseline, this machine gets duration
        # Other machines: we don't know their loads, estimate from ready ops
        
        # Simple estimate: 
        # 1. The machine that runs this op will be busy until completion_time
        # 2. Other ready ops will need to run after some time
        
        # Projected makespan lower bound:
        # - For the job of this op: completion_time + (remaining_work - duration)
        # - For other jobs: time + their remaining_work (they can't start until some time)
        # - Machine constraint: completion_time (this machine done)
        
        job = op["job"]
        rw = op["remaining_work"]
        new_rw = max(0, rw - duration)
        
        # This job's projected finish
        this_job_finish = completion_time + new_rw
        
        # Other jobs' projected finish (they start at earliest 'time', but machines may be busy)
        # Conservative: they finish at time + their remaining_work
        max_other_finish = time
        for j, other_op in enumerate(ready_operations):
            if j == i:
                continue
            other_rw = other_op["remaining_work"]
            # This other job's finish estimate
            other_finish = time + other_rw
            if other_finish > max_other_finish:
                max_other_finish = other_finish
        
        # Also consider machine load: the machine running this op is busy until completion_time
        # Other ops that want this machine must wait
        # For simplicity, use max of this_job_finish, max_other_finish, completion_time
        projected = max(this_job_finish, max_other_finish, completion_time)
        
        # Add a small penalty for machine inflexibility (fewer options = more blocking risk)
        num_options = len(machine_options)
        flexibility_penalty = 1.0 / max(1, num_options)  # lower flexibility = higher penalty
        
        # Combine: minimize projected makespan, with flexibility as tiebreaker
        # Use multiplicative interaction: urgency (remaining work) * constraint (1/flexibility)
        urgency = op["remaining_work"]
        score = projected + 0.1 * urgency * flexibility_penalty
        
        if score < best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
