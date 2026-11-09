"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Select the operation that, when started on its best machine,
    minimizes the resulting makespan estimate. We use a combination of:
    1. Critical Path Method (CPM)-like estimation of remaining completion times
    2. Machine load balancing to avoid bottlenecks
    
    For each ready operation, we estimate what the makespan would be if we
    started it now on its best available machine, and pick the one that
    leads to the smallest overall makespan estimate.
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Quick return for single operation
    if n_ops == 1:
        return 0
    
    # Build per-job remaining work and per-machine current load estimates
    # We need to track:
    # - For each job: total remaining work (including current op's duration)
    # - For each machine: total remaining work assigned to it
    
    # First pass: collect job info and machine loads
    job_remaining = {}  # job_id -> total remaining work
    machine_load = {}   # machine_id -> total remaining work on that machine
    
    # We need to figure out which machine each operation will be assigned to.
    # For estimation, assume each operation goes to the machine that minimizes
    # its duration (best machine for that op).
    
    op_best_machine = {}  # (job, op) -> (machine, duration)
    
    for i, op in enumerate(ready_operations):
        job = op["job"]
        mop = op["op"]
        machine_options = op["machine_options"]
        
        # Find the machine with minimum duration for this operation
        best_mach = None
        best_dur = float('inf')
        for m, d in machine_options.items():
            if d < best_dur:
                best_dur = d
                best_mach = m
        
        op_best_machine[i] = (best_mach, best_dur)
        
        # Track job remaining work
        if job not in job_remaining:
            job_remaining[job] = 0
        job_remaining[job] += op.get("remaining_work", 0)
        
        # Track machine load
        if best_mach not in machine_load:
            machine_load[best_mach] = 0
        machine_load[best_mach] += best_dur
    
    # Estimate makespan if we start each operation now
    # Makespan estimate = max over jobs of (job's remaining completion time)
    # and max over machines of (machine's remaining completion time)
    
    best_score = float('inf')
    best_idx = 0
    
    for i in range(n_ops):
        op = ready_operations[i]
        job = op["job"]
        mop = op["op"]
        machine_options = op["machine_options"]
        
        # Try each possible machine assignment for this operation
        # and compute the resulting makespan estimate
        best_for_this_op = float('inf')
        
        for m, d in machine_options.items():
            # If we assign this op to machine m with duration d:
            # Job completion time estimate: current time + d + (other remaining work for this job)
            # Machine completion time estimate: current time + d + (other remaining work on this machine)
            
            # Remaining work for this job excluding this operation
            job_other_work = job_remaining.get(job, 0) - op.get("remaining_work", 0)
            job_completion = time + d + job_other_work
            
            # Remaining work on this machine excluding this operation
            # (we need to subtract what we already counted for this op if it was assigned here)
            mach_other_work = machine_load.get(m, 0)
            # If this op's best machine was m, we already added best_dur to machine_load[m]
            # We need to be careful: machine_load was built assuming best machine
            # For estimation, let's just use machine_load as-is and add d
            # Actually, let's recompute more carefully
            
            # Machine completion: time + d + (sum of durations of other ops on this machine)
            # We don't have exact info about other ops' machine assignments, so use estimate
            mach_completion = time + d + mach_other_work
            
            # The makespan is at least max(job_completion, mach_completion)
            # But also need to consider other jobs and machines
            # For a quick estimate, use max of these two
            
            makespan_est = max(job_completion, mach_completion)
            
            if makespan_est < best_for_this_op:
                best_for_this_op = makespan_est
        
        # Additional heuristic: prefer operations that unblock more work
        # or that have high remaining work (to avoid long tail)
        # Combine: minimize makespan estimate, break ties by remaining work
        
        if best_for_this_op < best_score:
            best_score = best_for_this_op
            best_idx = i
        elif best_for_this_op == best_score:
            # Tie-break: prefer operation with more remaining work (longest processing first)
            if ready_operations[i].get("remaining_work", 0) > ready_operations[best_idx].get("remaining_work", 0):
                best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
