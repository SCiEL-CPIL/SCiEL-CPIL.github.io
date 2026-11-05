"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import defaultdict

def _simulate(ready_ops, current_time, max_depth=3, beam_width=5):
    """
    Beam search over the next `max_depth` dispatch decisions.
    State: (time, list of remaining ready ops, list of completed ops with their end times)
    We track per-job remaining ops count and per-machine busy-until time.
    
    Since we don't have full machine state, we approximate:
    - Each machine has a "busy_until" time (initially current_time for all, since we only know ready ops)
    - When an op is assigned to machine m at time t, it finishes at t + duration
    - Other ready ops waiting for that machine must wait until max(current_time, busy_until[m])
    
    Actually, in a proper simulation, we need to track:
    - For each job: which ops are done, which are ready
    - For each machine: when it becomes free
    
    But we only have ready_operations at the top level. We don't know non-ready ops.
    So we can only simulate the ready ops and assume no other ops exist or are already scheduled.
    
    This is a limitation. Let's assume the ready_operations list contains ALL ops that will ever be ready,
    and we're simulating their completion.
    """
    # Initialize state
    # We need to track:
    # - machine_free: dict machine -> time when machine is free
    # - For each op in ready_ops: we need to know its job, op index, machine_options
    # - We'll simulate by picking ops one at a time
    
    # For beam search, each beam element is:
    # (makespan_estimate, state)
    # state = (time, machine_free dict, list of remaining ops)
    
    # Initial state
    initial_machine_free = {}
    # Assume all machines are free at current_time initially (conservative)
    # Actually, we don't know the true machine free times. Let's assume they're free.
    
    # To make this more realistic, we could track that some machines might be busy,
    # but we have no info. Let's just use current_time as the start for all.
    
    initial_state = (current_time, {}, list(ready_ops))
    
    # Beam: list of (predicted_makespan, state)
    # We'll expand each beam element by trying all possible next ops
    # and selecting the best `beam_width` combinations
    
    def expand(state, depth_remaining):
        """
        Expand a state by trying all possible next operations.
        Returns list of (makespan_estimate, new_state) for each choice.
        """
        time_val, machine_free, remaining_ops = state
        if not remaining_ops:
            return [(time_val, state)]  # All done, makespan is current time
        
        results = []
        for i, op in enumerate(remaining_ops):
            options = op["machine_options"]
            if not options:
                continue
            
            # For each machine option, calculate when this op can start and finish
            for m, dur in options.items():
                # When can this op start?
                # It can start at max(current_time, machine_free.get(m, current_time))
                # But in a discrete event sim, we pick ONE op to start now.
                # The op starts at current_time if the machine is free, else waits.
                # However, in our dispatch interface, we pick one op to start NOW.
                # So the op starts at current_time, but if the machine is busy, it waits.
                
                # Actually, the dispatch function picks which op to start.
                # The simulation should model: given we pick op i on machine m,
                # when does it finish? And what happens to other ops?
                
                # Simplified model:
                # - The chosen op starts at max(current_time, machine_free.get(m, current_time))
                # - It finishes at start_time + dur
                # - Other ops remain ready, but their earliest start time is updated
                #   based on the new machine_free times
                
                start_time = max(time_val, machine_free.get(m, time_val))
                finish_time = start_time + dur
                
                # New machine_free: m is now busy until finish_time
                new_machine_free = dict(machine_free)
                new_machine_free[m] = finish_time
                
                # Remaining ops: remove op i
                new_remaining = [op for j, op in enumerate(remaining_ops) if j != i]
                
                # For the remaining ops, their earliest start time is now
                # max(current_time, min over their machine options of machine_free[m])
                # But we don't need to update their state explicitly; we just track machine_free.
                
                # If this is the last op, makespan is finish_time
                if not new_remaining:
                    results.append((finish_time, (finish_time, new_machine_free, new_remaining)))
                else:
                    # We need to estimate the makespan from here.
                    # For beam search, we can recursively expand or use a heuristic.
                    # Let's use a simple greedy completion estimate for the remaining ops.
                    est = _greedy_estimate(new_remaining, finish_time, new_machine_free)
                    results.append((est, (finish_time, new_machine_free, new_remaining)))
        
        return results
    
    def _greedy_estimate(remaining_ops, current_time, machine_free):
        """
        Simple greedy estimate: always pick the op with the shortest duration on its best machine.
        """
        time_val = current_time
        mf = dict(machine_free)
        ops = list(remaining_ops)
        
        while ops:
            # Find the op with the minimum finish time
            best_op_idx = -1
            best_finish = float('inf')
            best_machine = None
            best_dur = float('inf')
            
            for i, op in enumerate(ops):
                for m, dur in op["machine_options"].items():
                    start = max(time_val, mf.get(m, time_val))
                    finish = start + dur
                    if finish < best_finish:
                        best_finish = finish
                        best_op_idx = i
                        best_machine = m
                        best_dur = dur
            
            if best_op_idx == -1:
                break
            
            # Remove the chosen op
            ops.pop(best_op_idx)
            mf[best_machine] = best_finish
            time_val = best_finish  # Actually, time doesn't advance for other ops; they can start in parallel
            # Wait, in job-shop, multiple machines work in parallel. So time_val should stay the same?
            # No, the next op to start can start at the current simulation time, which is the time
            # when we're making the decision. But in a greedy sequential estimate, we're just
            # estimating the makespan, so we should track the maximum finish time.
            
            # Actually, for a proper makespan estimate, we need to simulate all ops.
            # Let's track the makespan as the max finish time of all ops.
        
        # This greedy approach is flawed because it doesn't account for parallelism properly.
        # Let's use a different estimate: sum of durations / number of machines, or just the max duration.
        # For simplicity, let's estimate makespan as current_time + sum of remaining durations / num_machines
        # This is a lower bound.
        
        total_remaining = sum(
            min(op["machine_options"].values()) for op in remaining_ops
        )
        num_machines = len(set(m for op in remaining_ops for m in op["machine_options"]))
        if num_machines == 0:
            num_machines = 1
        est = current_time + total_remaining / num_machines
        return est
    
    # Beam search
    beam = [(float('inf'), initial_state)]
    
    for depth in range(max_depth):
        new_beam = []
        for _, state in beam:
            expanded = expand(state, max_depth - depth - 1)
            new_beam.extend(expanded)
        
        # Sort by predicted makespan and keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    # The best estimate is the first element in the beam
    if beam:
        return beam[0][0]
    return current_time

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    start_time = _time.time()
    deadline = start_time + 10  # 10 seconds for safety
    
    # Use beam search to evaluate each candidate operation
    best_index = 0
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        if _time.time() > deadline:
            break
        
        # Simulate the schedule if we pick this operation now
        # We need to modify the ready_operations list to remove this op and add its effect
        # But we don't want to mutate the original list.
        
        # For simplicity, let's just estimate the makespan based on this choice
        # without full simulation, using a heuristic:
        # score = duration of this op + estimated remaining makespan
        
        options = op["machine_options"]
        if not options:
            continue
        
        min_dur = min(options.values())
        # Estimate: this op takes min_dur, and the rest takes... 
        # Let's use the beam search for a proper estimate, but it's expensive.
        # For speed, let's use a simpler heuristic:
        # score = min_dur + sum of min durations of other ready ops / num_machines
        
        other_ops = [op for j, op in enumerate(ready_operations) if j != i]
        total_other = sum(min(op["machine_options"].values()) for op in other_ops)
        num_machines = len(set(m for op in other_ops for m in op["machine_options"]))
        if num_machines == 0:
            num_machines = 1
        est = min_dur + total_other / num_machines
        
        if est < best_score:
            best_score = est
            best_index = i
    
    return int(best_index)

# EVOLVE-BLOCK-END
