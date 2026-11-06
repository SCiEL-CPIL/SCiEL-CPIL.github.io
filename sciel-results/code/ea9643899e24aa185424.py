"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global state to track machine loads dynamically across dispatch calls
# We assume the simulation runs sequentially within a single process for a given instance
_machine_loads = {}
_last_time = 0.0
_last_instance_id = None

def _update_machine_loads(ready_operations, time):
    """
    Heuristically update machine load estimates.
    Since we don't get explicit 'busy_until' from the interface, we approximate contention
    by tracking how often machines appear in the ready queue over time, 
    or simply by the sum of durations of ready ops assigned to them.
    
    A more robust approach for this interface:
    We estimate 'congestion' as the total remaining work that *could* go to a machine,
    or simply the count of ready ops preferring that machine.
    
    However, a better proxy for 'load' in a dispatch rule without future knowledge
    is the sum of processing times of all ready operations that have that machine as an option.
    This creates a dynamic penalty for machines with many pending jobs.
    """
    global _machine_loads, _last_time, _last_instance_id
    
    # Simple reset heuristic: if time jumps significantly or we detect a new instance, reset
    # For now, we compute a 'current contention score' for each machine based on the current ready set.
    # This is stateless per call but dynamic per state.
    
    machine_contention = {}
    for op in ready_operations:
        # We look at machine_options. Usually, a job has one preferred machine or a set.
        # The problem says machine_options {machine: duration}.
        # If a machine appears in multiple ready ops, it's contended.
        for m, dur in op["machine_options"].items():
            if m not in machine_contention:
                machine_contention[m] = 0.0
            # Accumulate the work that this machine might have to do soon
            machine_contention[m] += dur
            
    return machine_contention

def dispatch(ready_operations, time):
    """
    Dispatch rule: Dynamic Critical Ratio with Machine Contention Penalty.
    
    Score for each operation i:
    CR_i = remaining_work_i / max(time_horizon, epsilon)
    Contention_i = sum of durations of all ready ops that can use the machine i prefers (or all machines it can use)
    
    We want to pick the op with the highest 'urgency' but lowest 'contention cost'.
    Metric: (remaining_work / (time + 1)) / (1 + machine_contention_factor)
    """
    if not ready_operations:
        return 0

    # Calculate dynamic machine contention based on the current ready set
    # A machine with many pending operations is 'hot' and should be avoided if possible
    machine_load_estimate = {}
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            machine_load_estimate[m] = machine_load_estimate.get(m, 0.0) + dur
    
    # Normalize time to avoid division by zero or extreme values early on
    # We use a soft time horizon. As time increases, urgency (work/time) drops, 
    # but we also want to prioritize long jobs.
    # Actually, Critical Ratio is usually (Time Remaining - Work Remaining) / Work Remaining.
    # But we don't know Time Remaining (Makespan is unknown).
    # So we use a proxy: Urgency = Work Remaining / (Current Time + 1).
    # This favors jobs with lots of work early on (like MWR) but scales down as time passes.
    
    scores = []
    for i, op in enumerate(ready_operations):
        work = op["remaining_work"]
        
        # Determine the machine this operation is most likely to go to or the one with highest contention
        # If machine_options has multiple entries, the simulator might pick the first, or the best.
        # Assuming the dispatch index selects the op, and the machine is fixed by the op definition 
        # or the first key in machine_options. 
        # Let's assume the machine is the one with the highest duration or the first one.
        # To be safe, we penalize based on the *maximum* contention among its available machines,
        # or the contention of its primary machine.
        
        # Let's assume the machine is the first key in machine_options for simplicity of contention mapping
        # Or better, take the average contention of its options.
        if op["machine_options"]:
            # Use the machine with the longest duration as the 'primary' bottleneck candidate
            primary_machine = max(op["machine_options"], key=op["machine_options"].get)
            contention = machine_load_estimate.get(primary_machine, 0.0)
        else:
            contention = 0.0
        
        # Urgency: Higher is better (more work to do)
        # Time factor: We want to avoid waiting. 
        # Let's use a combined score:
        # Score = Work * (1 / (1 + Contention))
        # This prioritizes high work, but heavily discounts if the machine is loaded with other pending work.
        
        # To make it more dynamic with time, we can add a small time-decay to the work preference
        # so that as time goes on, we don't just stick to the biggest job if it's blocked.
        # But without explicit blocking info, contention is our proxy.
        
        score = work / (1.0 + contention * 0.5)
        scores.append(score)
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
