"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


# Global state to cache parameters and track time
_start_time = _time.time()
_DEADLINE = 150.0  # Safety margin before 150s kill

# Track machine loads to estimate contention
_machine_loads = {}  # machine_id -> estimated remaining load
_last_dispatch_time = 0.0  # Track when last dispatch happened for load decay

def dispatch(ready_operations, time):
    global _last_dispatch_time
    
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to the limit, use a very fast fallback
    elapsed = _time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback: pick the operation with minimum duration
        best_index = 0
        min_dur = float('inf')
        for i, op in enumerate(ready_operations):
            machine_options = op.get("machine_options", {})
            if not machine_options:
                continue
            md = min(machine_options.values())
            if md < min_dur:
                min_dur = md
                best_index = i
        return int(best_index)

    # Decay machine loads based on time elapsed since last dispatch
    dt = max(time - _last_dispatch_time, 0.0)
    if dt > 0:
        for m in list(_machine_loads.keys()):
            _machine_loads[m] = max(_machine_loads[m] - dt, 0.0)
            if _machine_loads[m] <= 1e-6:
                del _machine_loads[m]
    _last_dispatch_time = time

    # Dynamic resource contention model with soft-min surrogate:
    # For each ready operation, compute a smooth estimate of the best machine
    # using a soft-min (log-sum-exp) over effective processing times.
    # This provides a differentiable surrogate for the nonsmooth min operation.
    
    best_score = -float('inf')
    best_index = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        num_options = len(machine_options)
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        
        # Compute effective processing times for each machine
        # effective_time = duration + contention_penalty
        # contention_penalty scales with current machine load
        alpha = 0.5  # Weight for contention penalty
        beta = 2.0   # Soft-min temperature (lower = sharper min)
        
        eff_times = []
        machines = []
        for m, dur in machine_options.items():
            load = _machine_loads.get(m, 0.0)
            # Contention penalty: proportional to load, normalized
            cont_penalty = alpha * load / max(1.0, num_options)
            eff_time = dur + cont_penalty
            eff_times.append(eff_time)
            machines.append(m)
        
        eff_times = np.array(eff_times, dtype=np.float64)
        
        # Soft-min via log-sum-exp for smooth surrogate
        # soft_min = -log(sum(exp(-t/beta))) * beta
        # This is a smooth approximation to min(eff_times)
        min_eff = float(np.min(eff_times))
        # For the actual machine selection, use argmin
        best_mach_idx = int(np.argmin(eff_times))
        best_mach_for_op = machines[best_mach_idx]
        
        # Also compute a smooth penalty term using soft-min
        # This helps when multiple machines are similarly loaded
        scaled = -eff_times / beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_val = -log_sum_exp * beta
        
        # Raw min duration
        min_duration = float(min(machine_options.values()))
        
        # Priority score components with better weighting:
        # 1. Critical path urgency: remaining_work weighted by remaining_ops
        #    Higher remaining work relative to ops left = more urgent
        if remaining_ops > 0:
            urgency_term = remaining_work / remaining_ops
        else:
            urgency_term = remaining_work
        
        # 2. Constraint tightness: fewer options = more constrained = higher priority
        tightness_term = 1.0 / max(num_options, 1)
        
        # 3. Contention-aware efficiency: prefer ops that can run fast
        #    Use soft-min for smoothness
        efficiency_term = 1.0 / max(soft_min_val, 1e-9)
        
        # 4. Speed bonus: shorter processing time preferred
        speed_term = 1.0 / max(min_duration, 1e-9)
        
        # 5. Work remaining per op: jobs with more work left should go first
        #    to avoid stragglers
        work_per_op = remaining_work / max(remaining_ops, 1)
        
        # Combine with tuned weights:
        # Urgency and efficiency are primary, tightness and speed secondary
        score = (1.5 * urgency_term + 
                 2.0 * efficiency_term + 
                 1.0 * tightness_term + 
                 0.5 * speed_term +
                 0.3 * work_per_op)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_mach_for_op
    
    # Update machine loads: add the duration of the selected operation to its machine
    if best_machine is not None:
        selected_op = ready_operations[best_index]
        selected_dur = selected_op.get("machine_options", {}).get(best_machine, 0)
        _machine_loads[best_machine] = _machine_loads.get(best_machine, 0.0) + selected_dur
    
    return int(best_index)
# EVOLVE-BLOCK-END
