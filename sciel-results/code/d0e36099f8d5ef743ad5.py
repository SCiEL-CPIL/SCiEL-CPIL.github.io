"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE_MARGIN = 5.0
_START_TIME = None

# Global state to track machine loads across dispatch calls
# Since the simulation is event-driven, we can maintain a persistent view of how much work
# has been assigned to each machine so far.
_machine_loads = {}
_total_assigned_work = 0.0
_dispatch_count = 0

def _now():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME

def _deadline_exceeded():
    return _now() > (150.0 - _DEADLINE_MARGIN)

def _get_machine_load(m):
    """Get the current cumulative load for a specific machine."""
    return _machine_loads.get(m, 0.0)

def _update_machine_load(m, duration):
    """Update the cumulative load for a machine after assigning an operation."""
    _machine_loads[m] = _machine_loads.get(m, 0.0) + duration
    global _total_assigned_work
    _total_assigned_work += duration

def dispatch(ready_operations, time):
    """
    Dispatch rule based on Dynamic Machine-Load Balancing.
    
    Strategy:
    1. Maintain a global view of cumulative processing time assigned to each machine.
    2. For each ready operation, consider all possible machine assignments.
    3. For each (operation, machine) pair, simulate the assignment and calculate
       the resulting maximum machine load (makespan lower bound).
    4. Select the (operation, machine) pair that minimizes this maximum load.
    5. Tie-breaker: Minimize the variance of machine loads (balance).
    6. Tie-breaker: Shortest Processing Time (SPT).
    
    This approach directly targets the makespan by preventing any single machine
    from becoming a bottleneck relative to others.
    """
    global _dispatch_count
    _dispatch_count += 1

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        # If only one op is ready, we still need to pick the best machine for it
        # to maintain the global load balance strategy for future calls.
        op = ready_operations[0]
        opts = op.get("machine_options", {})
        if not opts:
            return 0
            
        best_machine = None
        best_max_load = float('inf')
        best_variance = float('inf')
        
        for m, dur in opts.items():
            # Simulate assignment
            new_load = _machine_loads.get(m, 0.0) + dur
            
            # Calculate resulting max load
            max_load = new_load
            for other_m, other_load in _machine_loads.items():
                if other_m == m:
                    continue
                if other_load > max_load:
                    max_load = other_load
            # Also consider machines not in _machine_loads yet (load 0)
            # But since we track all touched machines, we need to be careful.
            # Actually, machines with 0 load are fine. The max is determined by the busiest ones.
            
            # Calculate variance for tie-breaking
            # We only care about relative balance among machines that have work or this new work.
            # Simple metric: sum of squares of loads (proxy for variance/imbalance)
            sq_sum = new_load * new_load
            for other_m, other_load in _machine_loads.items():
                if other_m == m:
                    continue
                sq_sum += other_load * other_load
            
            # Primary: Minimize Max Load
            # Secondary: Minimize Sum of Squares (Balance)
            if max_load < best_max_load or (max_load == best_max_load and sq_sum < best_variance):
                best_max_load = max_load
                best_variance = sq_sum
                best_machine = m

        if best_machine is None:
            return 0

        # Update global state
        _update_machine_load(best_machine, opts[best_machine])
        
        # The return value should be the index of the operation.
        # The problem statement says: "dispatch(ready_operations, time) -> index of the ready operation to start (optionally (index, machine))"
        # The current seed returns just the index. The evaluator likely handles machine selection internally 
        # OR the evaluator expects us to return just the index and it picks the machine?
        # Re-reading contract: "dispatch(ready_operations, time) -> index of the ready operation to start (optionally (index, machine))"
        # "each ready operation is a dict with ... machine_options {machine: duration}"
        # If I return just the index, who picks the machine?
        # Usually in these tasks, if the return type allows a tuple, we should provide it to control machine selection.
        # If I return just an index, the simulator might pick the first available or min duration machine.
        # To be safe and effective with a load-balancing strategy, I MUST control the machine selection.
        # I will return (index, machine).
        
        return (0, best_machine)

    if _deadline_exceeded():
        # Fallback: Simple SPT on min duration machine
        best_idx = 0
        best_dur = float('inf')
        best_mach = -1
        for i, op in enumerate(ready_operations):
            opts = op.get("machine_options", {})
            if not opts:
                continue
            m, dur = min(opts.items(), key=lambda x: x[1])
            if dur < best_dur:
                best_dur = dur
                best_idx = i
                best_mach = m
        if best_mach != -1:
            _update_machine_load(best_mach, best_dur)
            return (best_idx, best_mach)
        return best_idx

    n = len(ready_operations)
    
    # We need to find the best (op_index, machine) pair globally.
    best_op_idx = 0
    best_machine = -1
    best_score = float('inf')
    best_secondary = float('inf')
    best_tertiary = float('inf')
    
    # Pre-compute current max load and sum of squares for baseline
    current_max_load = 0.0
    current_sq_sum = 0.0
    for m, load in _machine_loads.items():
        if load > current_max_load:
            current_max_load = load
        current_sq_sum += load * load
    
    for i, op in enumerate(ready_operations):
        opts = op.get("machine_options", {})
        if not opts:
            continue
            
        for m, dur in opts.items():
            # Simulate assigning op i to machine m
            new_load_m = _machine_loads.get(m, 0.0) + dur
            
            # Calculate new Max Load
            new_max_load = new_load_m
            for other_m, other_load in _machine_loads.items():
                if other_m == m:
                    continue
                if other_load > new_max_load:
                    new_max_load = other_load
            
            # Calculate new Sum of Squares (for balance)
            # old_sq_sum - old_load_m^2 + new_load_m^2
            old_load_m = _machine_loads.get(m, 0.0)
            new_sq_sum = current_sq_sum - (old_load_m * old_load_m) + (new_load_m * new_load_m)
            
            # Primary Objective: Minimize Max Load (Makespan Lower Bound)
            # Secondary Objective: Minimize Variance/Imbalance (Sum of Squares)
            # Tertiary Objective: Shortest Duration (SPT) to free up capacity quickly
            
            if new_max_load < best_score or \
               (new_max_load == best_score and new_sq_sum < best_secondary) or \
               (new_max_load == best_score and new_sq_sum == best_secondary and dur < best_tertiary):
                best_score = new_max_load
                best_secondary = new_sq_sum
                best_tertiary = dur
                best_op_idx = i
                best_machine = m

    # If no valid machine was found (should not happen), fallback to index 0
    if best_machine == -1:
        return 0

    # Update global state with the chosen assignment
    _update_machine_load(best_machine, ready_operations[best_op_idx]["machine_options"][best_machine])
    
    # Return tuple (index, machine) to ensure the simulator uses our chosen machine
    return (best_op_idx, best_machine)

# EVOLVE-BLOCK-END
