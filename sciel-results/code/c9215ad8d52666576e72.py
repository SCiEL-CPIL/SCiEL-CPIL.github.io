"""Open-shop dispatch rule: Composite CPM + Load-Balancing with smooth scoring."""
# EVOLVE-BLOCK-START
import time
import math

# Global time budget tracking
_START_TIME = time.time()
# Evaluator kills at 150s wall-clock. We reserve safety margin.
_DEADLINE = _START_TIME + 145.0

# State tracking for load balancing
_machine_total_load = {}
# Track max remaining work observed to normalize
_max_observed_remaining = 0.0

def _time_left():
    return _DEADLINE - time.time()


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Compute a composite score for each ready operation combining:
       - Critical path urgency: normalized remaining_work of its job
       - Machine load: minimum possible new load across machine options
       - Processing time efficiency: minimum duration available
    2. Select the operation with the best composite score.
    3. For that operation, assign the machine that minimizes the resulting load.
    4. Return (index, machine) to explicitly assign the machine.
    """
    global _max_observed_remaining
    
    if not ready_operations:
        return 0
    
    # Time guard: if we're running low on time, use a simple fallback
    if _time_left() < 0.5:
        max_remaining = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_remaining:
                max_remaining = rw
                best_idx = i
        return best_idx

    # Update max observed remaining for normalization
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        if rw > _max_observed_remaining:
            _max_observed_remaining = rw

    # If no scale yet, use 1.0 to avoid division by zero
    scale = _max_observed_remaining if _max_observed_remaining > 0 else 1.0

    # Compute composite scores for all operations
    # Score = alpha * urgency + beta * (min_new_load / scale) + gamma * (min_duration / scale)
    # We want to MINIMIZE this score.
    # Urgency: higher remaining_work = more urgent = should have LOWER score contribution
    # So we use (1 - urgency_normalized) or negative urgency
    
    alpha = 0.6  # Weight for critical path urgency
    beta = 0.3   # Weight for load balancing
    gamma = 0.1  # Weight for processing time efficiency

    best_score = float('inf')
    best_idx = 0
    best_machine = None
    best_min_duration = float('inf')
    best_min_new_load = float('inf')

    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        machine_options = op["machine_options"]
        
        if not machine_options:
            continue
        
        # Compute min duration and min new load across machine options
        min_duration = float('inf')
        min_new_load = float('inf')
        
        for m, d in machine_options.items():
            current_load = _machine_total_load.get(m, 0.0)
            new_load = current_load + d
            if d < min_duration:
                min_duration = d
            if new_load < min_new_load:
                min_new_load = new_load

        # Normalize urgency: higher rw = more urgent
        # We want urgent jobs to have lower composite score
        urgency_norm = rw / scale if scale > 0 else 0.0
        # Invert: less urgent = higher contribution
        urgency_contrib = 1.0 - urgency_norm

        # Normalize load contribution
        load_contrib = min_new_load / scale if scale > 0 else 0.0

        # Normalize duration contribution
        duration_contrib = min_duration / scale if scale > 0 else 0.0

        composite = alpha * urgency_contrib + beta * load_contrib + gamma * duration_contrib

        if composite < best_score:
            best_score = composite
            best_idx = i
            best_machine = None
            best_min_duration = min_duration
            best_min_new_load = min_new_load

    # Now find the best machine for the selected operation
    if best_machine is None:
        # Find the machine that minimizes new load for the chosen operation
        op = ready_operations[best_idx]
        machine_options = op["machine_options"]
        best_machine = None
        best_machine_load = float('inf')
        best_machine_duration = float('inf')
        
        for m, d in machine_options.items():
            current_load = _machine_total_load.get(m, 0.0)
            new_load = current_load + d
            if new_load < best_machine_load or (new_load == best_machine_load and d < best_machine_duration):
                best_machine_load = new_load
                best_machine_duration = d
                best_machine = m

    if best_machine is None:
        # Fallback: pick first available machine
        op = ready_operations[best_idx]
        machine_options = op["machine_options"]
        if machine_options:
            best_machine = list(machine_options.keys())[0]
            best_machine_load = _machine_total_load.get(best_machine, 0.0) + machine_options[best_machine]
        else:
            return best_idx

    # 3. Update global state
    _machine_total_load[best_machine] = best_machine_load

    # Return tuple to specify machine assignment
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
