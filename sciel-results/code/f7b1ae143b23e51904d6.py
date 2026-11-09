"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Strategy: Composite priority heuristic for Open-Shop scheduling
    # Primary: Remaining work (LRPT) - jobs with more work left should be prioritized
    # Secondary: Machine load balance - prefer operations on less-loaded machines
    # Tertiary: Shortest processing time among available machines
    
    # Compute max remaining work for normalization
    max_rw = 0.0
    for op in ready_operations:
        rw = op["remaining_work"]
        if rw > max_rw:
            max_rw = rw
    
    if max_rw <= 0:
        max_rw = 1.0

    # Compute machine load estimates from the ready operations
    # A machine that appears in many ready operations is "busy"
    machine_load = {}
    for op in ready_operations:
        options = op["machine_options"]
        for m in options:
            if m not in machine_load:
                machine_load[m] = 0
            machine_load[m] += 1
    
    # Normalize machine loads
    max_load = 1
    for m, load in machine_load.items():
        if load > max_load:
            max_load = load
    if max_load <= 0:
        max_load = 1

    best_idx = 0
    best_score = -float('inf')

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        
        # Normalize remaining work (0 to 1)
        rw_norm = rw / max_rw
        
        # Find the best machine option for this operation
        options = op["machine_options"]
        if not options:
            min_dur = 0.0
            best_machine = None
            best_machine_load_norm = 0.0
        else:
            # Among available machines, find the one with shortest duration
            # Tie-break: prefer less loaded machines
            min_dur = float('inf')
            best_machine = None
            best_machine_load_norm = float('inf')
            for m, dur in options.items():
                load_norm = machine_load.get(m, 0) / max_load
                # Prefer shorter duration, then less loaded machine
                if dur < min_dur - 1e-9:
                    min_dur = dur
                    best_machine = m
                    best_machine_load_norm = load_norm
                elif abs(dur - min_dur) < 1e-9:
                    if load_norm < best_machine_load_norm:
                        best_machine = m
                        best_machine_load_norm = load_norm
            
            if best_machine is None:
                min_dur = 0.0
                best_machine_load_norm = 0.0
        
        # Normalize min duration (lower is better, so invert)
        # We'll use a small weight for this component
        # Estimate typical duration scale
        dur_norm = 1.0  # Placeholder, will compute relative below
        
        # Composite score:
        # Primary: remaining work (higher is better)
        # Secondary: prefer operations on less-loaded machines (lower load is better)
        # Tertiary: shorter processing time is better
        
        # Machine load component: less loaded is better, so (1 - load_norm)
        machine_score = 1.0 - best_machine_load_norm
        
        # Duration component: shorter is better
        # We need a scale; use max_rw as a rough proxy or just use raw with small weight
        # Since durations can vary, use a relative measure
        # For now, use a small weight on the inverse duration
        if min_dur > 0:
            dur_score = 1.0 / (1.0 + min_dur)
        else:
            dur_score = 1.0
        
        # Weighted composite:
        # 0.7 * rw_norm + 0.2 * machine_score + 0.1 * dur_score
        score = 0.7 * rw_norm + 0.2 * machine_score + 0.1 * dur_score
        
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx
# EVOLVE-BLOCK-END
