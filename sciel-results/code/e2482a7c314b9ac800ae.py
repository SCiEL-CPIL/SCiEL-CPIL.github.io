"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s kill
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = _time.monotonic()
    return _time.monotonic() - _START


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.
    
    Strategy: Machine-load balancing with a preference for operations on
    the most congested machines, prioritizing shorter durations to reduce
    bottleneck. We compute a composite score that favors:
      1. Operations targeting the most loaded/congested machines (to avoid
         creating a single-machine bottleneck)
      2. Among those, shorter durations are preferred (to free up the
         congested machine sooner)
      3. Remaining work as a tiebreaker
    
    We also incorporate a look-ahead: for each ready operation, we consider
    which machine it would use (the one with the smallest duration among
    its options, as that minimizes time on that machine), and we penalize
    operations that would go to already-busy machines.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _elapsed() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)

    # For each ready operation, determine the best machine (shortest duration)
    # and collect relevant features
    scores = np.zeros(n_ops)
    best_machine_durations = np.zeros(n_ops)
    machine_loads = {}  # machine -> estimated remaining load

    # First pass: compute machine loads from all ready operations
    # (a proxy for congestion)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        # The machine with the shortest duration is the "best" choice
        best_mach = min(mo, key=lambda m: mo[m])
        best_dur = mo[best_mach]
        best_machine_durations[i] = best_dur
        # Accumulate load on that machine
        if best_mach not in machine_loads:
            machine_loads[best_mach] = 0.0
        machine_loads[best_mach] += best_dur

    # Also factor in remaining_work as a proxy for total future load per job
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])

    # Normalize features
    max_dur = np.max(best_machine_durations) if np.max(best_machine_durations) > 0 else 1.0
    max_mach_load = max(machine_loads.values()) if machine_loads else 1.0
    max_rw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_ro = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0

    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        best_mach = min(mo, key=lambda m: mo[m])
        best_dur = mo[best_mach]
        
        # Machine congestion: how loaded is the best machine relative to others
        mach_load = machine_loads.get(best_mach, 0.0) / max_mach_load
        
        # We want to prefer operations that go to LESS loaded machines
        # (to balance load), but also consider that if a machine is very
        # loaded, we might still want to do a short op on it to clear it.
        # Composite: prefer short duration on less-loaded machines.
        # Score = w1 * (1 - mach_load) + w2 * (1 - dur/max_dur) + w3 * (1 - rw/max_rw)
        # Higher score = better
        
        dur_score = 1.0 - (best_dur / max_dur)
        load_score = 1.0 - mach_load
        rw_score = 1.0 - (op["remaining_work"] / max_rw)
        ro_score = 1.0 - (op["remaining_ops"] / max_ro)
        
        # Weighted combination
        w_dur = 0.35
        w_load = 0.35
        w_rw = 0.20
        w_ro = 0.10
        
        scores[i] = w_dur * dur_score + w_load * load_score + w_rw * rw_score + w_ro * ro_score

    # Return index of best score
    best_idx = int(np.argmax(scores))
    
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return best_idx
    
    return best_idx
# EVOLVE-BLOCK-END
