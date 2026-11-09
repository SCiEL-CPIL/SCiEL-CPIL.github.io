"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0

# Global state to track estimated machine loads
# Key: machine_id, Value: estimated total load (sum of durations assigned)
_machine_loads = {}

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: MWR (Most Work Remaining)
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    try:
        if not ready_operations:
            return 0
        
        n = len(ready_operations)
        
        if n == 1:
            # If only one op, we still need to pick a machine if possible
            # But the interface says "index of ready operation" OR "(index, machine)"
            # The current seed returns just index. 
            # The prompt says "optionally (index, machine)".
            # Let's try returning (index, machine) to actively balance load.
            op = ready_operations[0]
            mo = op.get("machine_options", {})
            if mo:
                # Pick the machine with the lowest current estimated load
                best_machine = min(mo.keys(), key=lambda m: _machine_loads.get(m, 0))
                # Update the load for this machine
                dur = mo[best_machine]
                _machine_loads[best_machine] = _machine_loads.get(best_machine, 0) + dur
                return (0, best_machine)
            else:
                return 0

        best_pair_score = -1e10
        best_index = 0
        best_machine = None
        
        # We evaluate each (operation, machine) pair
        # Heuristic: Minimize the "completion time" of the operation on that machine
        # Estimated completion time = Current Time + Estimated Machine Load + Duration
        # Since 'time' is passed, we can use it, but relative load is more important.
        # Let's use a score that favors:
        # 1. Low estimated completion time (Load + Duration)
        # 2. High criticality (Remaining Work) to ensure we don't ignore urgent jobs
        
        # To balance these, we can use a weighted sum or a ratio.
        # Let's try: Score = Remaining_Work / (Estimated_Completion_Time + 1)
        # This prioritizes jobs with lots of work that can be finished quickly on a free machine.
        
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            mo = op.get("machine_options", {})
            if not mo:
                # Fallback if no machine options (should not happen)
                continue
                
            # Find the machine that minimizes (Load + Duration) for this op
            # This is the "best" machine for this specific operation right now
            best_m_for_op = None
            min_est_completion = float('inf')
            
            for m, dur in mo.items():
                current_load = _machine_loads.get(m, 0)
                est_completion = current_load + dur
                if est_completion < min_est_completion:
                    min_est_completion = est_completion
                    best_m_for_op = m
            
            if best_m_for_op is None:
                continue
                
            # Calculate score for this operation using its best machine
            # We want to maximize the score.
            # Higher RW is good. Lower est_completion is good.
            # Score = RW / (est_completion + 1)
            score = rw / (min_est_completion + 1.0)
            
            # Tie breaker: If scores are very close, prefer the one with lower est_completion
            # or higher remaining work.
            
            if score > best_pair_score + 1e-9:
                best_pair_score = score
                best_index = i
                best_machine = best_m_for_op
            elif abs(score - best_pair_score) <= 1e-9:
                # Tie breaker: Pick the one with lower estimated completion time
                # We need to re-calculate or store it. Let's store it.
                # For simplicity in the loop, we can just check if this one is better in tie-break
                # But we didn't store min_est_completion for the current best.
                # Let's refine: store the best est_completion too.
                pass # This logic is slightly incomplete in the single-pass above. 
                # Let's rewrite the loop to store candidates properly.

        # The above loop has a flaw in tie-breaking. Let's do a cleaner pass.
        # Reset
        best_pair_score = -1e10
        best_index = 0
        best_machine = None
        best_est_completion = float('inf')
        
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            mo = op.get("machine_options", {})
            if not mo:
                continue
                
            best_m_for_op = None
            min_est_completion = float('inf')
            
            for m, dur in mo.items():
                current_load = _machine_loads.get(m, 0)
                est_completion = current_load + dur
                if est_completion < min_est_completion:
                    min_est_completion = est_completion
                    best_m_for_op = m
            
            if best_m_for_op is None:
                continue
                
            score = rw / (min_est_completion + 1.0)
            
            if score > best_pair_score + 1e-9:
                best_pair_score = score
                best_index = i
                best_machine = best_m_for_op
                best_est_completion = min_est_completion
            elif abs(score - best_pair_score) <= 1e-9:
                # Tie breaker: Lower estimated completion time is better
                if min_est_completion < best_est_completion - 1e-9:
                    best_index = i
                    best_machine = best_m_for_op
                    best_est_completion = min_est_completion
                elif abs(min_est_completion - best_est_completion) <= 1e-9:
                    # Tie breaker 2: Higher remaining work
                    if rw > ready_operations[best_index].get("remaining_work", 0):
                        best_index = i
                        best_machine = best_m_for_op
                        best_est_completion = min_est_completion

        # Update the global load for the chosen machine
        if best_machine is not None:
            chosen_op = ready_operations[best_index]
            dur = chosen_op.get("machine_options", {}).get(best_machine, 0)
            _machine_loads[best_machine] = _machine_loads.get(best_machine, 0) + dur
            
        # Return (index, machine) to actively assign the machine
        if best_machine is not None:
            return (best_index, best_machine)
        else:
            return best_index

    except Exception:
        if not ready_operations:
            return 0
        return 0

# EVOLVE-BLOCK-END
