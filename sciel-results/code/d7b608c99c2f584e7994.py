"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0  # Safety margin

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: MWR (Most Work Remaining)
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0

    # Lookahead Strategy:
    # For each candidate ready operation, simulate the greedy dispatch of all ready operations
    # starting with that candidate. Estimate the makespan of the ready set under this ordering.
    # Pick the candidate that minimizes the estimated makespan.
    
    best_idx = 0
    best_estimated_makespan = float('inf')
    
    # Pre-compute min durations for speed
    min_durs = []
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if mo:
            min_durs.append(min(mo.values()))
        else:
            min_durs.append(0)
    
    # Simulate for each candidate
    for start_idx in range(n):
        # Virtual simulation
        # We assume all machines are free at 'time' for the purpose of estimating the relative order
        # Actually, a better estimate is to assume the candidate starts at 'time', 
        # and subsequent ops start when the previous one finishes (on the same machine or different).
        # Since we don't know machine availability, a common heuristic is to assume 
        # the "critical path" of the ready set.
        
        # Simple lookahead: 
        # 1. Start candidate at `time`
        # 2. For remaining ops, assume they start when the previous one finishes (serial) 
        #    OR in parallel if different machines. 
        #    Since we don't know machine availability, let's use a "virtual machine" model 
        #    where we track the earliest time each machine is free in the virtual simulation.
        
        # However, we don't know which machine is chosen. 
        # Let's assume the greedy choice: pick the machine that minimizes completion time 
        # given current virtual machine availability.
        
        # Virtual state: machine -> available_time
        # But we don't know the machine IDs. We can infer them from the first op.
        # Let's gather all machine IDs.
        machine_ids = set()
        for op in ready_operations:
            mo = op.get("machine_options", {})
            machine_ids.update(mo.keys())
        
        # Virtual machine availability starts at 'time' for all
        virtual_avail = {m: time for m in machine_ids}
        
        # Order: candidate first, then others sorted by some heuristic (e.g., min duration or critical path)
        # To keep it fast, we just simulate the candidate starting now, and then 
        # greedily schedule the rest based on their min duration or critical path.
        
        # Actually, a full simulation for each candidate is O(N^2 * M) per dispatch call.
        # With N=10, M=5, this is 100*5 = 500 ops per dispatch. 
        # Dispatch is called ~50 times. 25,000 ops total. Very fast.
        
        # Let's refine the simulation:
        # 1. Pick candidate `start_idx`.
        # 2. Determine its best machine given virtual_avail.
        # 3. Update virtual_avail.
        # 4. For the remaining ops, repeatedly pick the one with highest "urgency" (e.g., remaining_work) 
        #    and assign it to its best available machine.
        # 5. Record the max completion time.
        
        # Candidate
        cand_op = ready_operations[start_idx]
        mo = cand_op.get("machine_options", {})
        best_mach = None
        best_end = float('inf')
        for m, d in mo.items():
            start_t = virtual_avail[m]
            end_t = start_t + d
            if end_t < best_end:
                best_end = end_t
                best_mach = m
        
        if best_mach is not None:
            virtual_avail[best_mach] = best_end
        
        # Remaining ops
        remaining_indices = [i for i in range(n) if i != start_idx]
        
        # Sort remaining by a static priority to keep simulation deterministic and fast
        # e.g., by remaining_work descending
        remaining_indices.sort(key=lambda i: ready_operations[i].get("remaining_work", 0), reverse=True)
        
        current_max_end = best_end
        
        for idx in remaining_indices:
            op = ready_operations[idx]
            mo = op.get("machine_options", {})
            if not mo:
                continue
            
            best_mach = None
            best_end = float('inf')
            for m, d in mo.items():
                start_t = virtual_avail[m]
                end_t = start_t + d
                if end_t < best_end:
                    best_end = end_t
                    best_mach = m
            
            if best_mach is not None:
                virtual_avail[best_mach] = best_end
                if best_end > current_max_end:
                    current_max_end = best_end
        
        # The estimated makespan for this candidate is current_max_end
        if current_max_end < best_estimated_makespan:
            best_estimated_makespan = current_max_end
            best_idx = start_idx

    return int(best_idx)

# EVOLVE-BLOCK-END
