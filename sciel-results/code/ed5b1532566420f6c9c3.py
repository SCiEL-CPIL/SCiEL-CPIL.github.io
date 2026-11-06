"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation efficiently.
    Uses vectorized numpy operations where possible, but the inner loop
    over machines is sequential due to dependencies.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return 0.0
        
    # completion[m] is the completion time of the last job on machine m
    # We process jobs in the order of `perm`
    # For each job, we calculate its completion time on each machine
    # C_j_m = max(C_{j-1, m}, C_{j, m-1}) + P_{j, m}
    
    # To make this fast in Python, we can try to be smart.
    # However, for moderate n (e.g., < 1000), a tight loop is often okay if optimized.
    # Let's use a simple loop which is robust.
    
    completion = np.zeros(n_machines)
    makespan = 0.0
    
    for j in perm:
        # For job j, we need to update completion times for all machines
        # C_m = max(C_m, C_{m-1}) + P_{j, m}
        # This dependency chain prevents full vectorization easily without complex tricks.
        # But we can iterate.
        
        # Optimization: Pre-extract row? processing_times[j] is a view.
        row = processing_times[j]
        
        prev_completion = 0.0
        for m in range(n_machines):
            # C_m depends on C_{m-1} (current job) and C_m (previous job)
            # completion[m] currently holds C_{prev_job, m}
            # We need max(completion[m], prev_completion) + row[m]
            
            if m == 0:
                new_c = completion[0] + row[0]
            else:
                # completion[m] is C_{prev_job, m}
                # prev_completion is C_{current_job, m-1}
                if completion[m] > prev_completion:
                    new_c = completion[m] + row[m]
                else:
                    new_c = prev_completion + row[m]
            
            completion[m] = new_c
            prev_completion = new_c
            
    return completion[-1]


def _calc_makespan_fast(processing_times, perm):
    """
    Faster makespan calculation using numpy where possible.
    For small problem sizes, the overhead of numpy might be higher, 
    but for larger sizes it helps.
    We will use a hybrid approach or just optimize the loop.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return 0.0
    
    # If n_machines is 1, it's just the sum
    if n_machines == 1:
        return float(processing_times[perm, 0].sum())
    
    # Standard loop is often fastest in pure Python for small N
    # Let's stick to the optimized loop but maybe with local variables
    completion = np.zeros(n_machines)
    
    # Localize variables for speed
    pt = processing_times
    comp = completion
    nm = n_machines
    
    for j in perm:
        row = pt[j]
        prev_c = 0.0
        for m in range(nm):
            if m == 0:
                new_c = comp[0] + row[0]
            else:
                if comp[m] > prev_c:
                    new_c = comp[m] + row[m]
                else:
                    new_c = prev_c + row[m]
            comp[m] = new_c
            prev_c = new_c
            
    return float(comp[-1])


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: NEH Heuristic
    # Sort jobs by total processing time descending
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    # NEH insertion:
    # Start with first job
    current_perm = [sorted_jobs[0]]
    
    # For each subsequent job, insert it at the best position
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Try all positions
        for pos in range(len(current_perm) + 1):
            # Optimization: Use a faster makespan calculation if possible
            # For NEH construction, we can potentially optimize, but let's keep it simple first.
            
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            
            # Use the fast calculator
            ms = _calc_makespan_fast(processing_times, candidate_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
                
        # Insert best
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    best_perm = current_perm
    best_ms = _calc_makespan_fast(processing_times, best_perm)
    
    # 2. Local Search: Or-opt and 2-opt
    # We will iterate until time limit or no improvement.
    
    # Strategy: 
    # - Try all 2-opt moves (reverse a segment)
    # - Try all Or-opt moves (move a block of 1 or 2 jobs)
    # - If improvement, accept and restart search (or continue from new state)
    
    # To avoid O(N^3) complexity in the worst case (if we restart after every improvement),
    # we can use a "first improvement" or "best improvement" strategy per pass.
    # Best improvement: Find the single best move in the neighborhood, apply it, repeat.
    # This is safer for time management.
    
    improved = True
    while improved and (time.time() < deadline - 2.0): # 2s margin for final check
        improved = False
        
        # 2-opt moves
        # For each i < j, reverse perm[i:j+1]
        # There are N(N-1)/2 such moves.
        
        # To speed up, we can sample if N is large.
        # But let's try full neighborhood first if N is small (< 50).
        # If N > 50, sampling might be necessary to fit in time.
        
        if n_jobs <= 50:
            # Full 2-opt
            for i in range(n_jobs - 1):
                for j in range(i + 1, n_jobs):
                    # Reverse segment i to j
                    # Instead of creating new list, we can simulate?
                    # No, _calc_makespan takes a list.
                    # Creating list is O(N).
                    
                    # Optimization: 
                    # We can pre-compute prefix sums? No, flow shop is not additive.
                    
                    # Let's just do it.
                    # To save time, we can break early if time is running out.
                    if time.time() > deadline - 5.0:
                        return best_perm
                        
                    new_perm = best_perm[:i] + best_perm[i:j+1][::-1] + best_perm[j+1:]
                    ms = _calc_makespan_fast(processing_times, new_perm)
                    
                    if ms < best_ms - 1e-9: # Strict improvement
                        best_ms = ms
                        best_perm = new_perm
                        improved = True
                        # Break out of loops to restart search from new best
                        # This is "Best Improvement" within the pass, but we restart the pass.
                        break
                if improved:
                    break
        else:
            # Random sampling for large N
            # Sample 1000 random 2-opt moves
            for _ in range(1000):
                if time.time() > deadline - 5.0:
                    return best_perm
                    
                i = random.randint(0, n_jobs - 2)
                j = random.randint(i + 1, n_jobs - 1)
                
                new_perm = best_perm[:i] + best_perm[i:j+1][::-1] + best_perm[j+1:]
                ms = _calc_makespan_fast(processing_times, new_perm)
                
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
                    break
            
            # Also try Or-opt (move block of size 1 or 2)
            # If 2-opt didn't improve, try Or-opt
            if not improved:
                for _ in range(1000):
                    if time.time() > deadline - 5.0:
                        return best_perm
                        
                    # Random Or-opt
                    block_size = random.choice([1, 2])
                    if block_size == 1:
                        i = random.randint(0, n_jobs - 1)
                        if i == n_jobs - 1: continue
                        # Move job at i to random position
                        job = best_perm[i]
                        # Remove
                        temp_perm = best_perm[:i] + best_perm[i+1:]
                        # Insert at random pos
                        k = random.randint(0, n_jobs - 2)
                        new_perm = temp_perm[:k] + [job] + temp_perm[k:]
                    else:
                        i = random.randint(0, n_jobs - 2)
                        # Move block of 2 starting at i
                        block = best_perm[i:i+2]
                        temp_perm = best_perm[:i] + best_perm[i+2:]
                        k = random.randint(0, n_jobs - 3)
                        # Avoid inserting back to same place roughly
                        new_perm = temp_perm[:k] + block + temp_perm[k:]
                        
                    ms = _calc_makespan_fast(processing_times, new_perm)
                    if ms < best_ms - 1e-9:
                        best_ms = ms
                        best_perm = new_perm
                        improved = True
                        break

    return best_perm
# EVOLVE-BLOCK-END
