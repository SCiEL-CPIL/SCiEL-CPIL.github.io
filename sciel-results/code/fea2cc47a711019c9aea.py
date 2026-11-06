"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Fast makespan calculation for flow shop.
    perm: list of job indices
    processing_times: numpy array (N, M)
    """
    if not perm:
        return 0.0
    
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(np.sum(processing_times[perm, 0]))
        
    # completion[m] holds the completion time of the last job on machine m
    completion = [0.0] * n_machines
    
    for job_idx in perm:
        row = processing_times[job_idx]
        prev_c = 0.0
        for m in range(n_machines):
            # c_prev = completion[m]
            # c_left = prev_c
            # new_c = max(c_prev, c_left) + row[m]
            if completion[m] > prev_c:
                new_c = completion[m] + row[m]
            else:
                new_c = prev_c + row[m]
            completion[m] = new_c
            prev_c = new_c
            
    return completion[-1]

def _local_search_oropt(processing_times, perm, time_limit=0.1):
    """
    Local search using Or-opt moves (moving a segment of length 1, 2, or 3).
    Returns improved permutation and makespan.
    """
    current_perm = perm[:]
    current_ms = _calc_makespan(processing_times, current_perm)
    n_jobs = len(current_perm)
    
    if n_jobs <= 1:
        return current_perm, current_ms
    
    # We will iterate until no improvement is found or time limit is reached
    # For small N, a single pass of best-improvement Or-opt is often sufficient.
    # To be safe within tight time limits, we do a limited number of passes.
    
    max_passes = 5
    for _ in range(max_passes):
        improved = False
        # Try moving segments of length 1, 2, 3
        for seg_len in [1, 2, 3]:
            if n_jobs < seg_len + 1:
                continue
            for i in range(n_jobs - seg_len + 1):
                # Extract segment
                segment = current_perm[i:i+seg_len]
                # Remove segment
                remaining = current_perm[:i] + current_perm[i+seg_len:]
                # Try inserting at all possible positions in remaining
                # Optimization: we don't need to try inserting at i or i+seg_len 
                # effectively, but checking all is safer.
                # We can skip positions that result in the same order.
                for j in range(len(remaining) + 1):
                    # Skip if inserting at original position
                    if j == i or j == i + seg_len:
                        continue
                    
                    new_perm = remaining[:j] + segment + remaining[j:]
                    new_ms = _calc_makespan(processing_times, new_perm)
                    
                    if new_ms < current_ms:
                        current_perm = new_perm
                        current_ms = new_ms
                        improved = True
                        # Restart i loop? Or just continue? 
                        # Continuing is faster, might find more improvements in same pass.
                        # But indices change. To be safe and fast, we can break out 
                        # and restart the pass. However, for small N, continuing 
                        # with updated list is risky if we don't adjust indices.
                        # Let's just break and restart the outer loop for simplicity and correctness.
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
            
    return current_perm, current_ms

def _perturb(perm, n_perturb):
    """
    Perturb permutation by reversing a random segment or moving a random segment.
    """
    n = len(perm)
    if n < 2:
        return perm[:]
    
    # Randomly choose between 2-opt (reverse) and Or-opt (move)
    if random.random() < 0.5:
        # 2-opt reverse
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
    else:
        # Or-opt move: move a random segment to a random position
        seg_len = random.choice([1, 2, 3])
        if n < seg_len + 1:
            seg_len = max(1, n - 1)
        
        i = random.randint(0, n - seg_len)
        segment = perm[i:i+seg_len]
        remaining = perm[:i] + perm[i+seg_len:]
        
        # Choose insertion point in remaining
        j = random.randint(0, len(remaining))
        new_perm = remaining[:j] + segment + remaining[j:]
        
    return new_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Generate diverse initial solutions
    # Strategy: Multi-start ILS
    # We will run several independent ILS loops, each starting from a different seed.
    
    seeds = []
    
    # Seed 1: NEH (Total processing time descending)
    totals = processing_times.sum(axis=1)
    sorted_jobs_neh = np.argsort(-totals)
    
    # Implement simple NEH construction
    perm_neh = [sorted_jobs_neh[0]]
    for i in range(1, n_jobs):
        job = sorted_jobs_neh[i]
        best_pos = 0
        best_ms = float('inf')
        # For efficiency in construction, limit search if N is large? 
        # NEH is O(N^2 * M). For N=50, 2500*20 = 50k ops. Fast.
        for pos in range(len(perm_neh) + 1):
            candidate = perm_neh[:pos] + [job] + perm_neh[pos:]
            ms = _calc_makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm_neh = perm_neh[:best_pos] + [job] + perm_neh[best_pos:]
    seeds.append(perm_neh)
    
    # Seed 2: LPT (Longest Processing Time on first machine, or just random if M=1)
    # Let's use a heuristic based on the first machine or average
    if n_machines > 0:
        # Sort by processing time on machine 0 descending
        sorted_jobs_lpt = np.argsort(-processing_times[:, 0])
        seeds.append(sorted_jobs_lpt.tolist())
        
    # Seed 3: Random
    seeds.append(list(np.random.permutation(n_jobs)))
    
    # Remove duplicates (convert to tuple for comparison)
    unique_seeds = []
    seen = set()
    for s in seeds:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            unique_seeds.append(s)
    
    # 2. Iterated Local Search for each seed
    # We distribute time among the seeds.
    # If we have 3 seeds, we give each ~1/3 of the time.
    # But if one seed is much better, we might want to spend more time there?
    # Simple approach: run ILS for each seed for a fixed fraction of time.
    
    best_perm_global = unique_seeds[0]
    best_ms_global = _calc_makespan(processing_times, best_perm_global)
    
    time_per_seed = (deadline - time.time()) / len(unique_seeds)
    
    for seed_idx, seed_perm in enumerate(unique_seeds):
        # Update global best with seed
        seed_ms = _calc_makespan(processing_times, seed_perm)
        if seed_ms < best_ms_global:
            best_ms_global = seed_ms
            best_perm_global = seed_perm[:]
            
        # ILS loop for this seed
        current_perm = seed_perm[:]
        current_ms = seed_ms
        
        # Local search to start
        current_perm, current_ms = _local_search_oropt(processing_times, current_perm, time_limit=time_per_seed * 0.2)
        
        if current_ms < best_ms_global:
            best_ms_global = current_ms
            best_perm_global = current_perm[:]
            
        # Iterate
        while time.time() < deadline - 2.0:
            # Check if we should break if this seed is taking too long relative to others
            # Simple check: if we've spent more than time_per_seed, break
            if time.time() - start_time > deadline - 2.0 - (time_per_seed * (len(unique_seeds) - seed_idx - 1)):
                 break
            
            # Perturb
            perturbed_perm = _perturb(current_perm, 2) # 2 is a dummy arg, we use random internally
            
            # Local search on perturbed
            new_perm, new_ms = _local_search_oropt(processing_times, perturbed_perm, time_limit=time_per_seed * 0.1)
            
            # Acceptance criterion: Accept if better, or with some probability if worse?
            # Standard ILS accepts if better. To escape local minima, we can accept worse 
            # with a small probability or if it leads to a better basin.
            # Simple: Accept if new_ms < current_ms. 
            # To add diversity, if new_ms >= current_ms, we might still accept it 
            # if we haven't improved global best in a while? 
            # Let's stick to: Accept if new_ms < current_ms. 
            # If stuck, the perturbation strength could be increased, but let's keep it simple.
            
            if new_ms < current_ms:
                current_perm = new_perm
                current_ms = new_ms
                if current_ms < best_ms_global:
                    best_ms_global = current_ms
                    best_perm_global = current_perm[:]
            else:
                # With some probability, accept worse to escape?
                # Or just reject and perturb from current again.
                # Let's try a small acceptance probability for worse solutions to help exploration
                if random.random() < 0.05:
                    current_perm = new_perm
                    current_ms = new_ms
                    
    return best_perm_global
# EVOLVE-BLOCK-END
