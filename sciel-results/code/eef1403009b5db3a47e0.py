"""
Flow-shop permutation via structural change: Tabu Search with dynamic memory.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(pt, perm):
    """
    Compute makespan for a given permutation.
    O(N*M) DP approach.
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    prev = [0.0] * n_machines
    pt_local = pt
    n_m = n_machines
    
    for j in perm:
        curr = [0.0] * n_m
        curr[0] = prev[0] + pt_local[j, 0]
        for m in range(1, n_m):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + pt_local[j, m]
        prev = curr
    return prev[-1]

def _neh_initial(pt):
    """
    NEH heuristic: Sort jobs by total processing time descending.
    Insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _palmer_initial(pt):
    """
    Palmer's heuristic (WSPT variant).
    """
    n_jobs, n_machines = pt.shape
    weights = np.arange(n_machines, 0, -1)
    priorities = pt.dot(weights)
    order = list(np.argsort(-priorities))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _random_initial(n_jobs, seed):
    """Random initial solution."""
    rng = np.random.RandomState(seed)
    return rng.permutation(n_jobs).tolist()

def _tabu_search(pt, perm, deadline, tabu_tenure=7):
    """
    Tabu Search with insertion neighborhood.
    Uses a short-term memory (tabu list) to prevent cycling.
    Uses a diversification strategy (shifting focus to a random job) if stagnating.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    # Tabu memory: tabu[job][pos] = iteration when move (job to pos) becomes non-tabu
    # Or simpler: tabu_moves set of (job, target_pos)
    # Since N is small, we can use a dictionary or a 2D array.
    # Let's use a dictionary: key = (job, target_pos), value = iteration to untabu
    tabu_dict = {}
    
    # Diversification counter
    stagnation_count = 0
    max_stagnation = 100
    
    rng = np.random.RandomState(123)
    iteration = 0
    
    while time.time() < deadline:
        improved = False
        best_move_ms = float('inf')
        best_move_perm = None
        best_move_key = None
        
        # Check time periodically
        if iteration % 10 == 0 and time.time() > deadline:
            break
        
        # Generate neighbors
        # To speed up, we don't evaluate ALL neighbors if we find a good one early?
        # No, Tabu usually requires best-neighbor selection to be effective.
        # But N^2 evaluations per step might be slow if N is large.
        # For N=100, 10000 evaluations * 100 machines = 1M ops. 150s / 1M ops = 150s for 150 steps?
        # No, 150s is plenty for thousands of steps if optimized.
        # Let's limit the neighborhood size if N is large?
        # For now, assume N is small enough (< 50) or time is sufficient.
        
        for i in range(n_jobs):
            job = current_perm[i]
            # Remove job
            base_perm = current_perm[:i] + current_perm[i+1:]
            
            for k in range(n_jobs):
                # k is the insertion index in the ORIGINAL perm?
                # No, k is index in base_perm (length n_jobs-1)
                # If we insert at k in base_perm, the final position in new_perm is k.
                # But we need to map back to "move job i to position k".
                # If k < i, the job moves left. If k >= i, the job moves right (since i is removed, indices shift).
                
                # Skip if this is the same position (no move)
                # If k == i, it's inserting back at same spot (relative to base, it's i, which is original i+1? No.)
                # Let's just check if new_perm == current_perm.
                # new_perm = base_perm[:k] + [job] + base_perm[k:]
                # If k == i, new_perm is current_perm.
                if k == i:
                    continue
                
                # Check tabu
                # Key: (job, k) where k is insertion index in base_perm
                # Note: The "position" in the final permutation is k.
                key = (job, k)
                
                if key in tabu_dict and tabu_dict[key] > iteration:
                    continue
                
                new_perm = base_perm[:k] + [job] + base_perm[k:]
                new_ms = _makespan(pt, new_perm)
                
                # Aspiration criterion: if new_ms < best_ms, allow even if tabu
                if new_ms < best_ms:
                    is_tabu = False
                else:
                    is_tabu = (key in tabu_dict and tabu_dict[key] > iteration)
                    if is_tabu:
                        continue
                
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move_perm = new_perm
                    best_move_key = key
                    improved = True
        
        if not improved:
            # No non-tabu moves found? Or all worse?
            # If all moves are tabu, pick the best tabu move (aspiration) or break?
            # Usually, we pick the best move even if tabu if no non-tabu moves exist.
            # Let's re-scan for best move regardless of tabu if no non-tabu found.
            # But this is expensive.
            # Alternative: just break if no improvement possible? No, Tabu needs to move.
            # Let's just pick a random move to diversify.
            
            # Diversification: Move a random job to a random position
            i = rng.randint(n_jobs)
            k = rng.randint(n_jobs)
            if k == i:
                continue
            
            job = current_perm[i]
            base_perm = current_perm[:i] + current_perm[i+1:]
            new_perm = base_perm[:k] + [job] + base_perm[k:]
            new_ms = _makespan(pt, new_perm)
            
            current_perm = new_perm
            current_ms = new_ms
            best_move_key = (job, k)
            
        else:
            current_perm = best_move_perm
            current_ms = best_move_ms
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = current_perm[:]
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # Update tabu list
        if best_move_key:
            job, k = best_move_key
            tabu_dict[(job, k)] = iteration + tabu_tenure
            # Also tabu the reverse move?
            # Reverse of "job i to k" is "job at k to i"?
            # In insertion neighborhood, the reverse is not simply symmetric in indices.
            # We just tabu the move we made.
            
        iteration += 1
        
        # Clean up tabu dict occasionally to avoid memory leak
        if iteration % 1000 == 0:
            tabu_dict = {k: v for k, v in tabu_dict.items() if v > iteration}
            
        # If stagnation is high, increase tabu tenure or restart?
        # Let's just let it run.
        
    return best_perm, best_ms

def _simulated_annealing(pt, perm, deadline, T0=None, cooling=0.995):
    """
    Simulated Annealing with insertion neighborhood.
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    if T0 is None:
        T0 = max(10.0, current_ms * 0.5)
        
    T = T0
    rng = np.random.RandomState(42)
    
    while time.time() < deadline and T > 1e-6:
        i = rng.randint(n_jobs)
        k = rng.randint(n_jobs)
        if k == i or k == i - 1:
            continue
            
        job = current_perm[i]
        
        new_perm = current_perm[:i] + current_perm[i+1:]
        if k > i:
            k -= 1
        new_perm = new_perm[:k] + [job] + new_perm[k:]
        
        new_ms = _makespan(pt, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
                
        T *= cooling
        
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using multi-start Tabu Search.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    start_time = time.time()
    # 150s limit, leave 5s margin
    deadline = start_time + 145
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Generate initial candidates
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. Random starts
    for seed in range(3):
        candidates.append(_random_initial(n_jobs, seed))
        
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Phase 1: Tabu Search on each candidate
    # Allocate time evenly
    time_per_candidate = (deadline - 10.0) / len(candidates)
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        cand_deadline = start_time + (i + 1) * time_per_candidate
        if cand_deadline > deadline - 5.0:
            cand_deadline = deadline - 5.0
            
        perm, ms = _tabu_search(pt, cand, cand_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 2: Simulated Annealing from the best solution
    if time.time() < deadline - 2.0:
        sa_deadline = deadline - 1.0
        perm, ms = _simulated_annealing(pt, best_perm, sa_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    return best_perm

# EVOLVE-BLOCK-END
