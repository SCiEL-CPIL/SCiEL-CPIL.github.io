"""Flow-shop scheduling with NEH + Tabu Search (swap + insert moves)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a permutation (list or ndarray of job indices)."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    if m == 1:
        return float(pt[list(perm), 0].sum())
    
    # C[i, k] = completion time of job perm[i] on machine k
    # C[i, 0] = C[i-1, 0] + pt[perm[i], 0]
    # C[i, k] = max(C[i-1, k], C[i, k-1]) + pt[perm[i], k]
    
    # We only need the last row to get the makespan, but we need previous rows
    # Actually we can compute row by row and keep only the previous row.
    prev_row = np.zeros(m)
    curr_row = np.zeros(m)
    
    for i in range(n):
        job = perm[i]
        curr_row[0] = prev_row[0] + pt[job, 0]
        for k in range(1, m):
            curr_row[k] = max(prev_row[k], curr_row[k-1]) + pt[job, k]
        # swap
        prev_row, curr_row = curr_row, prev_row
    
    return float(prev_row[-1])


def _neh_initialization(pt):
    """
    NEH heuristic: sort jobs by total processing time descending,
    then insert each job into the position that minimizes makespan.
    """
    n, m = pt.shape
    if n <= 1:
        return list(range(n))
    
    # Sort by total processing time descending
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_mk = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            mk = _makespan(new_perm, pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _delta_makespan_swap(perm, pt, i, j):
    """
    Compute the new makespan after swapping positions i and j in perm.
    Uses full recomputation but only for the affected tail.
    For simplicity and correctness, we recompute from min(i,j) onward.
    Returns the new makespan.
    """
    n = len(perm)
    m = pt.shape[1]
    if n <= 2:
        return _makespan(perm, pt)
    
    # Create the swapped permutation
    new_perm = list(perm)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    
    # We can compute makespan from scratch; for small n this is fast.
    # For larger n, we could optimize, but let's keep it simple first.
    return _makespan(new_perm, pt)


def _try_swap(perm, pt, i, j):
    """Try swapping positions i and j. Return (new_perm, new_mk) if improved, else None."""
    n = len(perm)
    if i == j or i >= n or j >= n:
        return None
    new_perm = list(perm)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    new_mk = _makespan(new_perm, pt)
    return (new_perm, new_mk)


def _try_insert(perm, pt, i, j):
    """
    Try moving job from position i to position j.
    Return (new_perm, new_mk) or None.
    """
    n = len(perm)
    if i == j or i >= n or j >= n:
        return None
    job = perm[i]
    new_perm = list(perm)
    del new_perm[i]
    # Adjust j if removing from before j
    if i < j:
        j -= 1
    new_perm.insert(j, job)
    new_mk = _makespan(new_perm, pt)
    return (new_perm, new_mk)


def _tabu_search(perm0, pt, max_iters, time_limit, seed=None):
    """
    Tabu search with swap and insert moves, aspiration criteria.
    Returns best permutation found.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    n, m = pt.shape
    if n <= 1:
        return list(perm0)
    
    perm = list(perm0)
    best_perm = list(perm)
    best_mk = _makespan(perm, pt)
    curr_mk = best_mk
    
    # Tabu list: store (position, job) for insert moves, or (i,j) for swaps
    # We'll use a simple approach: tabu list of (i, j) pairs for swaps
    # and (i, j) for inserts (move from i to j)
    tabu_len = max(10, n // 2)
    tabu_list = []  # list of (move_type, i, j, iteration)
    
    start_time = time.time()
    
    for iteration in range(max_iters):
        # Time check
        if time.time() - start_time > time_limit:
            break
        
        # Age out tabu entries
        tabu_list = [t for t in tabu_list if iteration - t[3] < tabu_len]
        
        best_move = None
        best_move_mk = curr_mk  # We want to find the best non-tabu move, or tabu with aspiration
        
        # Generate candidate moves
        # Swaps: O(n^2)
        # Inserts: O(n^2)
        # For efficiency, we'll sample a subset if n is large
        
        n_moves_to_consider = min(n * n, 5000)  # Cap moves per iteration
        
        # Shuffle all possible moves and take a subset
        if n <= 30:
            # All moves
            swap_candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
            insert_candidates = [(i, j) for i in range(n) for j in range(n) if i != j]
        else:
            # Sample
            swap_candidates = [(rng.randint(0, n-1), rng.randint(0, n-1)) for _ in range(n_moves_to_consider // 2)]
            swap_candidates = [(a, b) for a, b in swap_candidates if a != b]
            insert_candidates = [(rng.randint(0, n-1), rng.randint(0, n-1)) for _ in range(n_moves_to_consider // 2)]
            insert_candidates = [(a, b) for a, b in insert_candidates if a != b]
        
        # Evaluate swaps
        for (i, j) in swap_candidates:
            if i == j:
                continue
            # Check tabu
            is_tabu = any(t[0] == 'swap' and t[1] == min(i,j) and t[2] == max(i,j) for t in tabu_list)
            
            result = _try_swap(perm, pt, i, j)
            if result is None:
                continue
            new_perm, new_mk = result
            
            # Aspiration: accept if it improves best
            if new_mk < best_mk:
                is_tabu = False
            
            if not is_tabu and new_mk < best_move_mk:
                best_move_mk = new_mk
                best_move = ('swap', i, j, new_perm, new_mk)
        
        # Evaluate inserts
        for (i, j) in insert_candidates:
            # Check tabu
            is_tabu = any(t[0] == 'insert' and t[1] == i and t[2] == j for t in tabu_list)
            
            result = _try_insert(perm, pt, i, j)
            if result is None:
                continue
            new_perm, new_mk = result
            
            # Aspiration
            if new_mk < best_mk:
                is_tabu = False
            
            if not is_tabu and new_mk < best_move_mk:
                best_move_mk = new_mk
                best_move = ('insert', i, j, new_perm, new_mk)
        
        # If no improving move found, accept the best non-tabu move even if it worsens
        if best_move is None:
            # Fallback: accept best move regardless of tabu (to escape)
            # Or just break
            # Let's try one more time with a random move
            i = rng.randint(0, n-1)
            j = rng.randint(0, n-1)
            if i != j:
                result = _try_swap(perm, pt, i, j)
                if result:
                    best_move = ('swap', i, j, result[0], result[1])
            if best_move is None:
                break
        
        # Apply the best move
        move_type, mi, mj, new_perm, new_mk = best_move
        
        # Update tabu list
        if move_type == 'swap':
            tabu_list.append(('swap', min(mi, mj), max(mi, mj), iteration))
        else:
            tabu_list.append(('insert', mi, mj, iteration))
        
        perm = new_perm
        curr_mk = new_mk
        
        # Update best
        if curr_mk < best_mk:
            best_mk = curr_mk
            best_perm = list(perm)
    
    return best_perm


def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    # Phase 1: NEH initialization
    perm = _neh_initialization(processing_times)
    curr_mk = _makespan(perm, processing_times)
    best_perm = list(perm)
    best_mk = curr_mk
    
    # Phase 2: Multiple tabu search restarts
    # First, do a full tabu search on the NEH solution
    remaining_time = deadline - time.time()
    
    # Determine how many restarts we can afford
    # Estimate time for one tabu search: roughly n * n * m operations
    # Let's do at least 1 full search, then use remaining time for more
    
    max_iters_one = max(100, n * 20)
    
    # First tabu search
    result = _tabu_search(perm, processing_times, max_iters_one, remaining_time * 0.5, seed=42)
    result_mk = _makespan(result, processing_times)
    if result_mk < best_mk:
        best_mk = result_mk
        best_perm = result
    
    # If time remains, do more restarts with different seeds/perturbations
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time.time()
    
    if time_remaining > 5.0:
        # Do additional restarts with perturbed solutions
        n_restarts = min(5, int(time_remaining / 15.0))  # ~15s per restart
        n_restarts = max(1, n_restarts)
        
        for r in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the best solution
            perturbed = list(best_perm)
            n_perturb = max(1, n // 5)
            for _ in range(n_perturb):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            # Tabu search on perturbed solution
            time_limit_r = (deadline - time.time()) * 0.7
            max_iters_r = max(50, n * 10)
            
            result = _tabu_search(perturbed, processing_times, max_iters_r, time_limit_r, seed=100 + r)
            result_mk = _makespan(result, processing_times)
            if result_mk < best_mk:
                best_mk = result_mk
                best_perm = result
    
    return best_perm
# EVOLVE-BLOCK-END
