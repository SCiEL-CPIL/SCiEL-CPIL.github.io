"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    if len(perm) == 0:
        return 0.0
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    # C[j, k] = max(C[j-1, k], C[j, k-1]) + p[j, k]
    # We maintain C_prev (completion times for previous job across machines)
    # and compute C_curr for current job.
    
    C_prev = np.zeros(m)
    for j in perm:
        C_curr = np.zeros(m)
        for k in range(m):
            if k == 0:
                C_curr[0] = C_prev[0] + pt[j, 0]
            else:
                C_curr[k] = max(C_prev[k], C_curr[k-1]) + pt[j, k]
        C_prev = C_curr
        
    return C_prev[-1]

def _neh_heuristic(pt):
    """
    NEH (Nawaz, Enscore, Ham) heuristic.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the position that minimizes makespan.
    """
    n, m = pt.shape
    if n == 0:
        return []
    
    # Calculate total processing time for each job
    totals = pt.sum(axis=1)
    # Sort indices by total processing time descending
    order = np.argsort(-totals)
    
    current_perm = []
    for job_idx in order:
        job = job_idx
        best_pos = 0
        best_mk = float('inf')
        
        # Try inserting job at each position in current_perm
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _makespan(candidate_perm, pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def _delta_insertion(pt, perm, job, pos):
    """
    Calculate the makespan if job is inserted at position pos in perm.
    This is O(n*m). For efficiency in local search, we might want a delta,
    but for simplicity and robustness in the time limit, full recalculation
    of the tail or full makespan is safer if n is small.
    However, insertion changes the sequence, so we need to recalculate from pos onwards.
    Actually, _makespan is O(n*m). If we call it n times for each of n jobs, it's O(n^2*m).
    For n=100, m=10, that's 100,000 * 10 = 1e6 ops, which is fast in Python if vectorized or tight loops.
    Let's stick to _makespan for correctness.
    """
    candidate = perm[:pos] + [job] + perm[pos:]
    return _makespan(candidate, pt)

def _local_search_insertion(perm, pt):
    """
    Perform a local search using insertion moves.
    For each job, try moving it to every other position.
    Accept the best move if it improves the makespan.
    Repeat until no improvement is found.
    """
    n = len(perm)
    current_mk = _makespan(perm, pt)
    improved = True
    
    while improved:
        improved = False
        best_move = None
        best_mk = current_mk
        
        for i in range(n):
            job = perm[i]
            # Try inserting job at every position j != i
            # We can skip i and i+1 as they result in the same permutation or minor shifts
            for j in range(n + 1):
                if j == i or j == i + 1:
                    continue
                
                # Create candidate permutation
                # Remove job from i
                candidate = perm[:i] + perm[i+1:]
                # Insert job at j
                # Note: if j > i, the index j in the original perm corresponds to j-1 in the shortened perm
                # But it's easier to just build the list
                candidate = perm[:i] + perm[i+1:]
                insert_pos = j
                if j > i:
                    insert_pos = j - 1
                candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
                
                cand_mk = _makespan(candidate, pt)
                
                if cand_mk < best_mk:
                    best_mk = cand_mk
                    best_move = candidate
        
        if best_move is not None:
            perm = best_move
            current_mk = best_mk
            improved = True
            
    return perm, current_mk

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # 1. Initial solution via NEH
    current_perm = _neh_heuristic(processing_times)
    current_mk = _makespan(current_perm, processing_times)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # 2. Local Search: Insertion
    # This is O(n^3 * m) in the worst case if we do a full pass, but usually converges fast.
    current_perm, current_mk = _local_search_insertion(current_perm, processing_times)
    if current_mk < best_mk:
        best_perm = current_perm[:]
        best_mk = current_mk
        
    # 3. Simulated Annealing with Insertion Moves
    # We use insertion moves (remove job from i, insert at j) which is a larger neighborhood than swap.
    
    # Estimate temperature
    # Typical delta in makespan for insertion can be significant.
    # Let's scale T based on the current makespan.
    T = best_mk * 0.1
    cooling = 0.999
    min_T = 1e-5
    
    # To allow more exploration, we can restart SA a few times if time permits.
    # But first, let's run one long SA.
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate neighbor: Insertion move
        i = random.randint(0, n - 1)
        j = random.randint(0, n)
        if j == i or j == i + 1:
            # Skip trivial moves
            T *= cooling
            continue
            
        job = current_perm[i]
        # Remove job
        candidate = current_perm[:i] + current_perm[i+1:]
        # Insert job at j
        insert_pos = j
        if j > i:
            insert_pos = j - 1
        candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
        
        cand_mk = _makespan(candidate, processing_times)
        delta = cand_mk - current_mk
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_perm = candidate
            current_mk = cand_mk
            
            if current_mk < best_mk:
                best_mk = current_mk
                best_perm = current_perm[:]
                
        T *= cooling
        
    return list(best_perm)
# EVOLVE-BLOCK-END
