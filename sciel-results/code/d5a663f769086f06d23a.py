# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_fast(perm, pt):
    """Optimized makespan computation using pure Python loops for small n."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    # Use list-based approach for better cache performance
    prev = [0.0] * n
    # Machine 0
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        prev[i] = s
    
    for k in range(1, m):
        curr = [0.0] * n
        curr[0] = prev[0] + pt[perm[0], k]
        for i in range(1, n):
            v1 = prev[i]
            v2 = curr[i-1]
            curr[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev = curr
    
    return prev[-1]

def _local_search_insertion(perm, pt, max_iters=100, deadline=None):
    """
    Perform local search using insertion moves.
    At each step, try moving each job to every other position and keep the best improvement.
    """
    n = len(perm)
    if n <= 1:
        return perm
    
    current_perm = perm.copy()
    current_ms = _makespan_fast(current_perm, pt)
    best_perm = current_perm.copy()
    best_ms = current_ms
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        best_move_i = -1
        best_move_j = -1
        best_candidate_ms = current_ms
        
        # Try all insertion moves
        for i in range(n):
            job = current_perm[i]
            # Remove job from position i
            # We simulate the move by creating a new list? That's O(n).
            # Then evaluate makespan O(n*m).
            # Total O(n^2 * m) per iteration.
            
            # Optimization: We can compute the makespan of the modified permutation
            # without creating the full list each time if we are clever, 
            # but for simplicity and correctness, we create the list.
            
            # To speed up, we can skip if i == j (no move)
            for j in range(n):
                if i == j:
                    continue
                
                # Create new permutation
                # If j > i, removing i shifts j to j-1
                # If j < i, removing i does not affect j's index
                # Let's just construct the list. It's O(n).
                new_perm = list(current_perm)
                new_perm.pop(i)
                # Insert at j
                # Note: if j > i, the target index in the new list is j-1
                # if j < i, the target index is j
                # Wait, standard insertion: 
                # Remove at i. The list size is n-1.
                # We want to insert the job such that its new index is j.
                # If j > i, in the original list, position j was after i.
                # After removing i, the element at j moves to j-1.
                # So if we want the job to end up at index j in the NEW list, 
                # we insert at j.
                # Let's verify:
                # Perm: [A, B, C, D] (0,1,2,3)
                # Move A (0) to pos 2 (C's pos).
                # Remove A: [B, C, D]
                # Insert A at 2: [B, C, A, D]
                # A is now at index 2. Correct.
                
                new_perm.insert(j, job)
                
                ms = _makespan_fast(np.array(new_perm, dtype=np.int64), pt)
                
                if ms < best_candidate_ms - 1e-9:
                    best_candidate_ms = ms
                    best_move_i = i
                    best_move_j = j
        
        if best_move_i != -1:
            # Apply the best move
            current_perm = list(current_perm)
            job = current_perm.pop(best_move_i)
            current_perm.insert(best_move_j, job)
            current_ms = best_candidate_ms
            improved = True
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm.copy()
        else:
            # No improvement found
            break
            
    return best_perm

def _neh_heuristic(pt):
    """NEH heuristic for flow shop. Returns permutation."""
    n, m = pt.shape
    
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build permutation greedily
    perm = []
    
    for job in order:
        if not perm:
            perm = [job]
            continue
        
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(np.array(new_perm, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return np.array(perm, dtype=np.int64)

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    # Safety margin: 110s out of 120s limit
    deadline = start_time + 110.0
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # 1. Initial Heuristic: NEH
    best_perm = _neh_heuristic(pt)
    best_ms = _makespan_fast(best_perm, pt)
    
    # 2. Local Search from NEH
    if time.time() < deadline:
        ls_perm = _local_search_insertion(best_perm, pt, max_iters=100, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # 3. Random Restarts
    # If we have time left, try random starts + local search
    # Estimate time per restart: NEH is O(n^2 m), LS is O(iter * n^2 m).
    # Let's try a few restarts.
    rng = np.random.default_rng(42)
    
    # Determine how many restarts we can afford
    # Assume each restart takes ~0.1s for small n, ~1s for large n.
    # We'll just loop until deadline.
    
    restart_count = 0
    while time.time() < deadline - 5.0: # Keep 5s buffer for final checks
        restart_count += 1
        random_perm = rng.permutation(n).astype(np.int64)
        
        # Run local search from random start
        # Limit iterations for random starts to save time
        max_ls_iters = 50
        ls_r_perm = _local_search_insertion(random_perm, pt, max_iters=max_ls_iters, deadline=deadline)
        ls_r_ms = _makespan_fast(ls_r_perm, pt)
        
        if ls_r_ms < best_ms:
            best_ms = ls_r_ms
            best_perm = ls_r_perm
            
        # If n is very small, we can do more restarts. If large, fewer.
        if n > 50:
             # Large problems are expensive, limit restarts
             if restart_count >= 10:
                 break
        else:
             # Small problems are cheap, allow more
             if restart_count >= 50:
                 break

    return list(best_perm)
# EVOLVE-BLOCK-END
