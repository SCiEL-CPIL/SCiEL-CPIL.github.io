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

def _completion_matrix(perm, pt):
    """Compute the completion time matrix C[j][k] = completion time of job j on machine k.
    Returns a 2D list of lists.
    """
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return []
    
    C = [[0.0] * m for _ in range(n)]
    
    # Machine 0
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        C[i][0] = s
    
    for k in range(1, m):
        prev_c = C[0][k-1]
        for i in range(1, n):
            # C[i][k] = max(C[i-1][k], C[i][k-1]) + pt[perm[i], k]
            v1 = C[i-1][k]
            v2 = C[i][k-1]
            C[i][k] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
            
    return C

def _eval_reversal(perm, C, i, j, pt):
    """Evaluate the makespan change if we reverse perm[i:j+1].
    Uses the existing completion matrix C to compute the new makespan efficiently.
    Returns the new makespan.
    """
    n = len(perm)
    m = pt.shape[1]
    
    # If reversal is small or n is small, just recalculate?
    # For now, let's try to update the affected rows.
    # The reversal affects jobs at indices i..j in the permutation.
    # Let's create a new permutation segment.
    
    # To avoid complex partial updates which are error-prone, and since n is often small,
    # we can use a hybrid approach: if the segment is large, recalculate.
    # However, the "incremental" idea is to only update rows i..j and potentially subsequent rows.
    # Actually, reversing i..j changes the sequence of jobs.
    # The completion times for jobs 0..i-1 are unchanged.
    # The completion times for jobs i..j are changed.
    # The completion times for jobs j+1..n-1 are changed because their predecessors changed.
    
    # So we effectively need to recalculate C[i:] for all machines.
    # This is O((n-i) * m).
    # If i is close to n, this is fast. If i is 0, it's O(n*m).
    
    # Let's implement a function that takes the current C and the new perm,
    # and updates C from row i onwards.
    
    new_perm = perm.copy()
    new_perm[i:j+1] = perm[i:j+1][::-1]
    
    # We need to update C[i:]
    # The values C[0..i-1] are the same.
    # We need to recalculate C[i..n-1] for all machines k.
    
    # We can do this in-place on a copy of C, or just return the new makespan.
    # Let's create a new C for the suffix.
    
    # Optimization: If i == 0, it's a full recalculation.
    # If i > 0, we can use the boundary conditions from C[i-1] and C[i][k-1] (old) ?
    # No, C[i][k-1] is also changed if k>0.
    
    # Actually, for flow shop, C[i][k] depends on C[i-1][k] and C[i][k-1].
    # So to update C[i][k], we need C[i-1][k] (unchanged if k>=0? No, C[i-1][k] is unchanged)
    # and C[i][k-1] (which is changed).
    
    # So we must recalculate row by row from i to n-1, and for each row, machine by machine from 0 to m-1.
    # Wait, C[i][0] depends only on sum of processing times of jobs 0..i.
    # So C[i][0] changes.
    
    # Algorithm:
    # 1. Copy C to new_C (or just overwrite suffix)
    # 2. For row r from i to n-1:
    #    For col c from 0 to m-1:
    #       if c == 0:
    #          new_C[r][0] = new_C[r-1][0] + pt[new_perm[r], 0]  (if r>0)
    #          else new_C[r][0] = pt[new_perm[r], 0]
    #       else:
    #          new_C[r][c] = max(new_C[r-1][c], new_C[r][c-1]) + pt[new_perm[r], c]
    
    # This is O((n-i)*m).
    
    # Let's implement this.
    
    # We need to be careful with indices.
    # C is a list of lists.
    
    # We will modify a copy of C to avoid side effects, or just return the makespan.
    # Since we only need the makespan (C[n-1][m-1]), we can just compute that.
    
    # Let's create a temporary array for the suffix.
    # To save memory/time, we can just use the existing C for the prefix (0..i-1)
    # and compute the suffix.
    
    # Let's create a 2D array for the suffix rows i..n-1.
    # Actually, it's easier to just copy the whole C if n is small, or use a rolling buffer.
    # Given the constraints, let's just do the calculation.
    
    # We need the last column of the row above (i-1) and the current row's previous column.
    
    # Let's just create a new C matrix for the suffix.
    # But we need C[i-1][k] for the first row of the suffix.
    
    # Let's use a simple loop.
    
    # Initialize a 2D list for the suffix C_suffix
    # Dimensions: (n-i) x m
    # But we also need the "previous row" for the first row of the suffix, which is C[i-1].
    
    # Let's just recalculate the whole C matrix if n is small, or use this incremental if n is large.
    # For simplicity and robustness, let's implement the incremental update.
    
    # We'll create a new list of lists for the updated C.
    # Actually, we can just update C in-place if we are careful, but we need the old values for the calculation?
    # No, C[i][k] depends on C[i-1][k] (old, unchanged) and C[i][k-1] (new, calculated).
    # So we can update C in-place row by row, from left to right?
    # C[i][0] = C[i-1][0] + pt[new_perm[i], 0]
    # C[i][1] = max(C[i-1][1], C[i][0]) + pt[new_perm[i], 1]
    # Here C[i][0] is new. C[i-1][1] is old (unchanged).
    # So yes, we can update C in-place for rows i..n-1.
    
    # Let's do it.
    
    # Note: We must not modify the original C if we want to evaluate multiple moves from the same state.
    # So we should copy C first? Or pass a copy?
    # Copying C is O(n*m). The update is O((n-i)*m).
    # If i is large, copy is expensive.
    # Alternative: Just calculate the makespan without storing the full matrix?
    # We can use a 1D array for the current row and keep the previous row.
    
    # Let's use a 1D array approach for the suffix calculation.
    
    # prev_row = C[i-1] (if i>0) else None
    # curr_row = [0.0] * m
    
    # But we need to propagate the changes to subsequent rows.
    # So we need to store the calculated rows to be used as "prev_row" for the next row.
    
    # Let's just create a new C_suffix array.
    
    # Actually, for the purpose of this task, if n is small (< 50), full recalculation is fast enough.
    # If n is large, we need incremental.
    # Let's assume n can be up to 100.
    
    # Let's implement the in-place update on a copy of C.
    
    C_new = [row[:] for row in C] # Copy C
    
    # Update rows i to n-1
    for r in range(i, n):
        # Machine 0
        if r == 0:
            C_new[r][0] = pt[new_perm[r], 0]
        else:
            C_new[r][0] = C_new[r-1][0] + pt[new_perm[r], 0]
        
        for c in range(1, m):
            v1 = C_new[r-1][c]
            v2 = C_new[r][c-1]
            C_new[r][c] = (v1 if v1 > v2 else v2) + pt[new_perm[r], c]
            
    return C_new[n-1][m-1], C_new, new_perm

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

def _tabu_search(perm, pt, max_iters=1000, tabu_tenure=10, deadline=None):
    """Tabu search with 2-opt moves (reversals)."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    best_C = _completion_matrix(best_perm, pt)
    
    current_perm = best_perm.copy()
    current_ms = best_ms
    current_C = best_C
    
    # Tabu list: store (i, j) of last move
    tabu_list = []
    
    iters = 0
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
            
        iters += 1
        
        best_move_i = -1
        best_move_j = -1
        best_candidate_ms = float('inf')
        best_candidate_perm = None
        best_candidate_C = None
        
        # Evaluate all 2-opt moves
        # To speed up, we can limit the length of the reversal?
        # Or just try all i, j.
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Check tabu
                if (i, j) in tabu_list:
                    continue
                
                # Evaluate reversal
                cand_ms, cand_C, cand_perm = _eval_reversal(current_perm, current_C, i, j, pt)
                
                # Aspiration criterion: accept if better than best found
                if cand_ms < best_candidate_ms:
                    best_candidate_ms = cand_ms
                    best_candidate_perm = cand_perm
                    best_candidate_C = cand_C
                    best_move_i = i
                    best_move_j = j
        
        if best_candidate_perm is None:
            # No non-tabu moves found, clear tabu list and restart? Or break?
            # Let's just break or accept the best tabu move?
            # For simplicity, break.
            break
            
        # Update current
        current_perm = best_candidate_perm
        current_ms = best_candidate_ms
        current_C = best_candidate_C
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = current_perm.copy()
            best_C = current_C
            
        # Update tabu list
        tabu_list.append((best_move_i, best_move_j))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
            
    return best_perm

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Start with NEH heuristic
    best_perm = _neh_heuristic(pt)
    best_ms = _makespan_fast(best_perm, pt)
    
    # Tabu search
    if time.time() < deadline:
        ts_perm = _tabu_search(best_perm, pt, max_iters=500, deadline=deadline)
        ts_ms = _makespan_fast(ts_perm, pt)
        if ts_ms < best_ms:
            best_perm = ts_perm
            best_ms = ts_ms
    
    # Try NEH with average processing time
    if time.time() < deadline:
        avgs = pt.mean(axis=1)
        order2 = np.argsort(-avgs)
        perm2 = []
        for job in order2:
            if not perm2:
                perm2 = [job]
                continue
            best_pos = 0
            best_ms2 = float('inf')
            for pos in range(len(perm2) + 1):
                new_perm = perm2[:pos] + [job] + perm2[pos:]
                ms = _makespan_fast(np.array(new_perm, dtype=np.int64), pt)
                if ms < best_ms2:
                    best_ms2 = ms
                    best_pos = pos
            perm2.insert(best_pos, job)
        perm2_arr = np.array(perm2, dtype=np.int64)
        ms2 = _makespan_fast(perm2_arr, pt)
        if ms2 < best_ms:
            best_perm = perm2_arr
            best_ms = ms2
            if time.time() < deadline:
                ts_perm2 = _tabu_search(best_perm, pt, max_iters=300, deadline=deadline)
                ts_ms2 = _makespan_fast(ts_perm2, pt)
                if ts_ms2 < best_ms:
                    best_perm = ts_perm2
                    best_ms = ts_ms2
    
    # Random restarts with Tabu Search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            # Quick Tabu search from random start
            ts_r = _tabu_search(random_perm, pt, max_iters=100, deadline=deadline)
            ts_ms_r = _makespan_fast(ts_r, pt)
            if ts_ms_r < best_ms:
                best_perm = ts_r
                best_ms = ts_ms_r
    
    return list(best_perm)
# EVOLVE-BLOCK-END
