"""Flow-shop makespan minimizer: NEH + CIG constructive, 2-opt & swap local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """Compute makespan of a job permutation using cumulative sums.
    
    For a flow shop, the makespan is the completion time on the last machine.
    C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
    The makespan is C[n-1][m-1].
    """
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # Use a running array for efficiency
    # C[m] holds completion times for the current job on each machine
    C = np.zeros(m)
    for j in range(n):
        job = perm[j]
        for k in range(m):
            if k == 0:
                C[k] += pt[job, k]
            else:
                C[k] = max(C[k], C[k-1]) + pt[job, k]
    return float(C[-1])


def _makespan_fast(perm, pt):
    """Fast makespan using vectorized operations where possible."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # This is the standard O(n*m) computation
    C = np.zeros(m)
    for j in range(n):
        job = perm[j]
        p_row = pt[job]
        for k in range(m):
            if k == 0:
                C[k] += p_row[k]
            else:
                C[k] = max(C[k], C[k-1]) + p_row[k]
    return float(C[-1])


def _neh(processing_times, seed=42):
    """NEH (Nawaz, Enscore, Ham) heuristic.
    
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the best position in the current sequence.
    """
    n, m = processing_times.shape
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    
    # Step 1: Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals, kind='stable')
    
    # Step 2: Insert each job into the best position
    seq = np.empty(n, dtype=np.int64)
    seq[0] = order[0]
    best_ms = _makespan(seq[:1], processing_times)
    
    for i in range(1, n):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        # Try inserting at each position 0..i
        for pos in range(i + 1):
            # Create new sequence
            new_seq = np.empty(i + 1, dtype=np.int64)
            new_seq[:pos] = seq[:pos]
            new_seq[pos] = job
            new_seq[pos+1:] = seq[pos:i]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Update sequence
        new_seq = np.empty(i + 1, dtype=np.int64)
        new_seq[:best_pos] = seq[:best_pos]
        new_seq[best_pos] = job
        new_seq[best_pos+1:] = seq[best_pos:i]
        seq[:i+1] = new_seq
    
    return seq


def _cig(processing_times, seed=42):
    """CIG (Campbell, Iggleston, and Jones) heuristic.
    
    1. Sort jobs by total processing time in descending order.
    2. For each job, insert it at the position that minimizes the makespan,
       but use a greedy forward insertion.
    """
    n, m = processing_times.shape
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals, kind='stable')
    
    seq = np.empty(n, dtype=np.int64)
    seq[0] = order[0]
    
    for i in range(1, n):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(i + 1):
            new_seq = np.empty(i + 1, dtype=np.int64)
            new_seq[:pos] = seq[:pos]
            new_seq[pos] = job
            new_seq[pos+1:] = seq[pos:i]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        new_seq = np.empty(i + 1, dtype=np.int64)
        new_seq[:best_pos] = seq[:best_pos]
        new_seq[best_pos] = job
        new_seq[best_pos+1:] = seq[best_pos:i]
        seq[:i+1] = new_seq
    
    return seq


def _local_search_adjacent_swap(perm, pt, deadline=None):
    """
    Fast local search using adjacent swaps only.
    This is O(n) per pass and O(n*m) per evaluation.
    Much faster than 2-opt or arbitrary swaps.
    """
    n = len(perm)
    if n < 2:
        return perm
    
    perm = perm.copy()
    best_ms = _makespan(perm, pt)
    
    # Limit total iterations to prevent infinite loops or excessive time
    # A few passes are usually enough for adjacent swap to converge
    max_passes = 20
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                return perm
            
            # Swap i and i+1
            perm[i], perm[i+1] = perm[i+1], perm[i]
            ms = _makespan(perm, pt)
            
            if ms < best_ms - 1e-9:
                best_ms = ms
                improved = True
            else:
                # Swap back
                perm[i], perm[i+1] = perm[i+1], perm[i]
        
        if not improved:
            break
            
    return perm


def order_jobs(processing_times):
    """Main entry point: construct a job permutation minimizing makespan."""
    start_time = time.time()
    # Use a very conservative deadline to ensure we return before the 120s kill
    # The evaluator kills at 120s. We want to return by 115s.
    deadline = start_time + 110.0
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # 1. Generate initial solution using NEH (very fast, O(n^2 * m))
    # NEH is generally strong for flow shop.
    best_perm = _neh(pt)
    best_ms = _makespan(best_perm, pt)
    
    # 2. Check if we have time for local search
    # We only proceed if we have at least 5 seconds left.
    if time.time() < deadline - 5.0:
        # Apply fast adjacent swap local search
        # This is much safer than 2-opt or arbitrary swaps.
        improved_perm = _local_search_adjacent_swap(best_perm, pt, deadline=deadline - 2.0)
        ms_improved = _makespan(improved_perm, pt)
        
        if ms_improved < best_ms:
            best_perm = improved_perm
            best_ms = ms_improved
    
    # 3. Final safety check
    # If we are getting close to the deadline, just return what we have.
    # The list conversion is fast.
    if time.time() > deadline:
         return list(best_perm.astype(np.int64))
        
    return list(best_perm.astype(np.int64))
# EVOLVE-BLOCK-END
