"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, P):
    """Compute makespan for a given permutation."""
    n, m = P.shape
    if n == 0:
        return 0.0
    # cum[j] = completion time on machine m-1 for job j in the sequence
    # We simulate the flow shop
    cum = np.zeros(n, dtype=np.float64)
    # For each machine, track completion times
    # Standard flow shop makespan computation
    # C[j][k] = max(C[j-1][k], C[j][k-1]) + P[j][k]
    # We only need the final makespan, which is C[n-1][m-1]
    
    # More efficient: track completion on last machine
    # C_last[j] = completion of job j on last machine
    # C_prev[j] = completion of job j on second-to-last machine (or just prev machine)
    
    # Let's use the standard O(n*m) approach but optimized
    # c[j] = completion of job j on the current machine being processed
    # We iterate machines, for each machine we compute completion times
    
    # Actually, the standard way:
    # C[0][0] = P[0][0]
    # C[j][0] = C[j-1][0] + P[j][0]  (first machine, sequential)
    # C[0][k] = C[0][k-1] + P[0][k]  (first job, across machines)
    # C[j][k] = max(C[j-1][k], C[0][k-1]... no
    # C[j][k] = max(C[j-1][k], C[j][k-1]) + P[j][k]
    
    # To save memory, we only need the previous row and current row
    # But we can be even more clever: we only need the last element of each row
    # Actually no, we need C[j][k-1] which is in the same row.
    
    # Let's just do O(n*m) with two arrays
    if m == 1:
        return P[perm].sum()
    
    # prev[j] = C[j][k-1] for the previous machine
    # curr[j] = C[j][k] for the current machine
    # For machine 0: C[j][0] = C[j-1][0] + P[j][0]
    # So prev[j] = sum of P[perm[0..j]][0]
    
    # Let's re-index: let job index in perm be i, so actual job is perm[i]
    # C[i][k] = max(C[i-1][k], C[i][k-1]) + P[perm[i]][k]
    
    # We'll track an array where entry i is C[i][k] for current k
    # For k=0: C[i][0] = C[i-1][0] + P[perm[i]][0]
    # So we can compute row by row
    
    c_prev = np.zeros(n, dtype=np.float64)  # C[i][k-1]
    # For k=0, C[i][0] is cumulative sum of P[perm][0]
    # Let's handle k=0 separately
    P_perm = P[perm]  # (n, m)
    
    # c[i] will hold C[i][k] as we iterate k
    c = np.zeros(n, dtype=np.float64)
    # k=0
    c[0] = P_perm[0, 0]
    for i in range(1, n):
        c[i] = c[i-1] + P_perm[i, 0]
    
    # For k >= 1
    for k in range(1, m):
        # C[i][k] = max(C[i-1][k], C[i][k-1]) + P[i][k]
        # C[i][k-1] is stored in c_prev[i]
        # C[i-1][k] is the new value we're computing
        new_c = np.zeros(n, dtype=np.float64)
        new_c[0] = c[0] + P_perm[0, k]  # C[0][k] = C[0][k-1] + P[0][k] since C[-1][k] doesn't exist
        for i in range(1, n):
            new_c[i] = max(new_c[i-1], c[i]) + P_perm[i, k]
        c_prev = c
        c = new_c
    
    return c[n-1]


def _makespan_fast(perm, P):
    """Faster makespan using vectorized numpy where possible."""
    n, m = P.shape
    if n == 0:
        return 0.0
    if m == 1:
        return P[perm].sum()
    
    P_perm = P[perm]
    
    # c[i] = C[i][k] for current machine k
    c = np.cumsum(P_perm[:, 0])
    
    for k in range(1, m):
        # C[i][k] = max(C[i-1][k], C[i][k-1]) + P[i][k]
        # This is a recurrence that's hard to fully vectorize due to the dependency on C[i-1][k]
        # But we can use a trick: 
        # C[i][k] - C[i][k-1] = max(C[i-1][k] - C[i][k-1], 0) + P[i][k]
        # Let d[i] = C[i][k] - C[i][k-1]
        # d[0] = P[0][k]
        # d[i] = max(d[i-1] + C[i-1][k-1] - C[i][k-1], 0) + P[i][k]
        # This is still sequential.
        
        # Just do it with a loop, but use local variables for speed
        new_c = np.empty(n, dtype=np.float64)
        new_c[0] = c[0] + P_perm[0, k]
        prev_val = new_c[0]
        for i in range(1, n):
            val = max(prev_val, c[i]) + P_perm[i, k]
            new_c[i] = val
            prev_val = val
        c = new_c
    
    return c[n-1]


def order_jobs(processing_times):
    P = np.asarray(processing_times, dtype=np.float64)
    n, m = P.shape
    
    if n <= 1:
        return list(range(n))
    
    deadline = time.time() + 140.0  # 10s safety margin
    
    # NEH heuristic as initial solution
    totals = P.sum(axis=1)
    neheh_order = np.argsort(-totals)
    
    # Build the NEH sequence
    seq = []
    for j in neheh_order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(seq) + 1):
            candidate = list(seq[:pos]) + [j] + list(seq[pos:])
            ms = _makespan_fast(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, j)
    
    best_perm = np.array(seq, dtype=np.int64)
    best_ms = _makespan_fast(best_perm, P)
    
    # Local search: 2-opt (swap adjacent or any two positions)
    # Use 2-opt: swap any two positions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Swap i and j
                new_perm = best_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                ms = _makespan_fast(new_perm, P)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
    
    # Insertion-based local search
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(n):
                    if i == j or time.time() > deadline:
                        continue
                    # Remove i, insert at j
                    new_perm = list(best_perm)
                    job = new_perm.pop(i)
                    # Adjust index if j > i
                    if j > i:
                        j -= 1
                    new_perm.insert(j, job)
                    ms = _makespan_fast(new_perm, P)
                    if ms < best_ms:
                        best_ms = ms
                        best_perm = np.array(new_perm, dtype=np.int64)
                        improved = True
    
    return list(best_perm)
# EVOLVE-BLOCK-END
