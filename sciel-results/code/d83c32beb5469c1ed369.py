import numpy as np
import time

def solve(a, b):
    """
    Solve the Quadratic Assignment Problem (QAP) for the given matrices a and b.
    Returns a permutation p of 0..n-1 that minimizes sum_{i,j} a[i,j] * b[p[i], p[j]].
    """
    n = len(a)
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5s safety margin

    def compute_cost(p):
        b_sub = b[np.ix_(p, p)]
        return int(np.sum(a * b_sub))

    def get_deltas_for_i(p, i):
        """
        Compute the delta for swapping p[i] with p[j] for all j > i.
        Returns an array of deltas of length n-i-1, where delta[k] corresponds to swap(i, i+1+k).
        """
        li = p[i]
        
        # Precompute differences involving i
        # A_diff[k] = a[i, k] - a[j, k]
        # We need this for all j > i.
        # a[i, :] is fixed. a[j, :] varies.
        # Let's compute S1 = sum_k (a[i, k] - a[j, k]) * (b[lj, p[k]] - b[li, p[k]])
        # = sum_k a[i, k] * (b[lj, p[k]] - b[li, p[k]]) - sum_k a[j, k] * (b[lj, p[k]] - b[li, p[k]])
        
        # Let V_b_row_diff[k] = b[lj, p[k]] - b[li, p[k]]
        # This depends on lj.
        
        # This direct vectorization for ALL j is tricky because lj changes.
        # However, we can compute the delta for a specific j quickly.
        # The previous plan was to loop i, and for each i, loop j.
        # The speedup comes from vectorizing the inner sum over k.
        
        # Let's stick to the O(n) vectorized delta for a single (i, j) pair, 
        # but optimize the Python loop overhead by using numpy arrays where possible.
        
        # Actually, the previous analysis suggested that vectorizing over j for fixed i is hard 
        # because b[lj, p] depends on j.
        # So we are back to O(n^2) delta computations, but each is O(1) or O(n) vectorized.
        # The bottleneck in the original code was likely the Python overhead of 
        # `swap_cost_delta` being called 11,000 times per pass.
        
        # Let's implement a faster delta function that uses precomputed arrays if possible, 
        # or just highly optimized numpy ops.
        
        pass

    def get_delta(p, i, j):
        """
        Compute delta for swapping p[i] and p[j].
        Optimized for speed.
        """
        li, lj = p[i], p[j]
        
        # Vectorized delta calculation
        # delta = sum_k (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        #       + sum_k (a[k,i] - a[k,j]) * (b[p[k], lj] - b[p[k], li])
        
        # Precompute differences
        a_i_j = a[i, j]
        a_j_i = a[j, i]
        a_i_i = a[i, i]
        a_j_j = a[j, j]
        
        b_lj_li = b[lj, li]
        b_li_lj = b[li, lj]
        b_lj_lj = b[lj, lj]
        b_li_li = b[li, li]
        
        # Explicit terms
        d_explicit = (a_i_j * (b_lj_li - b_li_lj) + 
                      a_j_i * (b_li_lj - b_lj_li) + 
                      a_i_i * (b_lj_lj - b_li_li) + 
                      a_j_j * (b_li_li - b_lj_lj))
        
        # Cross terms
        # We need sum_{k != i, j} (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        # + sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], lj] - b[p[k], li])
        
        # Let's compute the full sums and subtract the i, j terms.
        
        # Term 1: sum_k (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        # Let A_diff = a[i] - a[j]
        # Let B_diff = b[lj, p] - b[li, p]
        # Sum1 = np.dot(A_diff, B_diff)
        
        # Term 2: sum_k (a[k,i] - a[k,j]) * (b[p[k], lj] - b[p[k], li])
        # Let A_col_diff = a[:, i] - a[:, j]
        # Let B_row_diff = b[p, lj] - b[p, li]
        # Sum2 = np.dot(A_col_diff, B_row_diff)
        
        # Total Delta = Sum1 + Sum2 - (terms for k=i in Sum1 and Sum2) - (terms for k=j in Sum1 and Sum2) + d_explicit
        # Wait, the derivation earlier showed that the explicit terms are NOT part of the sums 
        # if we exclude i and j. 
        # If we include i and j in the sums, we have to be careful.
        
        # Let's use the exclusion method to be safe and clear.
        # Create mask for k not in {i, j}
        # This is slow due to mask creation.
        
        # Alternative: Compute full sums and subtract specific terms.
        
        A_diff = a[i] - a[j]
        B_diff = b[lj, p] - b[li, p]
        Sum1_full = np.dot(A_diff, B_diff)
        
        # Terms for k=i in Sum1:
        # k=i: A_diff[i] * B_diff[i] = (a[i,i] - a[j,i]) * (b[lj, li] - b[li, li])
        term1_i = (a_i_i - a_j_i) * (b_lj_li - b_li_li)
        
        # Terms for k=j in Sum1:
        # k=j: A_diff[j] * B_diff[j] = (a[i,j] - a[j,j]) * (b[lj, lj] - b[li, lj])
        term1_j = (a_i_j - a_j_j) * (b_lj_lj - b_li_lj)
        
        Sum1_excl = Sum1_full - term1_i - term1_j
        
        A_col_diff = a[:, i] - a[:, j]
        B_row_diff = b[p, lj] - b[p, li]
        Sum2_full = np.dot(A_col_diff, B_row_diff)
        
        # Terms for k=i in Sum2:
        # k=i: A_col_diff[i] * B_row_diff[i] = (a[i,i] - a[i,j]) * (b[li, lj] - b[li, li])
        # Note: B_row_diff[i] = b[p[i], lj] - b[p[i], li] = b[li, lj] - b[li, li]
        term2_i = (a_i_i - a_i_j) * (b_li_lj - b_li_li)
        
        # Terms for k=j in Sum2:
        # k=j: A_col_diff[j] * B_row_diff[j] = (a[j,i] - a[j,j]) * (b[lj, lj] - b[lj, li])
        # Note: B_row_diff[j] = b[p[j], lj] - b[p[j], li] = b[lj, lj] - b[lj, li]
        term2_j = (a_j_i - a_j_j) * (b_lj_lj - b_lj_li)
        
        Sum2_excl = Sum2_full - term2_i - term2_j
        
        delta = Sum1_excl + Sum2_excl + d_explicit
        return int(delta)

    def local_search(p, current_cost, time_limit):
        """
        Steepest descent local search with vectorized delta computation.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                if time.time() > time_limit:
                    return p, current_cost
                
                # We can vectorize the delta calculation for all j > i?
                # No, because b[lj, p] depends on j.
                # So we loop j.
                
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = j, i # Wait, best_i, best_j = i, j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
        
        return p, current_cost

    # Initial heuristics
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Heuristic 1: Rank matching
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    central_order = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy_order] = central_order
    
    # Heuristic 2: Random
    p2 = np.random.permutation(n).astype(np.int64)
    
    best_p = None
    best_cost = float('inf')
    
    for p_init in [p1, p2]:
        cost = compute_cost(p_init)
        if cost < best_cost:
            best_cost = cost
            best_p = p_init.copy()
    
    # Run local search
    p, current_cost = local_search(best_p.copy(), best_cost, deadline)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
        
    # If time remains, try perturbations
    while time.time() < deadline - 5:
        p_restart = best_p.copy()
        # Perturb by swapping a few random pairs
        num_swaps = max(1, n // 10)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
            
        cost_r = compute_cost(p_restart)
        p_r, cost_r = local_search(p_restart, cost_r, time.time() + 10) # Limit each restart to 10s
        
        if cost_r < best_cost:
            best_cost = cost_r
            best_p = p_r.copy()

    return best_p.tolist()
