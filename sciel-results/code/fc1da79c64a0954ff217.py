"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # 1. Initial Heuristic: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Helper to compute delta for swapping facilities i and j
    # p[i] = u, p[j] = v
    # Delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    # Note: We assume a and b are symmetric. If not, the formula is more complex.
    # QAPLIB instances typically have symmetric matrices.
    def get_delta(i, j, p, current_cost):
        u = p[i]
        v = p[j]
        
        # Vectorized computation for speed
        # Create a mask for k != i, j
        # We want sum over k of (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # Then subtract the k=i and k=j terms if they were included, or just exclude them.
        
        # Let's just loop for clarity and safety, n=100 is small.
        # Or use numpy which is fast enough.
        
        # Extract rows
        ai = a[i]
        aj = a[j]
        bv = b[v]
        bu = b[u]
        
        # p is the permutation array
        # b[v, p[k]] is bv[p[k]]
        # b[u, p[k]] is bu[p[k]]
        
        diff_a = ai - aj
        # b_diff_k = bv[p[k]] - bu[p[k]]
        
        # We need to exclude k=i and k=j
        # Let's compute the full sum and subtract the i and j terms
        # Term for k: diff_a[k] * (bv[p[k]] - bu[p[k]])
        
        # Full sum
        sum_all = np.sum(diff_a * (bv[p] - bu[p]))
        
        # Subtract k=i
        # k=i: diff_a[i] * (bv[p[i]] - bu[p[i]]) = (a[i,i]-a[j,i]) * (bv[u] - bu[u])
        term_i = (ai[i] - aj[i]) * (bv[u] - bu[u])
        
        # Subtract k=j
        # k=j: diff_a[j] * (bv[p[j]] - bu[p[j]]) = (a[i,j]-a[j,j]) * (bv[v] - bu[v])
        term_j = (ai[j] - aj[j]) * (bv[v] - bu[v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return int(delta)

    def local_search(p, current_cost):
        improved = True
        while improved:
            improved = False
            # Iterate over all pairs? Or random?
            # For n=100, 4950 pairs. 
            # Doing a full pass is good for local search.
            for i in range(n):
                for j in range(i + 1, n):
                    delta = get_delta(i, j, p, current_cost)
                    if delta < 0:
                        # Accept
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # ILS Loop
    while time.time() - start_time < timeout:
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Check time
        if time.time() - start_time > timeout:
            break
            
        # Perturbation
        # Perform a few random swaps to escape local optimum
        num_swaps = 5 + np.random.randint(10)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            # Calculate delta to update cost correctly even if we don't accept
            # Actually, for perturbation, we just swap and update cost
            delta = get_delta(i, j, p, current_cost)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
