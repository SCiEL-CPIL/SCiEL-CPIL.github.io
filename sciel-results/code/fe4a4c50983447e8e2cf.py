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

    def local_search(p, current_cost, max_iterations=1000):
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            improved = False
            iters += 1
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

    def simulated_annealing(p, current_cost, T_start, T_end, iterations):
        best_p_local = p.copy()
        best_cost_local = current_cost
        
        # Linear cooling schedule
        step = (T_start - T_end) / iterations
        T = T_start
        
        for _ in range(iterations):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = get_delta(i, j, p, current_cost)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost_local:
                    best_cost_local = current_cost
                    best_p_local = p.copy()
            
            T -= step
            if T < T_end:
                T = T_end
                
        return p, current_cost, best_p_local, best_cost_local

    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # ILS Loop
    # We will alternate between Local Search and Simulated Annealing
    # Or just use SA with restarts?
    # Let's stick to ILS with Local Search as the core, but maybe add SA for perturbation?
    # The current program uses random swaps for perturbation.
    # Let's try: LS -> Perturb (SA) -> LS
    
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
        # Use a short run of Simulated Annealing as perturbation
        # Start T high enough to escape, but not too high to destroy the solution
        # Let's try T_start = 100, T_end = 10, 100 iterations
        T_start = 100.0
        T_end = 10.0
        sa_iters = 200
        
        p, current_cost, _, _ = simulated_annealing(p, current_cost, T_start, T_end, sa_iters)
        
        # Optionally, we can do a quick local search after perturbation before the main loop continues?
        # The main loop will do LS at the start of the next iteration.
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
