"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 118.0
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        # delta = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # Vectorized: compute for all k, then subtract k=i and k=j terms
        diff_a = a[i] - a[j]  # shape (n,)
        diff_b = b[v, p] - b[u, p]  # shape (n,)
        sum_all = np.dot(diff_a, diff_b)
        # Subtract k=i: (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        # Subtract k=j: (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        return int(2 * (sum_all - term_i - term_j))
    
    def local_search(p, current_cost, max_passes=100):
        """First-improvement 2-swap local search."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() - start_time > timeout:
                    break
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, current_cost
    
    def best_improvement_ls(p, current_cost):
        """Best-improvement 2-swap local search for final polish."""
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            for i in range(n):
                if time.time() - start_time > timeout:
                    break
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_i >= 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
        return p, current_cost
    
    # Initial solution: rank matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Local search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # ILS with adaptive perturbation
    p = best_p.copy()
    current_cost = best_cost
    no_improve = 0
    np.random.seed(42)
    
    while time.time() - start_time < timeout - 2.0:
        # Adaptive perturbation strength
        if no_improve < 3:
            k = 2
        elif no_improve < 8:
            k = 4
        elif no_improve < 15:
            k = 7
        else:
            k = 10
        
        # Perturb: k random swaps
        for _ in range(k):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                delta = get_delta(p, i, j)
                p[i], p[j] = p[j], p[i]
                current_cost += delta
        
        # Local search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve = 0
        else:
            no_improve += 1
            p = best_p.copy()
            current_cost = best_cost
        
        if no_improve > 30:
            # Reset and try fresh perturbation
            no_improve = 0
            p = best_p.copy()
            current_cost = best_cost
    
    # Final polish with best-improvement
    p = best_p.copy()
    current_cost = best_cost
    if time.time() - start_time < timeout - 0.5:
        p, current_cost = best_improvement_ls(p, current_cost)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
