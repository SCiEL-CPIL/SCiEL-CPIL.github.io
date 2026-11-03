"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 115s
    deadline = time.time() + 115.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        """
        Calculate the exact change in cost when swapping facilities i and j 
        (currently at locations u and v).
        
        Corrected formula based on Inspiration 1:
        The naive vectorized sum includes terms for k=i and k=j which are 
        handled incorrectly by simple subtraction of rows/cols.
        """
        u = p[i]
        v = p[j]
        
        # Naive vectorized terms (sum over all k)
        # S1 = sum_k (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]])
        # S2 = sum_k (a[k, i] - a[k, j]) * (b[p[k], v] - b[p[k], u])
        
        # Optimization: Precompute differences? 
        # For now, direct calculation is O(n) per swap.
        
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        s1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a[:, i] - a[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        s2 = np.dot(diff_a_cols, diff_b_cols)
        
        # The naive sums s1 and s2 include the terms where k=i and k=j.
        # Let's analyze the contribution of k=i and k=j in the true delta.
        
        # True Delta = 
        # (New terms involving i,j) - (Old terms involving i,j)
        # + (Terms where i or j interacts with k != i,j)
        
        # The vectorized s1 + s2 effectively computes:
        # Sum_{k != i, j} (a_ik - a_jk)(b_v,p_k - b_u,p_k) + (a_ki - a_kj)(b_p_k,v - b_p_k,u)
        # PLUS the k=i and k=j terms which are:
        
        # k=i term in s1: (a_ii - a_ji) * (b_v,u - b_u,u)
        # k=i term in s2: (a_ii - a_ji) * (b_u,v - b_u,u)  -- Wait, p[i]=u, so b[p[i], v] = b[u, v]
        
        # Let's use the explicit correction from Inspiration 1 which has been validated.
        
        ai_i = a[i, i]
        aj_i = a[j, i]
        ai_j = a[i, j]
        aj_j = a[j, j]
        
        bu_u = b[u, u]
        bu_v = b[u, v]
        bv_u = b[v, u]
        bv_v = b[v, v]
        
        # Corrections for the k=i and k=j terms that were included in s1/s2 
        # but need to be adjusted to match the exact swap definition.
        
        # The standard derivation shows that the naive sum s1+s2 
        # is equal to the true delta PLUS some error terms that need to be subtracted,
        # OR the naive sum is not the true delta.
        
        # Let's trust the explicit formula from Inspiration 1:
        # Delta = s1 + s2 - (corr_1 + corr_2 + corr_3 + corr_4) + corr_interaction
        
        corr_1 = (ai_i - aj_i) * (bv_u - bu_u)
        corr_2 = (ai_i - aj_i) * (bu_v - bu_u)
        corr_3 = (ai_j - aj_j) * (bv_v - bu_v)
        corr_4 = (ai_j - aj_j) * (bu_v - bv_v)
        
        # Interaction term difference:
        # Old interaction: a[i,j]b[u,v] + a[j,i]b[v,u]
        # New interaction: a[i,j]b[v,u] + a[j,i]b[u,v]
        # Diff: (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        corr_interaction = (ai_j - aj_i) * (bv_u - bu_v)
        
        return s1 + s2 - (corr_1 + corr_2 + corr_3 + corr_4) + corr_interaction

    def local_search(p, current_cost, deadline):
        improved = True
        max_iter = 500 
        iter_count = 0
        
        while improved and time.time() < deadline - 2.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Scan all pairs to find the steepest descent move
            for i in range(n - 1):
                if time.time() >= deadline - 2.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() >= deadline - 2.0:
                        return p, current_cost
                    d = swap_delta(i, j, p)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                iter_count += 1
            else:
                break
                
        return p, current_cost

    # 1. Initialization: Try multiple strategies
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    # Strategy 1: Rank Matching
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    
    # Strategy 2: Reversed Rank Matching
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central[::-1]
    
    # Strategy 3: Random
    p3 = np.random.permutation(n)
    
    inits = [p1, p2, p3]
    
    best_cost = np.inf
    best_p = None
    
    for p_init in inits:
        if time.time() > deadline - 10.0:
            break
        cost = compute_cost(p_init)
        p_loc, cost = local_search(p_init.copy(), cost, deadline)
        if cost < best_cost:
            best_cost = cost
            best_p = p_loc.copy()
            
    # 2. Iterated Local Search (ILS)
    # Use random perturbations to escape local minima
    
    # Safety margin for final return
    while time.time() < deadline - 5.0: 
        # Perturb the best solution
        p = best_p.copy()
        current_cost = best_cost
        
        # Adaptive perturbation: 
        # Start with smaller perturbations, increase if stuck? 
        # For simplicity, use random swaps between 5 and 15.
        num_swaps = np.random.randint(5, 16)
        for _ in range(num_swaps):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Compute cost of perturbed solution
        current_cost = compute_cost(p)
        
        # Run local search from the perturbed state
        p, current_cost = local_search(p, current_cost, deadline)
        
        # Update best if improved
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Early exit if time is running out
        if time.time() > deadline - 10.0:
            break

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    # Fallback (should not happen)
    return list(range(n))

# EVOLVE-BLOCK-END
