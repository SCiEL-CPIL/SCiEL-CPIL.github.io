"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    # Safety margin: 150s limit, aim to finish by 148s
    deadline = time.time() + 148.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        """Compute the change in cost when swapping positions i and j in permutation p."""
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # S1 = sum_k (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]])
        # S2 = sum_k (a[k, i] - a[k, j]) * (b[p[k], v] - b[p[k], u])
        
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        s1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a[:, i] - a[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        s2 = np.dot(diff_a_cols, diff_b_cols)
        
        # Correction for k=i and k=j terms that were double-counted
        # When k=i: a[i,i]*b[u,u] and a[j,i]*b[v,u] appear in s1
        # When k=j: a[i,j]*b[u,v] and a[j,j]*b[v,v] appear in s1
        # Similar for s2 with columns
        
        # The delta formula: new_cost - old_cost
        # Old contribution involving i or j:
        #   sum_{k!=i,j} a[i,k]b[u,p[k]] + sum_{k!=i,j} a[k,i]b[p[k],u] + a[i,j]b[u,v] + a[j,i]b[v,u] + a[i,i]b[u,u] + a[j,j]b[v,v]
        #   Wait, let's be more careful.
        #
        # Total cost = sum_{i,j} a[i,j] b[p[i], p[j]]
        #
        # Terms involving facility i (as first index): sum_j a[i,j] b[u, p[j]]
        # Terms involving facility i (as second index): sum_j a[j,i] b[p[j], u]
        # But when j=i, the term a[i,i]b[u,u] is counted in both sums above? No:
        #   sum_j a[i,j] b[u,p[j]] includes j=i: a[i,i]b[u,u]
        #   sum_j a[j,i] b[p[j],u] includes j=i: a[i,i]b[u,u]
        # So a[i,i]b[u,u] is counted twice in the "terms involving i" if we just sum both.
        #
        # Let's use the standard swap delta formula:
        # delta = sum_{k != i,j} [a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] - a[i,k]b[u,p[k]] - a[k,i]b[p[k],u]]
        #         + a[i,j]b[v,v] + a[j,i]b[u,u] + a[i,j]b[v,u] + a[j,i]b[u,v]
        #         - a[i,j]b[u,v] - a[j,i]b[v,u] - a[i,i]b[u,u] - a[j,j]b[v,v]
        #
        # Wait, let me redo this more carefully.
        #
        # In the original assignment, facility i is at location u, facility j is at location v.
        # After swap, facility i is at v, facility j is at u.
        #
        # Change in cost = (new terms) - (old terms)
        #
        # Old terms that change (involving i or j as either index):
        #   For pair (i, k) where k != i,j: a[i,k] * b[u, p[k]] -> a[i,k] * b[v, p[k]]
        #   For pair (k, i) where k != i,j: a[k,i] * b[p[k], u] -> a[k,i] * b[p[k], v]
        #   For pair (i, j): a[i,j] * b[u, v] -> a[i,j] * b[v, u]
        #   For pair (j, i): a[j,i] * b[v, u] -> a[j,i] * b[u, v]
        #   For pair (i, i): a[i,i] * b[u, u] -> a[i,i] * b[v, v]
        #   For pair (j, j): a[j,j] * b[v, v] -> a[j,j] * b[u, u]
        #
        # New terms - Old terms:
        #   sum_{k!=i,j} a[i,k]*(b[v,p[k]] - b[u,p[k]])
        #   + sum_{k!=i,j} a[k,i]*(b[p[k],v] - b[p[k],u])
        #   + a[i,j]*(b[v,u] - b[u,v])
        #   + a[j,i]*(b[u,v] - b[v,u])
        #   + a[i,i]*(b[v,v] - b[u,u])
        #   + a[j,j]*(b[u,u] - b[v,v])
        #
        # The first two sums can be computed as:
        #   sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) - a[i,i]*(b[v,u] - b[u,u]) - a[i,j]*(b[v,v] - b[u,v])
        #   + sum_k a[k,i]*(b[p[k],v] - b[p[k],u]) - a[i,i]*(b[u] - b[u]) ... wait
        #
        # Let me just compute it directly with the dot products and subtract the self-terms.
        
        # sum_k a[i,k] * (b[v,p[k]] - b[u,p[k]])
        # = sum_k (a[i,k]) * (b[v,p[k]] - b[u,p[k]])
        # This includes k=i: a[i,i]*(b[v,u] - b[u,u]) and k=j: a[i,j]*(b[v,v] - b[u,v])
        
        # sum_k a[k,i] * (b[p[k],v] - b[p[k],u])
        # This includes k=i: a[i,i]*(b[u,v] - b[u,u]) and k=j: a[j,i]*(b[v,v] - b[v,u])
        
        s1_full = np.dot(a[i, :], b[v, p] - b[u, p])
        s2_full = np.dot(a[:, i], b[p, v] - b[p, u])
        
        # Subtract the k=i and k=j terms from s1_full
        s1_ki = a[i, i] * (b[v, u] - b[u, u])
        s1_kj = a[i, j] * (b[v, v] - b[u, v])
        
        # Subtract the k=i and k=j terms from s2_full
        s2_ki = a[i, i] * (b[u, v] - b[u, u])
        s2_kj = a[j, i] * (b[v, v] - b[v, u])
        
        # Add back the direct i-j and i-i, j-j terms
        direct_terms = (a[i,j]*(b[v,u] - b[u,v]) + a[j,i]*(b[u,v] - b[v,u]) 
                        + a[i,i]*(b[v,v] - b[u,u]) + a[j,j]*(b[u,u] - b[v,v]))
        
        delta = s1_full - s1_ki - s1_kj + s2_full - s2_ki - s2_kj + direct_terms
        
        return delta

    def local_search(p, current_cost, deadline, max_iter=100):
        """Steepest descent local search using swap moves."""
        improved = True
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

    # --- Initialization Strategies ---
    inits = []

    # 1. Rank-based initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy] = central[::-1]
    
    inits.extend([p_rank, p_rank_rev])

    # 2. Hungarian Algorithm Initialization (Linear Relaxation)
    try:
        if time.time() < deadline - 10.0:
            C = a @ b
            row_ind, col_ind = linear_sum_assignment(C)
            p_hung = np.empty(n, dtype=np.int64)
            p_hung[row_ind] = col_ind
            inits.append(p_hung)
    except Exception:
        pass

    # 3. Spectral initialization
    try:
        if time.time() < deadline - 30.0:
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            
            a_scale = np.sqrt(np.mean(a_centered ** 2)) + 1e-10
            b_scale = np.sqrt(np.mean(b_centered ** 2)) + 1e-10
            
            a_norm = a_centered / a_scale
            b_norm = b_centered / b_scale
            
            C = 0.5 * (a_norm + a_norm.T) + 0.5 * (b_norm + b_norm.T)
            C = 0.5 * (C + C.T)
            
            k = 5
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigvecs = eigenvectors[:, n-k:]
            
            for idx in range(k):
                eigvec = eigvecs[:, idx]
                facility_order = np.argsort(eigvec, kind="stable")
                location_order = np.argsort(-eigvec, kind="stable")
                
                p = np.empty(n, dtype=np.int64)
                p[facility_order] = location_order
                inits.append(p)
    except Exception:
        pass

    # 4. Random initializations
    rng = np.random.RandomState(42)
    for _ in range(5):
        inits.append(rng.permutation(n))

    # --- Main Search Loop ---
    
    best_cost = np.inf
    best_p = None
    
    # Run local search from each initialization
    for p_init in inits:
        if time.time() > deadline - 30.0:
            break
        
        p_work = p_init.copy()
        cost = compute_cost(p_work)
        
        p_work, cost = local_search(p_work, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
    
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)

    # --- Iterated Local Search (ILS) with Adaptive Perturbation ---
    
    no_improve_count = 0
    base_perturb = 3
    max_perturb = 20
    
    while time.time() < deadline - 15.0:
        p = best_p.copy()
        
        # Adaptive perturbation size
        perturb_size = base_perturb + min(no_improve_count, 15)
        perturb_size = min(perturb_size, max_perturb)
        
        # Occasionally do a large jump (25% chance)
        if rng.rand() < 0.25:
            perturb_size = max_perturb + rng.randint(0, 15)
        
        num_swaps = max(2, perturb_size)
        for _ in range(num_swaps):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = compute_cost(p)
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Reset counter if it gets too high to allow exploration
        if no_improve_count > 30:
            no_improve_count = 10

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    return list(range(n))

# EVOLVE-BLOCK-END
