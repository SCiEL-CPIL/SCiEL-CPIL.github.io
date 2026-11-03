"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Pre-compute some properties if needed, but a and b are fixed.
    
    def calc_cost(p):
        # p is permutation of 0..n-1
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # We can do this as np.sum(a * b[np.ix_(p, p)])
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j in permutation p.
        p[i] -> u, p[j] -> v.
        """
        u = p[i]
        v = p[j]
        
        # We need: 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # Let diff_a = a[i] - a[j]
        # Let diff_b_k = b[v, p[k]] - b[u, p[k]]
        
        # Vectorized calculation
        # p is array of locations.
        # b_v = b[v, p] gives array of length n: b[v, p[0]], b[v, p[1]], ...
        # b_u = b[u, p]
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        # Total sum over all k
        sum_all = np.sum(diff_a * diff_b)
        
        # Subtract terms for k=i and k=j
        # Term k=i: diff_a[i] * diff_b[i]
        # diff_b[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        
        # Term k=j: diff_a[j] * diff_b[j]
        # diff_b[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta

    def local_search(p, current_cost):
        # First Improvement Local Search - much faster, allows more passes
        improved = True
        while improved:
            if time.time() - start_time > timeout:
                break
            improved = False
            
            for i in range(n):
                if time.time() - start_time > timeout:
                    return p, current_cost
                
                u = p[i]
                b_u_p = b[u, p]
                a_i = a[i]
                
                best_delta = 0
                best_j = -1
                
                for j in range(i + 1, n):
                    v = p[j]
                    diff_b = b[v, p] - b_u_p
                    diff_a = a_i - a[j]
                    sum_all = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    delta = 2 * (sum_all - term_i - term_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j != -1:
                    j = best_j
                    p[i], p[j] = p[j], p[i]
                    current_cost += best_delta
                    improved = True
                    break # Restart scan from i=0 for first improvement
        
        return p, current_cost

    def spectral_init():
        # Spectral initialization using eigenvectors of flow and distance matrices
        # Center the matrices
        a_centered = a - a.mean(axis=1, keepdims=True)
        b_centered = b - b.mean(axis=1, keepdims=True)
        
        # Compute eigenvectors
        try:
            # For symmetric matrices, we can use eigh
            # But a and b might not be symmetric in general QAP, though wil100 likely is
            # Let's assume symmetry for wil100
            evals_a, evecs_a = np.linalg.eigh(a_centered)
            evals_b, evecs_b = np.linalg.eigh(b_centered)
            
            # Take the top eigenvector (largest eigenvalue)
            # eigh returns eigenvalues in ascending order
            top_idx_a = np.argmax(evals_a)
            top_idx_b = np.argmax(evals_b)
            
            vec_a = evecs_a[:, top_idx_a]
            vec_b = evecs_b[:, top_idx_b]
            
            # Assign facilities with high vec_a values to locations with high vec_b values
            fac_order = np.argsort(-vec_a)
            loc_order = np.argsort(-vec_b)
            
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            return p
        except:
            # Fallback to rank matching
            flow_sums = a.sum(axis=1)
            dist_sums = b.sum(axis=1)
            heavy_facilities = np.argsort(-flow_sums, kind="stable")
            central_locations = np.argsort(dist_sums, kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy_facilities] = central_locations
            return p

    start_time = time.time()
    timeout = 115.0
    
    # 1. Initial Solution: Spectral Initialization
    p = spectral_init()
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS Loop with adaptive perturbation
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        # Perturbation: vary the number of swaps based on stagnation
        if no_improve_count < 5:
            num_swaps = 3
        elif no_improve_count < 15:
            num_swaps = 7
        else:
            num_swaps = 12
        
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = get_delta(p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reset p to best_p to stay in the neighborhood of the best solution
            p = best_p.copy()
            current_cost = best_cost
            
        # Check timeout before next iteration
        if time.time() - start_time > timeout - 5.0:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
