"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 117.0  # Safety margin for 120s limit
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    # Multiple initialization strategies
    def init_rank():
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p
    
    def init_spectral():
        # Use eigendecomposition of combined matrix for spectral init
        # Cost = sum a[i,j] * b[p[i],p[j]] = trace(A * B_p) where B_p = P B P^T
        # Spectral approach: maximize trace of A^T @ (P B P^T)
        # Approximate by aligning eigenvectors
        try:
            # Symmetrize for spectral methods
            A_sym = (a + a.T) / 2
            B_sym = (b + b.T) / 2
            evals_a, evecs_a = np.linalg.eigh(A_sym)
            evals_b, evecs_b = np.linalg.eigh(B_sym)
            # Sort by eigenvalue magnitude (descending)
            idx_a = np.argsort(-np.abs(evals_a))
            idx_b = np.argsort(-np.abs(evals_b))
            p = np.empty(n, dtype=np.int64)
            p[idx_a] = idx_b
            return p
        except:
            return init_rank()
    
    def init_random():
        return np.random.permutation(n).astype(np.int64)
    
    # Try multiple initializations, keep best
    best_p = None
    best_cost = np.inf
    for init_func in [init_rank, init_spectral, init_random]:
        p_try = init_func()
        cost_try = calc_cost(p_try)
        if cost_try < best_cost:
            best_cost = cost_try
            best_p = p_try.copy()
    
    p = best_p.copy()
    current_cost = best_cost
    
    # Tabu parameters
    tabu_tenure = 15
    # Use 2D array for tabu: tabu[i,j] = iteration when move (i,j) becomes non-tabu
    tabu_array = np.zeros((n, n), dtype=np.int32)
    iteration = 0
    no_improve_count = 0
    max_no_improve = 80
    max_iterations = 100000
    
    # Precompute for delta calculation
    # For swap of i and j: delta = sum_k [a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]] + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]]
    #                          - sum_k [a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]] + a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]]
    # This can be vectorized.
    
    def compute_delta(i, j, p, current_cost):
        """Compute delta for swapping positions i and j in permutation p."""
        u = p[i]
        v = p[j]
        
        # For all k != i, j:
        # delta_k = a[i,k]*(b[v,p[k]] - b[u,p[k]]) + a[k,i]*(b[p[k],v] - b[p[k],u])
        #          + a[j,k]*(b[u,p[k]] - b[v,p[k]]) + a[k,j]*(b[p[k],u] - b[p[k],v])
        # = (a[i,k] - a[j,k])*(b[v,p[k]] - b[u,p[k]]) + (a[k,i] - a[k,j])*(b[p[k],v] - b[p[k],u])
        
        # And for k=i and k=j, we need special handling.
        
        # Vectorized computation for k != i, j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        diff_a_row = a[i] - a[j]  # shape (n,)
        diff_a_col = a[:, i] - a[:, j]  # shape (n,)
        
        # b[v, p[mask]] - b[u, p[mask]]
        diff_b_row = b[v, p[mask]] - b[u, p[mask]]  # shape (n-2,)
        diff_b_col = b[p[mask], v] - b[p[mask], u]  # shape (n-2,)
        
        # For the bulk (k != i, j)
        bulk = np.dot(diff_a_row[mask], diff_b_row) + np.dot(diff_a_col[mask], diff_b_col)
        
        # Now handle k = i and k = j explicitly
        # When k = i:
        # Original: a[i,i]*b[u,u] + a[i,i]*b[u,u] (wait, let me re-derive)
        
        # Let me re-derive carefully.
        # Cost = sum_{m,n} a[m,n] * b[p[m], p[n]]
        # After swap p[i]<->p[j], the terms involving i or j change.
        
        # Terms where m=i or n=i or m=j or n=j change.
        
        # Let's compute the change directly:
        # delta = sum_k [a[i,k]*b[v,p[k]] + a[k,i]*b[p[k],v] + a[j,k]*b[u,p[k]] + a[k,j]*b[p[k],u]]
        #       - sum_k [a[i,k]*b[u,p[k]] + a[k,i]*b[p[k],u] + a[j,k]*b[v,p[k]] + a[k,j]*b[p[k],v]]
        
        # = sum_k [(a[i,k] - a[j,k])*(b[v,p[k]] - b[u,p[k]]) + (a[k,i] - a[k,j])*(b[p[k],v] - b[p[k],u])]
        
        # This holds for ALL k including k=i and k=j!
        # Let me verify for k=i:
        # Term: (a[i,i] - a[j,i])*(b[v,u] - b[u,u]) + (a[i,i] - a[i,j])*(b[u,v] - b[u,u])
        # Original contribution when k=i in the "old" sum: a[i,i]*b[u,u] + a[i,i]*b[u,u] = 2*a[i,i]*b[u,u]
        # Wait, that's not right. Let me be more careful.
        
        # Actually the formula sum_k [(a[i,k] - a[j,k])*(b[v,p[k]] - b[u,p[k]]) + (a[k,i] - a[k,j])*(b[p[k],v] - b[p[k],u])]
        # should give the total delta. Let me just use this for all k.
        
        diff_a_row_full = a[i] - a[j]  # shape (n,)
        diff_a_col_full = a[:, i] - a[:, j]  # shape (n,)
        
        diff_b_row_full = b[v, p] - b[u, p]  # shape (n,)
        diff_b_col_full = b[p, v] - b[p, u]  # shape (n,)
        
        delta = np.dot(diff_a_row_full, diff_b_row_full) + np.dot(diff_a_col_full, diff_b_col_full)
        
        return int(delta)
    
    while time.time() < deadline:
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Search for best move
        for i in range(n):
            for j in range(i + 1, n):
                # Check tabu
                if tabu_array[i, j] > iteration:
                    # Check aspiration
                    delta = compute_delta(i, j, p, current_cost)
                    if current_cost + delta < best_cost:
                        # Aspiration: allow this move
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                else:
                    delta = compute_delta(i, j, p, current_cost)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
        
        # If no valid move found, pick random
        if best_i == -1:
            best_i = random.randint(0, n-1)
            best_j = random.randint(0, n-1)
            while best_j == best_i:
                best_j = random.randint(0, n-1)
            best_delta = compute_delta(best_i, best_j, p, current_cost)
        
        # Perform the move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update tabu
        tabu_array[best_i, best_j] = iteration + tabu_tenure
        tabu_array[best_j, best_i] = iteration + tabu_tenure
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        iteration += 1
        
        # Stagnation handling
        if no_improve_count > max_no_improve:
            # Perturb with random swaps
            for _ in range(7):
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
            tabu_array.fill(0)
        
        if iteration > max_iterations:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
