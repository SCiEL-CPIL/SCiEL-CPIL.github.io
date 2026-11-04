# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    
    # 1. Symmetrize A for dynamics (Cost is invariant if B is symmetric)
    a_sym = (a + a.T) / 2.0
    
    # 2. MDS Embedding of locations
    # b is distance matrix. We need squared distances for MDS.
    D = b ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    M = -0.5 * J @ D @ J
    # Symmetrize M to handle numerical errors
    M = (M + M.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure eigenvalues are non-negative
    eigenvalues = np.maximum(eigenvalues, 0)
    Y = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    # Gram matrix of locations
    G = Y.T @ Y
    
    # 3. Dynamics Simulation
    # Initialize X (facility coordinates)
    X = Y.copy()
    V = np.zeros_like(X)
    
    # Symplectic Integrator: Velocity Verlet
    # Potential E(X) = -tr(A X G X^T)
    # Force F = -grad E = 2 A X G
    
    dt = 0.01
    num_steps = 1000
    sample_freq = 20
    
    candidates = []
    
    for step in range(num_steps):
        # Compute force at current X
        F = 2 * (a_sym @ X) @ G
        
        # Half step position
        X_half = X + 0.5 * dt * V
        # Project X_half to Stiefel manifold
        U, s, Vt = np.linalg.svd(X_half, full_matrices=False)
        X_half_proj = U @ Vt
        
        # Compute force at X_half
        F_half = 2 * (a_sym @ X_half_proj) @ G
        
        # Update velocity
        V = V + 0.5 * dt * (F + F_half)
        
        # Full step position
        X = X_half_proj + 0.5 * dt * V
        # Project X to Stiefel manifold
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X = U @ Vt
        
        # Sample state
        if step % sample_freq == 0:
            # Discretize X to permutation
            C = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * (X @ Y.T)
            row_ind, col_ind = linear_sum_assignment(C)
            p = col_ind.copy()
            D_p = b[p[:, None], p[None, :]]
            cost = np.sum(a * D_p)
            candidates.append((cost, p))
    
    # 4. Local Search on best candidates
    candidates.sort(key=lambda x: x[0])
    
    best_p = None
    best_cost = float('inf')
    seen = set()
    
    start_time = time.time()
    time_limit = 110
    
    # Process top candidates
    for cost, p in candidates:
        if (time.time() - start_time) > time_limit - 10:
            break
        p_tuple = tuple(p)
        if p_tuple in seen:
            continue
        seen.add(p_tuple)
        
        p = p.copy()
        current_cost = cost
        
        # Optimized 2-opt local search with vectorized delta calculation
        improved = True
        while improved and (time.time() - start_time) < time_limit - 5:
            improved = False
            # Precompute b rows for u and v to speed up access? 
            # Actually, we can vectorize the entire pass.
            
            # We iterate through pairs i < j
            # To make it faster, we can try to find the best swap in a vectorized way?
            # That's complex. Let's stick to the loop but optimize the inner calculation.
            
            for i in range(n - 1):
                if (time.time() - start_time) >= time_limit - 5:
                    break
                u = p[i]
                a_i = a[i, :]
                a_i_T = a[:, i]
                
                for j in range(i + 1, n):
                    v = p[j]
                    a_j = a[j, :]
                    a_j_T = a[:, j]
                    
                    # Mask for k != i, j
                    # Instead of creating a mask array every time, we can compute the full sum and subtract i, j terms?
                    # Or use the vectorized formula derived earlier:
                    # Delta_k_sum = sum_{k != i, j} (a_ik - a_jk)(b[v, p_k] - b[u, p_k]) + (a_ki - a_kj)(b[p_k, v] - b[p_k, u])
                    
                    # Let's compute the full vector for all k, then zero out i and j.
                    # diff_a_row = a_i - a_j
                    # diff_a_col = a_i_T - a_j_T
                    # diff_b_row = b[v, :] - b[u, :]
                    # diff_b_col = b[:, v] - b[:, u]
                    
                    # Term 1: diff_a_row[k] * (b[v, p_k] - b[u, p_k])
                    # Note: b[v, p_k] is the row v of b, indexed by p_k.
                    # Let B_v = b[v, :], B_u = b[u, :]
                    # Then b[v, p_k] is B_v[p_k].
                    # So term 1 is diff_a_row * (B_v[p] - B_u[p])
                    
                    # Term 2: diff_a_col[k] * (b[p_k, v] - b[p_k, u])
                    # b[p_k, v] is the column v of b, indexed by p_k? No, b[p_k, v] is element at row p_k, col v.
                    # Let B_col_v = b[:, v], B_col_u = b[:, u]
                    # Then b[p_k, v] is B_col_v[p_k].
                    # So term 2 is diff_a_col * (B_col_v[p] - B_col_u[p])
                    
                    # We need to sum over k != i, j.
                    # Let's compute the sum over all k, then subtract the terms for k=i and k=j.
                    
                    B_v = b[v, :]
                    B_u = b[u, :]
                    B_col_v = b[:, v]
                    B_col_u = b[:, u]
                    
                    diff_a_row = a_i - a_j
                    diff_a_col = a_i_T - a_j_T
                    
                    # Vectorized products
                    term1_all = diff_a_row * (B_v[p] - B_u[p])
                    term2_all = diff_a_col * (B_col_v[p] - B_col_u[p])
                    
                    delta_k_sum = np.sum(term1_all + term2_all)
                    
                    # Subtract terms for k=i and k=j
                    # For k=i:
                    # term1_i = (a_ii - a_ji) * (b[v, p_i] - b[u, p_i]) = (a_ii - a_ji) * (b[v, u] - b[u, u])
                    # term2_i = (a_ii - a_ji) * (b[p_i, v] - b[p_i, u]) = (a_ii - a_ji) * (b[u, v] - b[u, u])
                    # Note: a_i_T[i] is a[i,i], a_j_T[i] is a[i,j]. So diff_a_col[i] = a[i,i] - a[i,j].
                    # Wait, diff_a_col = a[:, i] - a[:, j]. So diff_a_col[i] = a[i,i] - a[i,j].
                    # diff_a_row[i] = a[i,i] - a[j,i].
                    
                    # Let's just compute the full delta including i and j in the sum, then adjust?
                    # The formula derived was for k != i, j.
                    # The direct terms for i and j are:
                    # Direct = a_ii(b_vv - b_uu) + a_jj(b_uu - b_vv) + a_ij(b_vu - b_uv) + a_ji(b_uv - b_vu)
                    #        = (a_ii - a_jj)(b[v,v] - b[u,u]) + (a_ji - a_ij)(b[u,v] - b[v,u])
                    
                    # So Total Delta = delta_k_sum_excluding_ij + Direct
                    
                    # We computed delta_k_sum_including_ij = sum_{all k} (term1 + term2)
                    # We need to subtract the contributions of k=i and k=j from this sum.
                    
                    # Contribution of k=i to the sum:
                    # t1_i = diff_a_row[i] * (B_v[p[i]] - B_u[p[i]]) = (a[i,i] - a[j,i]) * (b[v, u] - b[u, u])
                    # t2_i = diff_a_col[i] * (B_col_v[p[i]] - B_col_u[p[i]]) = (a[i,i] - a[i,j]) * (b[u, v] - b[u, u])
                    # Sum_i = t1_i + t2_i
                    
                    # Contribution of k=j to the sum:
                    # t1_j = diff_a_row[j] * (B_v[p[j]] - B_u[p[j]]) = (a[i,j] - a[j,j]) * (b[v, v] - b[u, v])
                    # t2_j = diff_a_col[j] * (B_col_v[p[j]] - B_col_u[p[j]]) = (a[j,i] - a[j,j]) * (b[v, v] - b[v, u])
                    # Sum_j = t1_j + t2_j
                    
                    # delta_k_sum_excluding_ij = delta_k_sum_including_ij - Sum_i - Sum_j
                    
                    t1_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    t2_i = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
                    Sum_i = t1_i + t2_i
                    
                    t1_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    t2_j = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                    Sum_j = t1_j + t2_j
                    
                    delta_k_excl = delta_k_sum - Sum_i - Sum_j
                    
                    Direct = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u]) + (a[j, i] - a[i, j]) * (b[u, v] - b[v, u])
                    
                    delta = delta_k_excl + Direct
                    
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p
    
    if best_p is None:
        # Fallback
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
