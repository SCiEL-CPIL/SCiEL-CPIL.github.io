import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Target tensor T (sparse representation)
    # T[i, j, k] = 1
    # Flattened indices:
    # u_idx = i * m + j
    # v_idx = j * p + k
    # w_idx = k * n + i
    # We need sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r] == 1 for all (i,j,k)
    # and == 0 for other combinations? 
    # Wait, the tensor T is only defined on the "valid" triples?
    # No, the tensor T is in R^{nm x mp x np}.
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k).
    # T[a, b, c] = 0 otherwise.
    
    # We will check the condition on the non-zero entries of T.
    # And we must ensure that for entries where T=0, the sum is 0.
    # However, usually in these problems, we only require the decomposition to match the non-zero entries 
    # IF the zero entries are automatically satisfied by the structure? 
    # No, the tensor is dense in the sense that we must match ALL entries.
    # But T is sparse. There are nm*mp*np total entries. 
    # Only nmp are non-zero.
    # We must ensure the sum is 0 for all other entries.
    
    # Let's try ranks from lower bound up to schoolbook.
    # Lower bound: np + mp - 1 = 28 + 28 - 1 = 55.
    
    # We'll try a few specific ranks and random restarts.
    ranks_to_try = [55, 56, 58, 60, 64, 70, 80, 90, 100, 110]
    
    best_result = None
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Precompute indices for non-zero entries
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                nonzero_indices.append((u_idx, v_idx, w_idx))
                
    # Total entries in flattened tensor space
    num_u = n * m
    num_v = m * p
    num_w = p * n
    total_entries = num_u * num_v * num_w
    
    # To check validity, we need to verify the sum for ALL entries.
    # This is expensive if done naively for every iteration.
    # We can check only non-zero entries during optimization, 
    # and do a full check only when rounding.
    
    def check_decomposition(U, V, W):
        # Check non-zero entries
        for u_idx, v_idx, w_idx in nonzero_indices:
            val = 0
            for r in range(U.shape[1]):
                val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            if not np.isclose(val, 1.0, atol=1e-9):
                return False
        # Check zero entries (sample some if too slow, but dimensions are small)
        # num_u=16, num_v=28, num_w=28. Total 12544 entries.
        # 112 are non-zero. 12432 are zero.
        # We can check all of them.
        for u_idx in range(num_u):
            for v_idx in range(num_v):
                # Determine if this (u_idx, v_idx) corresponds to a valid j
                i_u = u_idx // m
                j_u = u_idx % m
                j_v = v_idx // p
                k_v = v_idx % p
                # If j_u != j_v, the entry must be 0
                if j_u != j_v:
                    val = 0
                    for r in range(U.shape[1]):
                        val += U[u_idx, r] * V[v_idx, r] * W[k_v * n + i_u, r] # w_idx depends on k, i
                        # Wait, w_idx is fixed by the triple.
                        # The tensor entry T[u_idx, v_idx, w_idx] is 0 if the indices don't align.
                        # We need to check T[u_idx, v_idx, w_idx] for all w_idx.
                        # This is getting complicated to iterate.
                        # Let's just compute the full tensor product and compare.
        # Simpler: Compute the full tensor T_approx and compare with T_true.
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        T_true = np.zeros((num_u, num_v, num_w))
        for u_idx, v_idx, w_idx in nonzero_indices:
            T_true[u_idx, v_idx, w_idx] = 1.0
        if not np.allclose(T_approx, T_true, atol=1e-9):
            return False
        return True

    for R in ranks_to_try:
        if time.time() > deadline:
            break
        if R >= best_rank:
            continue
            
        for attempt in range(20): # 20 restarts per rank
            if time.time() > deadline:
                break
                
            np.random.seed(42 + attempt + R)
            U = np.random.randn(num_u, R) * 0.5
            V = np.random.randn(num_v, R) * 0.5
            W = np.random.randn(num_w, R) * 0.5
            
            # Alternating minimization
            for step in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                # T_approx = sum_r U[:,r] outer V[:,r] outer W[:,r]
                # Fix V, W.
                # We want U such that sum_r U[:,r] * (V[:,r] * W[:,r]) approx T
                # This is a linear system for U.
                # Let Z_r = V[:,r] * W[:,r] (element-wise product, shape num_v * num_w)
                # T = sum_r U[:,r] outer Z_r
                # Reshape T to (num_u, num_v * num_w)
                T_flat = np.zeros((num_u, num_v * num_w))
                for u_idx, v_idx, w_idx in nonzero_indices:
                    T_flat[u_idx, v_idx * num_w + w_idx] = 1.0
                
                # Construct matrix M where M[:, u_idx, :] = sum_r U[u_idx, r] * Z_r.flatten()
                # This is equivalent to solving for U given V, W.
                # We can use least squares.
                # The equation is: T_flat = U @ M_Z
                # Where M_Z is a matrix of size (R, num_v * num_w) ? No.
                # T_flat[u, x] = sum_r U[u, r] * Z_r[x]
                # So T_flat = U @ Z_matrix
                # Where Z_matrix[r, x] = Z_r[x] = V[r, v] * W[r, w] for x = v*num_w+w
                Z_matrix = np.zeros((R, num_v * num_w))
                for r in range(R):
                    Z_r = V[:, r] * W[:, r] # shape (num_v, num_w)
                    Z_matrix[r, :] = Z_r.flatten()
                
                # Solve U @ Z_matrix = T_flat
                # U = T_flat @ pinv(Z_matrix)
                try:
                    U_new, residuals, rank_U, s = np.linalg.lstsq(Z_matrix.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Update V
                # T_flat2 = zeros((num_v, num_u * num_w))
                T_flat2 = np.zeros((num_v, num_u * num_w))
                for u_idx, v_idx, w_idx in nonzero_indices:
                    T_flat2[v_idx, u_idx * num_w + w_idx] = 1.0
                    
                Z_matrix2 = np.zeros((R, num_u * num_w))
                for r in range(R):
                    Z_r = U[:, r] * W[:, r]
                    Z_matrix2[r, :] = Z_r.flatten()
                    
                try:
                    V_new, residuals, rank_V, s = np.linalg.lstsq(Z_matrix2.T, T_flat2.T, rcond=None)
                    V = V_new.T
                except:
                    pass

                # Update W
                T_flat3 = np.zeros((num_w, num_u * num_v))
                for u_idx, v_idx, w_idx in nonzero_indices:
                    T_flat3[w_idx, u_idx * num_v + v_idx] = 1.0
                    
                Z_matrix3 = np.zeros((R, num_u * num_v))
                for r in range(R):
                    Z_r = U[:, r] * V[:, r]
                    Z_matrix3[r, :] = Z_r.flatten()
                    
                try:
                    W_new, residuals, rank_W, s = np.linalg.lstsq(Z_matrix3.T, T_flat3.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                best_rank = R
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
                break # Found a valid decomposition for this R, try smaller R?
                # Actually, we iterate R ascending, so first found is minimal.
                # But we should check if we can find a smaller one?
                # The loop goes 55, 56... so if 55 works, we break.
                # If 55 fails, we try 56.
                
    # Convert best result to list of lists
    if best_rank < rank_school:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = U_school.tolist()
        V_list = V_school.tolist()
        W_list = W_school.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
