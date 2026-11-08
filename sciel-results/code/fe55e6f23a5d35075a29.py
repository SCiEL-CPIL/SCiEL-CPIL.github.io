import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    time_limit = 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor M (dense)
    # M[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i
    M = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0
                
    # Function to verify and snap
    def verify_and_snap(U, V, W, tol=1e-6):
        # Check error
        err = 0.0
        # Efficient check: compute tensor product and compare
        # But M is small, we can just compute the sum
        # Sum_r U[:,r] V[:,r] W[:,r]
        # This is a batch matmul or einsum
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # np.einsum('ar,br,cr->abc', U, V, W)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - M)
        if err < tol:
            # Snap to nearest half-integer
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify snapped version
            T_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            if np.linalg.norm(T_snap - M) < 1e-9:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        return None

    # Seed solution
    seed_rank = n * m * p
    U_seed = np.zeros((dim_u, seed_rank))
    V_seed = np.zeros((dim_v, seed_rank))
    W_seed = np.zeros((dim_w, seed_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    seed_result = {
        "U": U_seed.tolist(),
        "V": V_seed.tolist(),
        "W": W_seed.tolist()
    }
    
    # Try to find lower rank
    # We try ranks from seed_rank - 1 down to max(30, seed_rank - 10)
    for rank in range(seed_rank - 1, max(30, seed_rank - 10) - 1, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few restarts for this rank
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + restart + rank)
            U = np.random.randn(dim_u, rank) * 0.1
            V = np.random.randn(dim_v, rank) * 0.1
            W = np.random.randn(dim_w, rank) * 0.1
            
            # ALS Iterations
            # We want to solve min || M - sum U_r V_r^T W_r ||
            # This is not standard ALS for CP because the indices are different.
            # M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is a 3-way tensor decomposition (CP).
            # We can use standard CP ALS.
            
            for it in range(50):
                # Update U: fix V, W.
                # M[a, :, :] = sum_r U[a,r] (V[:,r] W[:,r]^T)
                # Let Z_r = V[:,r] * W[:,r] (outer product)
                # M is (dim_u, dim_v, dim_w)
                # We want U[a,r] such that sum_r U[a,r] Z_r[a_v, a_w] approx M[a, :, :]
                # This is a linear least squares problem for each r? No, for each a?
                # Actually, for CP, we can update one factor at a time.
                # Fix V, W. Compute M_unfolded along mode 1.
                # M_(1) is (dim_u, dim_v * dim_w)
                # M_(1)[a, b*c+w] = M[a,b,c]
                # We want M_(1) approx U * (V * W)_(1)^T?
                # Let's use the property:
                # M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix V, W.
                # For each a, we have a vector M[a, :, :].
                # M[a, :, :] = sum_r U[a,r] (V[:,r] * W[:,r])^T ? No.
                # (V[:,r] * W[:,r]) is a vector of length dim_v * dim_w?
                # No, V[:,r] is length dim_v. W[:,r] is length dim_w.
                # Their outer product is dim_v x dim_w.
                # So M[a, :, :] = sum_r U[a,r] (V[:,r] W[:,r]^T).
                # Let Z_r = V[:,r] W[:,r]^T.
                # We want to approximate M[a, :, :] by sum_r U[a,r] Z_r.
                # This is a matrix factorization problem: M_(1) approx U Z^T?
                # M_(1) has shape (dim_u, dim_v * dim_w).
                # Z^T has shape (rank, dim_v * dim_w).
                # So M_(1) approx U Z^T.
                # Solve U = M_(1) Z Z^T)^{-1} M_(1) Z^T?
                # Let's form the matrix Z_mat where Z_mat[r, :] is vec(Z_r).
                # Z_mat shape (rank, dim_v * dim_w).
                # M_(1) shape (dim_u, dim_v * dim_w).
                # U = M_(1) @ np.linalg.pinv(Z_mat.T)
                
                # Update U
                V_W_outer = np.einsum('br,cr->bc', V, W) # (dim_v, dim_w) for each r? No.
                # We need Z_r for each r.
                # V is (dim_v, rank). W is (dim_w, rank).
                # Z_r = V[:, r] W[:, r]^T.
                # We can compute this in a loop or batch.
                # Z_mat[r, b, c] = V[b, r] W[c, r]
                Z_mat = np.einsum('br,cr->bcr', V, W) # (dim_v, dim_w, rank)
                Z_mat_flat = Z_mat.transpose(2, 0, 1).reshape(rank, dim_v * dim_w) # (rank, dim_v*dim_w)
                
                M_1 = M.reshape(dim_u, dim_v * dim_w)
                U = M_1 @ np.linalg.pinv(Z_mat_flat.T)
                
                # Update V
                U_W_outer = np.einsum('ar,cr->acr', U, W) # (dim_u, dim_w, rank)
                U_W_flat = U_W_outer.transpose(2, 0, 1).reshape(rank, dim_u * dim_w)
                M_2 = M.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                V = M_2 @ np.linalg.pinv(U_W_flat.T)
                
                # Update W
                U_V_outer = np.einsum('ar,br->abr', U, V) # (dim_u, dim_v, rank)
                U_V_flat = U_V_outer.transpose(2, 0, 1).reshape(rank, dim_u * dim_v)
                M_3 = M.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                W = M_3 @ np.linalg.pinv(U_V_flat.T)
                
                # Check convergence
                if it % 10 == 9:
                    T = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(T - M)
                    if err < 1e-4:
                        break
                
            # Final check
            result = verify_and_snap(U, V, W)
            if result:
                return result

    return seed_result
