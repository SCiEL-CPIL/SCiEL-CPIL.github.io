"""Matrix multiplication 2x3x10: rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 10]

def _build_rank55():
    """Build the rank-55 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part A: j=0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: 5 blocks of 2x2 Strassen
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        U[u_11][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1; V[v_22][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        col += 1

        U[u_21][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1; W[w_11][col] += -1
        col += 1

        U[u_11][col] += 1
        V[v_12][col] += 1; V[v_22][col] += -1
        W[w_01][col] += 1; W[w_11][col] += 1
        col += 1

        U[u_22][col] += 1
        V[v_21][col] += 1; V[v_11][col] += -1
        W[w_00][col] += 1; W[w_10][col] += 1
        col += 1

        U[u_11][col] += 1; U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1; W[w_01][col] += 1
        col += 1

        U[u_21][col] += 1; U[u_11][col] += -1
        V[v_11][col] += 1; V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_12][col] += 1; U[u_22][col] += -1
        V[v_21][col] += 1; V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    return U, V, W, rank

def _verify(U, V, W, rank, tol=1e-10):
    """Verify the decomposition is exact."""
    n, m, p = DIMENSIONS
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_rec = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        T_rec += np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, k*n+i] = 1.0
    
    return np.allclose(T_rec, T_true, atol=tol)

def _try_reduce_rank(U, V, W, rank, deadline, max_iters=2000):
    """Try to reduce rank by finding linear dependencies among rank-1 tensors."""
    n, m, p = DIMENSIONS
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # Build the tensor T from the rank-r decomposition
    # T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # We want to check if we can remove one column.
    # If column r is a linear combination of the others in the tensor space,
    # we can remove it.
    
    # The tensor space has dimension dim_u * dim_v * dim_w
    # Each rank-1 tensor u_c outer v_c outer w_c is a vector in this space.
    # We want to check if the last vector is in the span of the first r-1.
    
    # Flatten each rank-1 tensor
    def flatten_rank1(u, v, w):
        return np.einsum('i,j,k->ijk', u, v, w).flatten()
    
    # Build matrix of first r-1 rank-1 tensors
    # This is expensive: (r-1) x (dim_u*dim_v*dim_w)
    # dim_u*dim_v*dim_w = 6*30*20 = 3600
    # r-1 = 54, so matrix is 54 x 3600
    
    # Check if we can drop the last column
    # Solve: A * x = b where A is 3600 x 54, b is the last tensor flattened
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    for attempt in range(rank - 1):
        if time.time() > deadline - 10:
            break
        
        # Try dropping column 'attempt'
        # Keep all columns except 'attempt'
        keep = [c for c in range(rank) if c != attempt]
        
        # Build A: 3600 x len(keep)
        A = np.zeros((dim_u * dim_v * dim_w, len(keep)))
        for idx, c in enumerate(keep):
            A[:, idx] = flatten_rank1(U_np[:, c], V_np[:, c], W_np[:, c])
        
        # b = the tensor we want to drop
        b = flatten_rank1(U_np[:, attempt], V_np[:, attempt], W_np[:, attempt])
        
        # Solve least squares
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except Exception:
            continue
        
        # Check if b is in the span of A
        # residual = b - A @ coeffs
        residual = b - A @ coeffs
        if np.linalg.norm(residual) < 1e-8:
            # We can drop column 'attempt'
            new_rank = rank - 1
            new_U = [U[i] for i in range(dim_u)]
            new_V = [V[i] for i in range(dim_v)]
            new_W = [W[i] for i in range(dim_w)]
            
            # Reconstruct with kept columns
            for idx, c in enumerate(keep):
                for i in range(dim_u):
                    new_U[i].append(U[i][c])
                for i in range(dim_v):
                    new_V[i].append(V[i][c])
                for i in range(dim_w):
                    new_W[i].append(W[i][c])
            
            # Verify
            if _verify(new_U, new_V, new_W, new_rank):
                best_rank = new_rank
                best_U, best_V, best_W = new_U, new_V, new_W
                rank = new_rank
                U, V, W = new_U, new_V, new_W
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                # Restart the dropping process with the new rank
                break
        else:
            continue
    
    return best_U, best_V, best_W, best_rank

def _try_als_reduction(U, V, W, rank, deadline, max_iters=500):
    """Try ALS to find a lower-rank approximation that is still exact."""
    n, m, p = DIMENSIONS
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for c in range(rank):
        T += np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try reducing by 1
    target_rank = rank - 1
    if target_rank < 1:
        return best_U, best_V, best_W, best_rank
    
    # Initialize with random perturbation of the first target_rank columns
    U_init = U_np[:, :target_rank].copy()
    V_init = V_np[:, :target_rank].copy()
    W_init = W_np[:, :target_rank].copy()
    
    # Add small random perturbation
    U_init += np.random.randn(dim_u, target_rank) * 0.01
    V_init += np.random.randn(dim_v, target_rank) * 0.01
    W_init += np.random.randn(dim_w, target_rank) * 0.01
    
    for it in range(max_iters):
        if time.time() > deadline - 15:
            break
        
        # Update U: fix V, W, solve for U
        # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # For each i: T[i,:,:] = sum_c U[i,c] * (V[:,c] outer W[:,c])
        # Let M_i[c, j, k] = V[j,c] * W[k,c], then T[i,:,:] = sum_c U[i,c] * M_i[c,:,:]
        # This is a linear system: T_flat[i] = U[i,:] @ M_flat
        
        # Reshape T to (dim_u, dim_v*dim_w)
        T_2d = T.reshape(dim_u, dim_v * dim_w)
        
        # For each i, build the system
        # T_2d[i, :] = sum_c U[i,c] * (V[:,c] outer W[:,c]).flatten()
        # Let B_i[c, j*k] = V[j,c] * W[k,c]
        # Then T_2d[i, :] = U[i,:] @ B_i.T where B_i is (target_rank, dim_v*dim_w)
        
        # Build B: (target_rank, dim_v*dim_w)
        B = np.zeros((target_rank, dim_v * dim_w))
        for c in range(target_rank):
            B[c, :] = np.einsum('j,k->jk', V_init[:, c], W_init[:, c]).flatten()
        
        # Solve: T_2d[i, :] = U[i,:] @ B.T
        # U[i,:] = T_2d[i, :] @ (B.T @ B)^-1 @ B
        # Use least squares
        try:
            U_new, _, _, _ = np.linalg.lstsq(B.T, T_2d.T, rcond=1e-10)
        except Exception:
            break
        U_init = U_new.T
        
        # Update V
        T_2d_v = T.reshape(dim_v, dim_u * dim_w)
        B_v = np.zeros((target_rank, dim_u * dim_w))
        for c in range(target_rank):
            B_v[c, :] = np.einsum('i,k->ik', U_init[:, c], W_init[:, c]).flatten()
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_v.T, T_2d_v.T, rcond=1e-10)
        except Exception:
            break
        V_init = V_new.T
        
        # Update W
        T_2d_w = T.reshape(dim_w, dim_u * dim_v)
        B_w = np.zeros((target_rank, dim_u * dim_v))
        for c in range(target_rank):
            B_w[c, :] = np.einsum('i,j->ij', U_init[:, c], V_init[:, c]).flatten()
        try:
            W_new, _, _, _ = np.linalg.lstsq(B_w.T, T_2d_w.T, rcond=1e-10)
        except Exception:
            break
        W_init = W_new.T
        
        # Check convergence
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for c in range(target_rank):
            T_rec += np.einsum('i,j,k->ijk', U_init[:, c], V_init[:, c], W_init[:, c])
        
        err = np.linalg.norm(T_rec - T)
        if err < 1e-8:
            # Found exact decomposition with lower rank
            # Convert to integer/half-integer if possible
            U_list = [[round(x * 2) / 2 for x in row] for row in U_init.tolist()]
            V_list = [[round(x * 2) / 2 for x in row] for row in V_init.tolist()]
            W_list = [[round(x * 2) / 2 for x in row] for row in W_init.tolist()]
            
            if _verify(U_list, V_list, W_list, target_rank, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                return best_U, best_V, best_W, best_rank
            else:
                # Try with float verification
                if _verify(U_init.tolist(), V_init.tolist(), W_init.tolist(), target_rank, tol=1e-6):
                    best_rank = target_rank
                    best_U = [[float(x) for x in row] for row in U_init.tolist()]
                    best_V = [[float(x) for x in row] for row in V_init.tolist()]
                    best_W = [[float(x) for x in row] for row in W_init.tolist()]
                    return best_U, best_V, best_W, best_rank
        
        # Normalize to prevent divergence
        for c in range(target_rank):
            norm_u = np.linalg.norm(U_init[:, c])
            norm_v = np.linalg.norm(V_init[:, c])
            norm_w = np.linalg.norm(W_init[:, c])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                U_init[:, c] /= norm_u
                V_init[:, c] /= norm_v
                W_init[:, c] /= norm_w
                U_init[:, c] *= scale
                V_init[:, c] *= scale
                W_init[:, c] *= scale
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build rank-55 decomposition
    U, V, W, rank = _build_rank55()
    
    # Verify
    if not _verify(U, V, W, rank):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col_s] = 1
                    V_school[j * p + k][col_s] = 1
                    W_school[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to reduce rank
    if time.time() < deadline - 20:
        U, V, W, rank = _try_reduce_rank(U, V, W, rank, deadline)
    
    if time.time() < deadline - 10:
        U, V, W, rank = _try_als_reduction(U, V, W, rank, deadline)
    
    # Final verification
    if not _verify(U, V, W, rank):
        U, V, W, rank = _build_rank55()
    
    return {"U": U, "V": V, "W": W}
