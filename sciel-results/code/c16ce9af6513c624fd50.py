import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[0,2]=-1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[1,3]=-1; W[3,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=-1; W[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=-1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W

def _build_4x4x4_49():
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)

    def offset(b):
        return {0:0, 1:2, 2:8, 3:10}.get(b, 0)

    A_blocks = [
        [(0,1),(3,1)],[(2,1),(3,1)],[(0,1)],[(3,1)],
        [(0,1),(1,1)],[(2,1),(0,-1)],[(1,1),(3,-1)]
    ]
    B_blocks = [
        [(0,1),(3,1)],[(0,1)],[(2,1),(3,-1)],[(1,1),(0,-1)],
        [(3,1)],[(0,1),(1,1)],[(2,1),(3,1)]
    ]
    C_blocks = [
        [(0,1),(3,1),(1,-1),(2,1)],[(2,1),(3,-1)],[(0,-1),(1,1)],
        [(1,-1),(3,1)],[(0,1),(1,-1),(3,1)],[(2,-1),(3,1)],[(0,1),(2,-1)]
    ]

    for t in range(7):
        for s in range(7):
            col = t*7+s
            for i_local in range(4):
                val = U2[i_local, s]
                if val != 0:
                    for bA, sA in A_blocks[t]:
                        U4[offset(bA)+i_local, col] += val*sA
            for j_local in range(4):
                val = V2[j_local, s]
                if val != 0:
                    for bB, sB in B_blocks[t]:
                        V4[offset(bB)+j_local, col] += val*sB
            for k_local in range(4):
                val = W2[k_local, s]
                if val != 0:
                    for bC, sC in C_blocks[t]:
                        W4[offset(bC)+k_local, col] += val*sC
    return U4, V4, W4

def _build_4x4x8_98():
    U4, V4, W4 = _build_4x4x4_49()
    U = np.zeros((16, 98), dtype=np.int64)
    V = np.zeros((32, 98), dtype=np.int64)
    W = np.zeros((32, 98), dtype=np.int64)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    U[:, 49:98] = U4
    for j in range(4):
        for kl in range(4):
            V[j*8+kl+4, 49:98] = V4[j*4+kl, 0:49]
    for kl in range(4):
        for i in range(4):
            W[(kl+4)*4+i, 49:98] = W4[kl*4+i, 0:49]
    return U, V, W

def _als_reduce(U0, V0, W0, T, max_iter=200, tol=1e-10, deadline=None):
    """
    Alternating Least Squares to reduce rank.
    Start with U0, V0, W0 (rank R). Try to fit T with rank R' < R by
    projecting and zeroing out small singular values in the factor matrices.
    """
    n, m, p = DIMENSIONS
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    R = U0.shape[1]
    
    U = U0.astype(np.float64).copy()
    V = V0.astype(np.float64).copy()
    W = W0.astype(np.float64).copy()
    
    # Flatten T for efficient operations
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed r: T[:, :, :] contribution is U[:,r] outer V[:,r] outer W[:,r]
    
    def compute_residual_sq():
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_rec - T)**2)
    
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try reducing rank step by step
    for target_rank in range(R-1, max(20, R-20), -1):
        if deadline and time.time() > deadline - 15:
            break
        
        # Initialize with current factors, truncate to target_rank
        U_t = U[:, :target_rank].copy()
        V_t = V[:, :target_rank].copy()
        W_t = W[:, :target_rank].copy()
        
        # ALS iterations
        for it in range(max_iter):
            if deadline and time.time() > deadline - 10:
                break
            
            # Update U: for each a, T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # Let M_a = T[a,:,:].reshape(dim_b, dim_c)
            # M_a = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # This is a linear system in U[a, :] for each a.
            # But we can vectorize: 
            # T_flat[a, b*dim_c+c] = sum_r U[a,r] * V[b,r] * W[c,r]
            
            # Build the design matrix for U update
            # For each (b,c): T[:, b, c] = sum_r U[:, r] * V[b,r] * W[c,r]
            # Let D_U[b*c, r] = V[b,r] * W[c,r]
            # Then T_flat[a, b*c] = sum_r U[a,r] * D_U[b*c, r]
            # So T_flat = U @ D_U^T
            
            # D_U has shape (dim_b*dim_c, R)
            D_U = np.outer(V.flatten(), W.flatten()).reshape(dim_b, dim_c, R)
            D_U = D_U.transpose(1, 0, 2).reshape(dim_c, dim_b, R)  # need to be careful
            # Actually: D_U[bc, r] = V[b,r]*W[c,r]
            D_U = np.einsum('br,cr->bcr', V, W).reshape(dim_b*dim_c, R)
            # T_flat[a, bc] = sum_r U[a,r] * D_U[bc, r]
            # So T_flat = U @ D_U.T
            
            try:
                U_t, _, _, _ = np.linalg.lstsq(D_U.T, T.reshape(dim_a, dim_b*dim_c).T, rcond=1e-12)
                U_t = U_t.T
            except:
                pass
            
            # Update V
            D_V = np.einsum('ar,cr->acr', U_t, W_t).reshape(dim_a*dim_c, R)
            # T_flat[ac, b] = sum_r V[b,r] * D_V[ac, r]
            # T_flat[ac, b] = sum_r D_V[ac,r] * V[b,r]
            # So T_flat.T = V @ D_V.T
            try:
                V_t, _, _, _ = np.linalg.lstsq(D_V.T, T.transpose(1,0,2).reshape(dim_b, dim_a*dim_c).T, rcond=1e-12)
                V_t = V_t.T
            except:
                pass
            
            # Update W
            D_W = np.einsum('ar,br->abr', U_t, V_t).reshape(dim_a*dim_b, R)
            # T_flat[ab, c] = sum_r W[c,r] * D_W[ab, r]
            try:
                W_t, _, _, _ = np.linalg.lstsq(D_W.T, T.transpose(2,0,1).reshape(dim_c, dim_a*dim_b).T, rcond=1e-12)
                W_t = W_t.T
            except:
                pass
        
        # Check if this target_rank works
        T_rec = np.einsum('ar,br,cr->abc', U_t, V_t, W_t)
        err = np.max(np.abs(T_rec - T))
        if err < 1e-6:
            best_rank = target_rank
            best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
            # Update U, V, W for next iteration
            U = U_t
            V = V_t
            W = W_t
    
    return best_U, best_V, best_W, best_rank

def _try_svd_reduction(U, V, W, T, deadline=None):
    """
    Try to reduce rank by finding linear dependencies in the factor matrices
    and projecting out null directions.
    """
    n, m, p = DIMENSIONS
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    R = U.shape[1]
    
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    
    # Find columns of U that are linearly dependent
    # Use SVD to find rank
    U_sv, U_s, U_vt = np.linalg.svd(U, full_matrices=False)
    V_sv, V_s, V_vt = np.linalg.svd(V, full_matrices=False)
    W_sv, W_s, W_vt = np.linalg.svd(W, full_matrices=False)
    
    # The effective rank is the minimum of the three ranks
    # But we need to be careful: the decomposition must still be valid
    
    # Try to find a lower-rank approximation
    for target_rank in range(R-1, max(20, R-15), -1):
        if deadline and time.time() > deadline - 10:
            break
        
        U_t = U_sv[:, :target_rank] * U_s[:target_rank] * U_vt[:target_rank, :]
        V_t = V_sv[:, :target_rank] * V_s[:target_rank] * V_vt[:target_rank, :]
        W_t = W_sv[:, :target_rank] * W_s[:target_rank] * W_vt[:target_rank, :]
        
        T_rec = np.einsum('ar,br,cr->abc', U_t, V_t, W_t)
        err = np.max(np.abs(T_rec - T))
        if err < 1e-6:
            return U_t, V_t, W_t, target_rank
    
    return U, V, W, R

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    dim_a, dim_b, dim_c = n*m, m*p, p*n

    # Build the full 4x4x8 multiplication tensor
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Fallback: Schoolbook (rank 128)
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_a, rank_schoolbook), dtype=np.int64)
    V_school = np.zeros((dim_b, rank_schoolbook), dtype=np.int64)
    W_school = np.zeros((dim_c, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook

    # Try 98-rank constructive solution
    if time.time() < deadline - 30:
        try:
            U98, V98, W98 = _build_4x4x8_98()
            T_recon = np.einsum('ar,br,cr->abc', U98, V98, W98)
            if np.allclose(T_recon, T, atol=1e-9):
                best_U, best_V, best_W = U98.astype(np.float64), V98.astype(np.float64), W98.astype(np.float64)
                best_rank = 98
        except Exception:
            pass

    # Try ALS rank reduction
    if best_rank == 98 and time.time() < deadline - 20:
        try:
            U_als, V_als, W_als, rank_als = _als_reduce(
                best_U, best_V, best_W, T,
                max_iter=100, deadline=deadline
            )
            if rank_als < best_rank:
                T_recon = np.einsum('ar,br,cr->abc', U_als, V_als, W_als)
                if np.allclose(T_recon, T, atol=1e-6):
                    best_U, best_V, best_W = U_als, V_als, W_als
                    best_rank = rank_als
        except Exception:
            pass

    # Try SVD-based reduction
    if time.time() < deadline - 10:
        try:
            U_svd, V_svd, W_svd, rank_svd = _try_svd_reduction(
                best_U, best_V, best_W, T, deadline=deadline
            )
            if rank_svd < best_rank:
                T_recon = np.einsum('ar,br,cr->abc', U_svd, V_svd, W_svd)
                if np.allclose(T_recon, T, atol=1e-6):
                    best_U, best_V, best_W = U_svd, V_svd, W_svd
                    best_rank = rank_svd
        except Exception:
            pass

    def to_half_integer(mat):
        mat = np.asarray(mat, dtype=np.float64)
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
