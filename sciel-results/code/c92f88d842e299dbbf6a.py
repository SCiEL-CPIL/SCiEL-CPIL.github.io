"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Build the matrix multiplication tensor T with indices (i, j, k)
    # T[i, j, k] = 1 if the (i,k) entry of A@B uses A[i,j]*B[j,k]
    # In the task convention: W[k*n+i, r], so output index is (k*n+i)
    # U[i*m+j, r], V[j*p+k, r]
    # Tensor T[i, j, k] = 1 for all valid i,j,k (since it's exact multiplication)
    
    # T is a 3-way tensor of size (n*m) x (m*p) x (p*n) where
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    
    # For CP decomposition of rank r:
    # T[a, b, c] = sum_{t=1}^{r} U[a,t] * V[b,t] * W[c,t]
    
    # The schoolbook has rank n*m*p = 24
    # We want to find minimal rank.
    
    # For 2x3 * 3x4, let's try ranks from 14 to 24
    # The tensor has shape (6, 12, 8)
    
    U_shape = (n * m,)  # 6
    V_shape = (m * p,)  # 12
    W_shape = (p * n,)  # 8
    
    # Build the tensor as a flat list of non-zero entries
    # T[i*m+j, j*p+k, k*n+i] = 1
    nz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz.append((a, b, c, 1.0))
    
    # ALS for CP decomposition
    def als_cp(nz, rank, n_iters=200, n_restarts=10, seed=0):
        """Run ALS for CP decomposition, return (U, V, W, error)"""
        U_dims = n * m
        V_dims = m * p
        W_dims = p * n
        best_err = float('inf')
        best_UVW = None
        rng = random.Random(seed)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            # Random initialization
            U = rng.gauss(0, 1, (U_dims, rank))
            V = rng.gauss(0, 1, (V_dims, rank))
            W = rng.gauss(0, 1, (W_dims, rank))
            
            err = float('inf')
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                # Update U: for each a, solve least squares
                # T[a, b, c] = sum_t U[a,t] * (V[b,t] * W[c,t])
                # For fixed a: T[a, :, :] = U[a, :] * (V * W)^T reshaped
                # Let M[t, (b,c)] = V[b,t] * W[c,t]
                # Then T[a, b, c] = sum_t U[a,t] * M[t, b, c]
                
                # Build M = V * W (element-wise outer product per rank)
                # M[t, b, c] = V[b, t] * W[c, t]
                M = np.einsum('bt,ct->btc', V, W)  # (rank, V_dims, W_dims)
                M_flat = M.reshape(rank, V_dims * W_dims)  # (rank, 96)
                
                # For each a, we have observations T[a, b, c] for all (b,c) in nz
                # Solve: T[a, bc] = sum_t U[a,t] * M_flat[t, bc]
                # This is a linear system. But we only know T at certain (a,b,c).
                # For ALS, we use all entries (including zeros).
                
                # Actually, let's use the full tensor approach.
                # Build full T matrix: T[a, b, c] for all a,b,c
                # T is sparse: only 24 non-zero entries, all = 1.
                
                # For updating U: for each a, gather all (b,c) pairs where T[a,b,c] is known
                # But T is zero everywhere except the 24 entries.
                # So for a given a, we have some non-zero entries and many zeros.
                
                # Standard ALS for sparse tensors is tricky. Let's use dense approach.
                # Build full T as (U_dims, V_dims, W_dims)
                
                # For U update:
                # For each a: minimize sum_{b,c} (T[a,b,c] - sum_t U[a,t]*V[b,t]*W[c,t])^2
                # This is a linear least squares: U[a,:] @ M_flat^T = T[a,:,:].reshape(-1)
                
                # Build T_a for each a
                # Pre-compute T as dense array
                T_dense = np.zeros((U_dims, V_dims, W_dims))
                for (a, b, c, val) in nz:
                    T_dense[a, b, c] = val
                
                # U update
                new_U = np.zeros((U_dims, rank))
                for a in range(U_dims):
                    T_a = T_dense[a].reshape(-1)  # (V_dims * W_dims,)
                    # M_flat is (rank, V_dims * W_dims)
                    # Solve: U[a,t] such that M_flat @ U[a,:] = T_a
                    # i.e., (M_flat^T @ M_flat) @ U[a,:] = M_flat^T @ T_a
                    try:
                        new_U[a] = np.linalg.lstsq(M_flat, T_a, rcond=None)[0]
                    except:
                        new_U[a] = U[a]
                U = new_U
                
                # V update
                M2 = np.einsum('at,ct->act', U, W)  # (rank, U_dims, W_dims)
                M2_flat = M2.reshape(rank, U_dims * W_dims)
                new_V = np.zeros((V_dims, rank))
                for b in range(V_dims):
                    T_b = T_dense[:, b, :].reshape(-1)  # (U_dims * W_dims,)
                    try:
                        new_V[b] = np.linalg.lstsq(M2_flat, T_b, rcond=None)[0]
                    except:
                        new_V[b] = V[b]
                V = new_V
                
                # W update
                M3 = np.einsum('at,bt->abt', U, V)  # (rank, U_dims, V_dims)
                M3_flat = M3.reshape(rank, U_dims * V_dims)
                new_W = np.zeros((W_dims, rank))
                for c in range(W_dims):
                    T_c = T_dense[:, :, c].reshape(-1)  # (U_dims * V_dims,)
                    try:
                        new_W[c] = np.linalg.lstsq(M3_flat, T_c, rcond=None)[0]
                    except:
                        new_W[c] = W[c]
                W = new_W
                
                # Compute error
                # Reconstruct T and compute Frobenius error
                T_rec = np.einsum('at,bt,ct->abc', U, V, W)
                err = np.linalg.norm(T_rec - T_dense)
                if err < 1e-8:
                    break
                
                # Check time
                if time.time() > deadline:
                    break
            
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-8:
                break
        
        return best_UVW, best_err
    
    # Try ranks from 14 to 24
    best_rank = n * m * p  # 24, schoolbook
    best_result = None
    
    for r in range(14, n * m * p + 1):
        if time.time() > deadline:
            break
        if r >= best_rank and best_result is not None:
            break
        # Try to find a decomposition of rank r
        try:
            UVW, err = als_cp(nz, r, n_iters=100, n_restarts=5, seed=r * 1000)
            if err < 1e-8 and UVW is not None:
                # Snap to rationals
                U, V, W = UVW
                # Try to snap to simple fractions
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                
                # Verify
                T_dense = np.zeros((n*m, m*p, p*n))
                for (a, b, c, val) in nz:
                    T_dense[a, b, c] = val
                T_rec = np.einsum('at,bt,ct->abc', U_snap, V_snap, W_snap)
                if np.linalg.norm(T_rec - T_dense) < 1e-10:
                    # Convert to list of lists with exact fractions
                    def to_exact(mat):
                        result = []
                        for row in mat:
                            rrow = []
                            for val in row:
                                # Check if close to integer or half-integer
                                v2 = round(val * 2)
                                if abs(val * 2 - v2) < 1e-9:
                                    rrow.append(v2 / 2.0)
                                else:
                                    rrow.append(val)
                            result.append(rrow)
                        return result
                    U_list = to_exact(U_snap)
                    V_list = to_exact(V_snap)
                    W_list = to_exact(W_snap)
                    best_rank = r
                    best_result = {'U': U_list, 'V': V_list, 'W': W_list}
                    break
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
