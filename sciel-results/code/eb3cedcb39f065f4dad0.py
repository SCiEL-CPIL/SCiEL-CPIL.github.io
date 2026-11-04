import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _build_strassen_decomp():
    """Build the rank-55 Strassen-based decomposition as numpy arrays."""
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    W_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j, r] = val
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k), r] = val
            for local_idx, val in W_coeffs[s].items():
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i, r] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2, r] = 1
            V[2 * p + k, r] = 1
            W[k * n + i, r] = 1
            col += 1
    
    return U, V, W

def _tensor_to_matrix(U, V, W):
    """Convert factor matrices to the tensor in matrix form (p*n) x (n*m) x (m*p) flattened."""
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to verify: for all i,j,k, T[i,j,k] = delta(i,j,k) where the tensor is 1 iff the multiplication is correct.
    # Actually the tensor for matmul is: T[i,j,k] = 1 if (i,j,k) corresponds to a valid triple.
    # The matmul tensor: C_ik = sum_j A_ij B_jk
    # So T[i,j,k] = 1 for all valid (i,j,k) in the matmul tensor.
    # The decomposition satisfies: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = T[i,j,k]
    # where W is indexed as W[k*n+i, r] which is W[i,k,r] in (i,k) ordering.
    
    # Let's build the tensor as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # The matmul tensor has all 1s
    
    # Verify decomposition
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal T[i,j,k]
    return T

def _check_decomposition(U, V, W):
    """Check if the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal 1 for all (i,j,k)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :])
                # Wait, that's not right. The tensor decomposition is:
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This is a trilinear form, not a bilinear one.
                # Let me compute it properly.
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _check_decomposition_fast(U, V, W):
    """Fast check using einsum."""
    n, m, p = DIMENSIONS
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # U[i*m+j, r] -> reshape to (n, m, r)
    # V[j*p+k, r] -> reshape to (m, p, r)
    # W[k*n+i, r] -> W is indexed as W[k*n+i], so reshape to (p, n, r)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    max_err = np.max(np.abs(T - 1.0))
    return max_err

def _als_from_init(U_init, V_init, W_init, target_rank, max_iters=30, tol=1e-10, deadline=None):
    """ALS starting from truncated initial decomposition."""
    n, m, p = DIMENSIONS
    
    # Build target tensor
    U3_init = U_init.reshape(n, m, -1)
    V3_init = V_init.reshape(m, p, -1)
    W3_init = W_init.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3_init, V3_init, W3_init)
    
    # Initialize by keeping first target_rank columns
    Ur = U_init[:, :target_rank].copy()
    Vr = V_init[:, :target_rank].copy()
    Wr = W_init[:, :target_rank].copy()
    
    best_err = float('inf')
    best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        M = np.einsum('jkr,kir->ijk', Vr.reshape(m, p, target_rank), Wr.reshape(p, n, target_rank))
        U_new = np.zeros((n * m, target_rank))
        for i in range(n):
            for j in range(m):
                Mij = M[i, j, :]
                Tij = T[i, j, :]
                try:
                    u, _, _, _ = np.linalg.lstsq(Mij, Tij, rcond=None)
                except:
                    u = np.zeros(target_rank)
                U_new[i * m + j, :] = u
        
        # Update V
        N = np.einsum('ijr,kir->jki', Ur.reshape(n, m, target_rank), Wr.reshape(p, n, target_rank))
        V_new = np.zeros((m * p, target_rank))
        for j in range(m):
            for k in range(p):
                Njk = N[j, k, :]
                Tjk = T[:, j, k]
                try:
                    v, _, _, _ = np.linalg.lstsq(Njk, Tjk, rcond=None)
                except:
                    v = np.zeros(target_rank)
                V_new[j * p + k, :] = v
        
        # Update W
        P = np.einsum('ijr,jkr->kij', Ur.reshape(n, m, target_rank), Vr.reshape(m, p, target_rank))
        W_new = np.zeros((p * n, target_rank))
        for k in range(p):
            for i in range(n):
                Pki = P[k, i, :]
                Tki = T[i, :, k]
                try:
                    w, _, _, _ = np.linalg.lstsq(Pki, Tki, rcond=None)
                except:
                    w = np.zeros(target_rank)
                W_new[k * n + i, :] = w
        
        Ur, Vr, Wr = U_new, V_new, W_new
        
        U3 = Ur.reshape(n, m, target_rank)
        V3 = Vr.reshape(m, p, target_rank)
        W3 = Wr.reshape(p, n, target_rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T_recon - T))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _round_to_halfint(arr):
    """Round to nearest half-integer and check if it still works."""
    rounded = np.round(arr * 2) / 2
    return rounded

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the initial rank-55 decomposition
    U, V, W = _build_strassen_decomp()
    
    # Verify it's correct
    err = _check_decomposition_fast(U, V, W)
    if err > 1e-6:
        # Fallback: return the original
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 55
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try to reduce rank using ALS with warm start from rank-55
    # Try a few specific lower ranks rather than sweeping all
    for target_rank in [50, 45, 40, 35, 30]:
        if time.time() > deadline - 5:
            break
        
        Ur, Vr, Wr, err = _als_from_init(U, V, W, target_rank, max_iters=30, tol=1e-10, deadline=deadline)
        
        if Ur is not None and err < 1e-8:
            # Try to round to half-integers
            Ur_r = _round_to_halfint(Ur)
            Vr_r = _round_to_halfint(Vr)
            Wr_r = _round_to_halfint(Wr)
            
            # Check if rounded version still works
            err_rounded = _check_decomposition_fast(Ur_r, Vr_r, Wr_r)
            if err_rounded < 1e-6:
                best_rank = target_rank
                best_result = (Ur_r, Vr_r, Wr_r)
            else:
                best_rank = target_rank
                best_result = (Ur, Vr, Wr)
            break
    
    U_best, V_best, W_best = best_result
    
    # Convert to lists
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
