import numpy as np
import time
from itertools import combinations

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

def _check_decomposition_fast(U, V, W):
    """Fast check using einsum."""
    n, m, p = DIMENSIONS
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    max_err = np.max(np.abs(T - 1.0))
    return max_err

def _round_to_halfint(arr):
    """Round to nearest half-integer."""
    rounded = np.round(arr * 2) / 2
    return rounded

def _build_block_decomp(n, m, p, split_m1, split_m2, split_p1, split_p2):
    """
    Build a block-based decomposition.
    Split m into (split_m1, split_m2) and p into (split_p1, split_p2).
    Use Strassen-like for the (2, split_m1) x (split_m1, split_p1) block,
    and schoolbook for the rest.
    """
    # This is a generalization - for now, just use the existing Strassen + schoolbook
    # but with different splits
    rank = 0
    U = []
    V = []
    W = []
    
    # Block 1: (n x split_m1) * (split_m1 x split_p1) using Strassen if possible
    # For simplicity, use schoolbook for all blocks in this helper
    # The main improvement will be in the search strategy
    
    return None, None, None, rank

def _rank_reduction_via_svd(U, V, W, target_rank, deadline):
    """
    Try to reduce rank by finding a lower-rank approximation.
    Uses the fact that for a rank-r decomposition, we can try to find
    a rank-r' decomposition with r' < r by solving a constrained optimization.
    """
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W, 0.0
    
    # Create the tensor from the current decomposition
    U3 = U.reshape(n, m, current_rank)
    V3 = V.reshape(m, p, current_rank)
    W3 = W.reshape(p, n, current_rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    rng = np.random.RandomState(42)
    
    for trial in range(10):
        if time.time() > deadline:
            break
        
        # Initialize with random low-rank factors
        Ur = rng.randn(n * m, target_rank) * 0.1
        Vr = rng.randn(m * p, target_rank) * 0.1
        Wr = rng.randn(p * n, target_rank) * 0.1
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Update U
            M = np.einsum('jkr,kir->ijk', Vr.reshape(m, p, target_rank), Wr.reshape(p, n, target_rank))
            U_new = np.zeros((n * m, target_rank))
            for i in range(n):
                for j in range(m):
                    Mij = M[i, j, :]
                    Tij = T[i, j, :]
                    if np.linalg.matrix_rank(Mij) >= target_rank:
                        u, _, _, _ = np.linalg.lstsq(Mij, Tij, rcond=None)
                    else:
                        u = np.zeros(target_rank)
                    U_new[i * m + j, :] = u
            Ur = U_new
            
            # Update V
            N = np.einsum('ijr,kir->jki', Ur.reshape(n, m, target_rank), Wr.reshape(p, n, target_rank))
            V_new = np.zeros((m * p, target_rank))
            for j in range(m):
                for k in range(p):
                    Njk = N[j, k, :]
                    Tjk = T[:, j, k]
                    if np.linalg.matrix_rank(Njk) >= target_rank:
                        v, _, _, _ = np.linalg.lstsq(Njk, Tjk, rcond=None)
                    else:
                        v = np.zeros(target_rank)
                    V_new[j * p + k, :] = v
            Vr = V_new
            
            # Update W
            P = np.einsum('ijr,jkr->kij', Ur.reshape(n, m, target_rank), Vr.reshape(m, p, target_rank))
            W_new = np.zeros((p * n, target_rank))
            for k in range(p):
                for i in range(n):
                    Pki = P[k, i, :]
                    Tki = T[i, :, k]
                    if np.linalg.matrix_rank(Pki) >= target_rank:
                        w, _, _, _ = np.linalg.lstsq(Pki, Tki, rcond=None)
                    else:
                        w = np.zeros(target_rank)
                    W_new[k * n + i, :] = w
            Wr = W_new
            
            # Check error
            U3 = Ur.reshape(n, m, target_rank)
            V3 = Vr.reshape(m, p, target_rank)
            W3 = Wr.reshape(p, n, target_rank)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            err = np.max(np.abs(T_recon - T))
            
            if err < best_err:
                best_err = err
                best_U = Ur.copy()
                best_V = Vr.copy()
                best_W = Wr.copy()
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10:
        return best_U, best_V, best_W, best_err
    else:
        return None, None, None, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Build the initial rank-55 decomposition
    U, V, W = _build_strassen_decomp()
    
    err = _check_decomposition_fast(U, V, W)
    if err > 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 55
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try to reduce rank using SVD-based approach
    for target_rank in range(54, 29, -1):
        if time.time() > deadline - 10:
            break
        
        Ur, Vr, Wr, err = _rank_reduction_via_svd(U, V, W, target_rank, deadline)
        
        if Ur is not None and err < 1e-10:
            Ur_r = _round_to_halfint(Ur)
            Vr_r = _round_to_halfint(Vr)
            Wr_r = _round_to_halfint(Wr)
            err_rounded = _check_decomposition_fast(Ur_r, Vr_r, Wr_r)
            if err_rounded < 1e-6:
                best_rank = target_rank
                best_result = (Ur_r, Vr_r, Wr_r)
            else:
                best_rank = target_rank
                best_result = (Ur, Vr, Wr)
            break
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
