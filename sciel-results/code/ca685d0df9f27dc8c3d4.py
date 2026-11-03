"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, atol=1e-7):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=atol)

def _schoolbook(n_sub, m_sub, p_sub):
    """Return schoolbook decomposition for n_sub x m_sub x p_sub."""
    rank = n_sub * m_sub * p_sub
    U = np.zeros((n_sub * m_sub, rank), dtype=np.float64)
    V = np.zeros((m_sub * p_sub, rank), dtype=np.float64)
    W = np.zeros((p_sub * n_sub, rank), dtype=np.float64)
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i * m_sub + j, col] = 1.0
                V[j * p_sub + k, col] = 1.0
                W[k * n_sub + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7. Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """
    Embed a sub-decomposition into the full 3x5x6 problem.
    """
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)
    
    for idx in range(n_sub * m_sub):
        i_s = idx // m_sub
        j_s = idx % m_sub
        i_f = i_off + i_s
        j_f = j_off + j_s
        U_full[i_f * m + j_f, :] = U_sub[idx, :]
    
    for idx in range(m_sub * p_sub):
        j_s = idx // p_sub
        k_s = idx % p_sub
        j_f = j_off + j_s
        k_f = k_off + k_s
        V_full[j_f * p + k_f, :] = V_sub[idx, :]
    
    for idx in range(p_sub * n_sub):
        k_s = idx // n_sub
        i_s = idx % n_sub
        k_f = k_off + k_s
        i_f = i_off + i_s
        W_full[k_f * n + i_f, :] = W_sub[idx, :]
    
    return U_full, V_full, W_full

def _block_decomposition_356():
    """
    Construct a block decomposition for 3x5x6.
    Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    Uses Strassen for 2x2x2 blocks, schoolbook for others.
    Total rank: 42+12+24+6 = 84
    """
    n, m, p = 3, 5, 6
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_total = None
    V_total = None
    W_total = None
    
    i_off = 0
    for i_s, n_sub in enumerate(n_splits):
        j_off = 0
        for j_s, m_sub in enumerate(m_splits):
            k_off = 0
            for k_s, p_sub in enumerate(p_splits):
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    U_sub, V_sub, W_sub = _strassen_222()
                else:
                    U_sub, V_sub, W_sub = _schoolbook(n_sub, m_sub, p_sub)
                
                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p
                )
                
                if U_total is None:
                    U_total = U_full
                    V_total = V_full
                    W_total = W_full
                else:
                    U_total = np.hstack([U_total, U_full])
                    V_total = np.hstack([V_total, V_full])
                    W_total = np.hstack([W_total, W_full])
                
                k_off += p_sub
            j_off += m_sub
        i_off += n_sub
    
    return U_total, V_total, W_total

def _als_refine(U, V, W, n, m, p, T, max_iters=50, tol=1e-10, deadline=None):
    """
    Alternating least squares to refine a decomposition.
    Given initial U, V, W of rank r, try to minimize ||T - U·V·W||.
    Returns refined (U, V, W) and final error.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update W: for each (a,b), W[:,k] = (U[:,k] ⊗ V[:,k])^T · T[a,b,:]
        # More efficiently: reshape T to (a*b, c) and solve
        T2 = T.reshape(n*m*m*p, p*n)
        UV = np.einsum('ak,bk->abk', U, V)  # (n*m, m*p, r)
        UV_flat = UV.reshape(n*m*m*p, r)
        # W should satisfy: UV_flat @ W.T = T2
        # W = pinv(UV_flat) @ T2
        W_new = np.linalg.lstsq(UV_flat, T2, rcond=None)[0].T  # (r, p*n)
        W = W_new.T  # (p*n, r)
        
        # Update V: for each (b,c), V[:,k] = (U[:,k] ⊗ W[:,k])^T · T[:,b,c]
        T3 = T.reshape(n*m, m*p*p*n)
        UW = np.einsum('ak,ck->akc', U, W)  # (n*m, r, p*n)
        UW_flat = UW.reshape(n*m*p*n, r)
        # Need: for each b, T[:,b,:].reshape(-1) = UW_flat_b @ V[:,b-related]
        # This is more complex. Let's do it per-b.
        V_new = np.zeros_like(V)
        for b in range(m*p):
            T_b = T[:, b, :].reshape(n*m*p*n)
            # UW_flat is (n*m*p*n, r), but we need it indexed by (a, c) for fixed b
            # Actually UW doesn't depend on b, so we can use the same UW_flat
            V_b = np.linalg.lstsq(UW_flat, T_b, rcond=None)[0]  # (r,)
            V_new[:, b] = V_b
        V = V_new
        
        # Update U: for each (a,c), U[:,k] = (V[:,k] ⊗ W[:,k])^T · T[a,:,c]
        T4 = T.reshape(n*m*m*p, p*n)
        VW = np.einsum('bk,ck->bck', V, W)  # (m*p, r, p*n)
        VW_flat = VW.reshape(m*p*p*n, r)
        U_new = np.zeros_like(U)
        for a in range(n*m):
            T_a = T[a, :, :].reshape(m*p*p*n)
            U_a = np.linalg.lstsq(VW_flat, T_a, rcond=None)[0]
            U_new[a, :] = U_a
        U = U_new
        
        # Check convergence
        T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
        err = np.linalg.norm(T_comp - T)
        if err < tol:
            break
    
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    err = np.linalg.norm(T_comp - T)
    return U, V, W, err

def _random_init(n, m, p, r, seed=42):
    """Generate random initial decomposition of rank r."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, r) * 0.1
    V = rng.randn(m*p, r) * 0.1
    W = rng.randn(p*n, r) * 0.1
    return U, V, W

def _try_als_search(n, m, p, T, start_rank, end_rank, deadline):
    """
    Try ALS from various ranks, starting from end_rank down to start_rank.
    Returns best (U, V, W, rank) found.
    """
    best = None
    best_rank = start_rank
    
    for r in range(end_rank, start_rank - 1, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline:
                break
            
            seed = 42 + r * 100 + trial
            U, V, W = _random_init(n, m, p, r, seed)
            
            # Also try starting from the block decomposition truncated
            if r <= 84:
                U_b, V_b, W_b = _block_decomposition_356()
                # Truncate to rank r by keeping first r columns
                U_trunc = U_b[:, :r]
                V_trunc = V_b[:, :r]
                W_trunc = W_b[:, :r]
                U, V, W = U_trunc, V_trunc, W_trunc
            
            U_r, V_r, W_r, err = _als_refine(U, V, W, n, m, p, T, 
                                              max_iters=30, tol=1e-10, deadline=deadline)
            
            if err < 1e-7:
                if best is None or r < best_rank:
                    best = (U_r, V_r, W_r)
                    best_rank = r
                    break  # Found valid decomposition at this rank, try lower
        
        if best is not None and best_rank < end_rank:
            # Found a valid decomposition, try to go lower
            continue
    
    return best, best_rank

def _try_winograd_333():
    """
    Attempt to use a rank-23 Winograd decomposition for 3x3x3.
    Falls back to schoolbook rank 27 if not verified.
    """
    # For now, return schoolbook as a safe fallback
    return _schoolbook(3, 3, 3)

def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using ALS search.
    """
    n, m, p = 3, 5, 6
    T = _build_target_tensor(n, m, p)
    
    # Get current best from block decomposition
    U_block, V_block, W_block = _block_decomposition_356()
    current_rank = U_block.shape[1]
    
    # Try ALS search for lower rank
    best, best_rank = _try_als_search(n, m, p, T, start_rank=40, 
                                       end_rank=current_rank, deadline=deadline)
    
    if best is not None and best_rank < current_rank:
        return best
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 10s safety margin
    
    # Start with block decomposition
    U, V, W = _block_decomposition_356()
    
    # Verify
    if _verify_decomp(n, m, p, U, V, W):
        rank = U.shape[1]
        # Try to find even better
        better = _try_better_decomposition(deadline)
        if better is not None:
            U_b, V_b, W_b = better
            if _verify_decomp(n, m, p, U_b, V_b, W_b):
                if U_b.shape[1] < rank:
                    U, V, W = U_b, V_b, W_b
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W_s.tolist()
    }
# EVOLVE-BLOCK-END
