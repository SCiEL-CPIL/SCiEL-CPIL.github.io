"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np

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

def _verify_decomp(n, m, p, U, V, W):
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
    return np.allclose(T_comp, T_target, atol=1e-9)

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
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    W convention: W[k*n+i] where n=2, so W[k*2+i] for C[i,k]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
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
    Construct a block decomposition for 3x5x6 with rank 84.
    Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    """
    n, m, p = 3, 5, 6
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_total = np.zeros((n * m, 0), dtype=np.float64)
    V_total = np.zeros((m * p, 0), dtype=np.float64)
    W_total = np.zeros((p * n, 0), dtype=np.float64)
    
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
                
                if U_total.shape[1] == 0:
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

def _als_polish(U, V, W, T, max_iter=50, tol=1e-10):
    """
    Perform ALS iterations to polish a decomposition.
    U: (nm, r), V: (mp, r), W: (pn, r)
    T: (nm, mp, pn)
    """
    nm, mp, pn = T.shape
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_err = float('inf')
    for it in range(max_iter):
        # Update U
        Z = np.einsum('bk,ck->bck', V, W, optimize=True)
        Z_flat = Z.reshape(mp * pn, r)
        T_flat = T.reshape(nm, mp * pn)
        try:
            U = np.linalg.lstsq(Z_flat, T_flat.T, rcond=None)[0].T
        except Exception:
            pass
        
        # Update V
        Z = np.einsum('ak,ck->ack', U, W, optimize=True)
        Z_flat = Z.reshape(nm * pn, r)
        T_flat = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        try:
            V = np.linalg.lstsq(Z_flat, T_flat.T, rcond=None)[0].T
        except Exception:
            pass
        
        # Update W
        Z = np.einsum('ak,bk->abk', U, V, optimize=True)
        Z_flat = Z.reshape(nm * mp, r)
        T_flat = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        try:
            W = np.linalg.lstsq(Z_flat, T_flat.T, rcond=None)[0].T
        except Exception:
            pass
        
        T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
        err = np.linalg.norm(T_comp - T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using ALS search.
    Uses vectorized ALS updates for speed, then attempts rank reduction
    by zeroing columns and re-solving.
    """
    import numpy as np
    import time

    n, m, p = 3, 5, 6
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    T = _build_target_tensor(n, m, p)

    # Start from block decomposition
    U0, V0, W0 = _block_decomposition_356()
    r0 = U0.shape[1]  # 84

    best_result = None
    best_rank = r0

    # Try to reduce rank by dropping columns and re-solving
    for target_rank in range(r0 - 1, max(r0 - 8, 75), -1):
        if time.time() > deadline - 10:
            break

        # Initialize from the top target_rank columns of the block decomposition
        # But first, let's try to get a better initialization via SVD
        T_mat = T.reshape(nm, mp * pn)
        try:
            U_svd, s, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
            # Use top target_rank singular vectors for U
            U = U_svd[:, :target_rank].copy()
            # For V and W, use random initialization scaled
            np.random.seed(42 + target_rank)
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
        except Exception:
            U = U0[:, :target_rank].copy()
            V = V0[:, :target_rank].copy()
            W = W0[:, :target_rank].copy()

        # Run ALS iterations with vectorized updates
        max_iter = 30
        tol = 1e-8
        prev_err = float('inf')

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Update U: Z_flat[b*pn+c, k] = V[b,k] * W[c,k]
            # Vectorized: Z_flat = np.einsum('bk,ck->bck', V, W).reshape(mp*pn, target_rank)
            Z_flat = np.einsum('bk,ck->bck', V, W, optimize=True).reshape(mp * pn, target_rank)
            # U[a,:] = lstsq(Z_flat, T[a,:,:].flatten())
            # Vectorized: U = lstsq(Z_flat, T.reshape(nm, -1).T).T
            T_flat = T.reshape(nm, mp * pn)
            try:
                U = np.linalg.lstsq(Z_flat, T_flat.T, rcond=None)[0].T
            except Exception:
                pass

            # Update V: Z_flat_V[a*pn+c, k] = U[a,k] * W[c,k]
            Z_flat_V = np.einsum('ak,ck->ack', U, W, optimize=True).reshape(nm * pn, target_rank)
            T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V = np.linalg.lstsq(Z_flat_V, T_flat_V.T, rcond=None)[0].T
            except Exception:
                pass

            # Update W: Z_flat_W[a*mp+b, k] = U[a,k] * V[b,k]
            Z_flat_W = np.einsum('ak,bk->abk', U, V, optimize=True).reshape(nm * mp, target_rank)
            T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W = np.linalg.lstsq(Z_flat_W, T_flat_W.T, rcond=None)[0].T
            except Exception:
                pass

            # Compute error
            T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
            err = np.linalg.norm(T_comp - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err

        # Verify the result
        if _verify_decomp(n, m, p, U, V, W):
            if target_rank < best_rank:
                best_rank = target_rank
                best_result = (U.copy(), V.copy(), W.copy())
                break

    if best_result is not None:
        return best_result
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 140s budget, 10s safety margin
    
    # Try block decomposition
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
