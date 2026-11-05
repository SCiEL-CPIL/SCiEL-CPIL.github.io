"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_BUDGET = 140.0  # safety margin under 150s kill

# EVOLVE-BLOCK-START

def _time_left():
    return TIME_BUDGET - (time.time() - START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify U,V,W exactly decompose the matrix multiplication tensor."""
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    rank = U_arr.shape[1]
    if rank != V_arr.shape[1] or rank != W_arr.shape[1]:
        return False
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return bool(np.max(np.abs(T_recon - T_true)) <= tol)

def _verify_half(U, V, W, n, m, p, tol=1e-9):
    """Verify with half-integer (scale by 2) rounding tolerance."""
    U_arr = np.asarray(U, dtype=float) * 2.0
    V_arr = np.asarray(V, dtype=float) * 2.0
    W_arr = np.asarray(W, dtype=float) * 2.0
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return bool(np.max(np.abs(T_recon - T_true)) <= tol)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank(r, n, m, p, max_iters=200, seed=42, patterns=8):
    """Try to find a rank-r decomposition using ALS on the full tensor."""
    if r < max(n, m, p):
        return None
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0

    T_flat = T_true.reshape(n*m, m*p * p*n)

    for pat in range(patterns):
        if _time_left() < 5.0:
            break
        rng = np.random.RandomState(seed + pat * 1000)
        # Initialize with structured random
        U = rng.randn(n*m, r) * 0.3
        V = rng.randn(m*p, r) * 0.3
        W = rng.randn(p*n, r) * 0.3

        # Precompute M matrices (depend only on the other two matrices)
        for it in range(max_iters):
            if _time_left() < 3.0:
                break
            # Update U: U[a,:] @ M_u = T_flat[a,:]  where M_u[b*p*n+c, r] = V[b,r]*W[c,r]
            M_u = np.zeros((m*p * p*n, r))
            for b in range(m*p):
                for c in range(p*n):
                    M_u[b * p*n + c, :] = V[b, :] * W[c, :]
            try:
                U = np.linalg.lstsq(M_u, T_flat, rcond=None)[0]
            except Exception:
                pass

            # Update V: V[b,:] @ M_v = T_flat_v[b,:]  where M_v[a*p*n+c, r] = U[a,r]*W[c,r]
            # T_flat_v[b, a*p*n+c] = T_true[a, b, c]
            T_v = np.transpose(T_true, (1, 0, 2)).reshape(m*p, n*m * p*n)
            M_v = np.zeros((n*m * p*n, r))
            for a in range(n*m):
                for c in range(p*n):
                    M_v[a * p*n + c, :] = U[a, :] * W[c, :]
            try:
                V = np.linalg.lstsq(M_v, T_v, rcond=None)[0]
            except Exception:
                pass

            # Update W: W[c,:] @ M_w = T_flat_w[c,:]  where M_w[a*m*p+b, r] = U[a,r]*V[b,r]
            T_w = np.transpose(T_true, (2, 0, 1)).reshape(p*n, n*m * m*p)
            M_w = np.zeros((n*m * m*p, r))
            for a in range(n*m):
                for b in range(m*p):
                    M_w[a * m*p + b, :] = U[a, :] * V[b, :]
            try:
                W = np.linalg.lstsq(M_w, T_w, rcond=None)[0]
            except Exception:
                pass

            # Check convergence and try rounding
            if it % 10 == 0:
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_recon - T_true))
                if err < 1e-6:
                    for scale in [1, 2, 4, 8, 16]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale
                        if _verify(U_r, V_r, W_r, n, m, p):
                            return (U_r, V_r, W_r)
                    break
    return None

def _tile_3x(U_sub, V_sub, W_sub, n, m, p):
    """
    Given a decomposition of the 4x8 multiplication tensor (for one i block),
    tile it 3 times to get a decomposition of the 3x4x8 tensor.
    
    U_sub: (m, r)  -- for the 4x8 case, m=4
    V_sub: (m*p, r) -- for 4x8, m=4, p=8, so (32, r)
    W_sub: (p, r) -- for 4x8, p=8, so (8, r)
    
    Full U: (n*m, 3r) where U[i*m+j, i*r + c] = U_sub[j, c]
    Full V: (m*p, 3r) where V[j*p+k, i*r + c] = V_sub[j*p+k, c]
    Full W: (p*n, 3r) where W[k*n+i, i*r + c] = W_sub[k, c]
    """
    r = U_sub.shape[1]
    R = 3 * r
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    for i in range(n):
        # U: row i*m+j, col i*r + c
        for j in range(m):
            U[i*m + j, i*r:(i+1)*r] = U_sub[j, :]
        # V: row j*p+k, col i*r + c
        for j in range(m):
            for k in range(p):
                V[j*p + k, i*r:(i+1)*r] = V_sub[j*p + k, :]
        # W: row k*n+i, col i*r + c
        for k in range(p):
            W[k*n + i, i*r:(i+1)*r] = W_sub[k, :]
    return U, V, W

def _try_rank_sub(r, m, p, max_iters=300, seed=42, patterns=12):
    """
    Try to find a rank-r decomposition of the m x p multiplication tensor
    (i.e., multiplying an m x p matrix by... wait, we need 4x8).
    
    Actually, the sub-problem is: decompose the tensor for multiplying
    a 1 x m row vector by an m x p matrix? No.
    
    Let me reconsider. The 3x4x8 tensor: T[i*m+j, j*p+k, k*n+i] = 1.
    For fixed i, the slice is: T[i*m+j, j*p+k, k*n+i] = 1 for all j,k.
    This is the tensor for multiplying a 1 x m matrix by an m x p matrix?
    No, it's the tensor where input1 index is j (size m), input2 index is k (size p),
    output index is (j, k) pair... 
    
    Actually the sub-tensor for fixed i has:
    - First index: j in range(m)  [from i*m+j, the j part]
    - Second index: k in range(p) [from j*p+k, the k part]  
    - Third index: (j,k) pair     [from k*n+i, but i is fixed, so this is j*p+k? No]
    
    Wait, let me re-examine. T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i.
    For fixed i: a = i*m+j (j varies 0..m-1), b = j*p+k (j,k vary), c = k*n+i (k varies 0..p-1).
    So the sub-tensor S_i has dimensions: first = m (index j), second = m*p (index (j,k)), third = p (index k).
    S_i[j, j*p+k, k] = 1.
    
    This is the tensor for: given a matrix A (1 x m) and B (m x p), compute C = A @ B (1 x p).
    The tensor is: C[k] = sum_j A[j] * B[j, k].
    So S_i[j, j*p+k, k] = 1 means: the (j, (j,k), k) entry is 1.
    
    This is the standard 1 x m x p matrix multiplication tensor!
    Its rank is at most m*p (schoolbook) and at least max(m, p).
    
    For m=4, p=8: the tensor is for 1x4 times 4x8 = 1x8.
    Schoolbook rank = 4*8 = 32.
    """
    # Sub-tensor: S[j, j*p+k, k] = 1 for j in 0..m-1, k in 0..p-1
    # Dimensions: (m, m*p, p)
    T_sub = np.zeros((m, m*p, p), dtype=float)
    for j in range(m):
        for k in range(p):
            T_sub[j, j*p + k, k] = 1.0

    T_flat = T_sub.reshape(m, m*p * p)

    for pat in range(patterns):
        if _time_left() < 5.0:
            break
        rng = np.random.RandomState(seed + pat * 1000)
        U = rng.randn(m, r) * 0.3
        V = rng.randn(m*p, r) * 0.3
        W = rng.randn(p, r) * 0.3

        for it in range(max_iters):
            if _time_left() < 3.0:
                break
            # Update U: U[j,:] @ M_u = T_flat[j,:]  where M_u[(j'*p+k)*p + k', r] = V[j'*p+k, r]*W[k', r]
            # Actually M_u index: for fixed j, we sum over (j', k) and k': U[j, r] * V[j'*p+k, r] * W[k', r]
            # The output index is (j'*p+k, k') but we need it to match T_sub[j, j'*p+k, k']
            # So M_u[(j'*p+k)*p + k', r] = V[j'*p+k, r] * W[k', r]
            M_u = np.zeros((m*p * p, r))
            for jp in range(m*p):
                for kp in range(p):
                    M_u[jp * p + kp, :] = V[jp, :] * W[kp, :]
            try:
                U = np.linalg.lstsq(M_u, T_flat, rcond=None)[0]
            except Exception:
                pass

            # Update V: V[jp,:] @ M_v = T_flat_v[jp,:]  where M_v[j*p + kp, r] = U[j, r]*W[kp, r]
            # T_sub[j, jp, kp] where jp = j'*p+k
            T_v = np.transpose(T_sub, (1, 0, 2)).reshape(m*p, m * p)
            M_v = np.zeros((m * p, r))
            for j in range(m):
                for kp in range(p):
                    M_v[j * p + kp, :] = U[j, :] * W[kp, :]
            try:
                V = np.linalg.lstsq(M_v, T_v, rcond=None)[0]
            except Exception:
                pass

            # Update W: W[kp,:] @ M_w = T_flat_w[kp,:]  where M_w[j*m*p + jp, r] = U[j, r]*V[jp, r]
            T_w = np.transpose(T_sub, (2, 0, 1)).reshape(p, m * m*p)
            M_w = np.zeros((m * m*p, r))
            for j in range(m):
                for jp in range(m*p):
                    M_w[j * m*p + jp, :] = U[j, :] * V[jp, :]
            try:
                W = np.linalg.lstsq(M_w, T_w, rcond=None)[0]
            except Exception:
                pass

            # Check convergence and try rounding
            if it % 10 == 0:
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_recon - T_sub))
                if err < 1e-6:
                    for scale in [1, 2, 4, 8, 16, 32]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale
                        T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                        if np.max(np.abs(T_check - T_sub)) < 1e-9:
                            return (U_r, V_r, W_r)
                    break
    return None

def _schoolbook_sub(m, p):
    """Schoolbook decomposition of the 1 x m x p tensor: S[j, j*p+k, k] = 1."""
    rank = m * p
    U = np.zeros((m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p, rank))
    col = 0
    for j in range(m):
        for k in range(p):
            U[j, col] = 1.0
            V[j*p + k, col] = 1.0
            W[k, col] = 1.0
            col += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a 2D list.
    """
    n, m, p = DIMENSIONS

    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_sb, V_sb, W_sb)

    # The 3x4x8 tensor is 3 independent 1x4x8 sub-tensors.
    # Full rank = 3 * (rank of 1x4x8 tensor).
    # Schoolbook for 1x4x8 has rank 32, so full schoolbook = 96.
    
    # Try to find a lower-rank decomposition of the 1x4x8 sub-tensor,
    # then tile it 3x.
    
    # Lower bound for 1x4x8 tensor rank: max(4, 8) = 8.
    # Known: rank of 1xm x p multiplication is at most m*p, and for small cases
    # there may be improvements. Let's try ranks from 8 upward.
    
    # Try ranks for the sub-problem
    # Theoretical lower bound: max(m, p) = 8
    # Let's try a range of ranks
    for r_sub in range(8, 33):  # up to schoolbook 32
        if _time_left() < 10.0:
            break
        result_sub = _try_rank_sub(r_sub, m, p, max_iters=150, seed=42, patterns=6)
        if result_sub is not None:
            U_sub, V_sub, W_sub = result_sub
            # Tile 3x
            U_full, V_full, W_full = _tile_3x(U_sub, V_sub, W_sub, n, m, p)
            # Verify the full decomposition
            if _verify(U_full, V_full, W_full, n, m, p):
                full_rank = 3 * r_sub
                if full_rank < best_rank:
                    best_rank = full_rank
                    best_result = (U_full, V_full, W_full)
                # Found a valid one, break (we want minimal, so continue to try lower)
                # Actually we're going upward, so this is the first (lowest) that works
                break

    # Also try direct full-tensor search with ranks below schoolbook
    # but this is more expensive, so only if time permits
    if _time_left() > 30.0:
        for r_full in range(24, best_rank + 1):  # 24 = 3*8 lower bound
            if _time_left() < 10.0:
                break
            result_full = _try_rank(r_full, n, m, p, max_iters=100, seed=100, patterns=3)
            if result_full is not None:
                if r_full < best_rank:
                    best_rank = r_full
                    best_result = result_full
                break

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
