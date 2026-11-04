import time
import numpy as np

_DIMENSIONS = [4, 4, 8]
_TIME_LIMIT = 140.0

def _verify_decomposition(U, V, W, n, m, p):
    R = U.shape[1]
    tol = 1e-9
    # Vectorized verification for speed
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    # Reshape to (n, m, R), (m, p, R), (p, n, R)
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    # We want to check if the tensor T_ijk = sum_r U_ijk_r V_jkr W_kir is all 1s.
    # This is a batched operation.
    # Iterate over i, j, k is slow in Python loops.
    # Use einsum or broadcasting.
    
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # This is not a standard einsum because indices are mixed.
    # U: i, j, r
    # V: j, k, r
    # W: k, i, r
    
    # Let's compute for each r, the outer product? No.
    # We can sum over r.
    # Term_r[i, j, k] = U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    # Efficient way:
    # For each r, we have U_r (n,m), V_r (m,p), W_r (p,n).
    # We want sum_r (U_r[i,j] * V_r[j,k] * W_r[k,i])
    
    # This is equivalent to:
    # sum_r ( U_r[i,j] * (V_r * W_r)[j,k]? No.
    # (V_r * W_r) is not defined directly.
    
    # Let's use einsum:
    # 'ijr,jkr,kir->ijk'
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True
    except Exception:
        # Fallback to loops if einsum fails or is too slow/memory heavy
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

def _strassen_222():
    """Strassen's algorithm for 2x2x2 matrix multiplication, rank 7."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    """Kronecker product decomposition."""
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape
    
    R = r1 * r2
    U = np.zeros((n1 * n2, R), dtype=float)
    V = np.zeros((m1 * m2, R), dtype=float)
    W = np.zeros((p1 * p2, R), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            U[:, r_idx] = np.kron(U1[:, r1_idx], U2[:, r2_idx])
            V[:, r_idx] = np.kron(V1[:, r1_idx], V2[:, r2_idx])
            W[:, r_idx] = np.kron(W1[:, r1_idx], W2[:, r2_idx])
            
    return U, V, W

def _direct_sum_decomposition(U1, V1, W1):
    """Direct sum: doubles the problem size in m and p."""
    n, r = U1.shape
    m, _ = V1.shape
    p, _ = W1.shape
    
    R = 2 * r
    U = np.zeros((n, R), dtype=float)
    V = np.zeros((2 * m, R), dtype=float)
    W = np.zeros((2 * p, R), dtype=float)
    
    U[:, :r] = U1
    V[:m, :r] = V1
    W[:p, :r] = W1
    
    U[:, r:] = U1
    V[m:, r:] = V1
    W[p:, r:] = W1
    
    return U, V, W

def _greedy_remove_columns(U, V, W, n, m, p, time_limit, pair_mode=False):
    """
    Greedily remove columns (single or pairs) that allow the decomposition to remain valid.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    start = time.time()
    
    changed = True
    while changed:
        changed = False
        current_rank = U.shape[1]
        if current_rank <= 2:
            break
        if time.time() - start > time_limit:
            break
            
        if not pair_mode:
            # Single column removal
            for col in range(current_rank):
                if time.time() - start > time_limit:
                    break
                mask = np.ones(current_rank, dtype=bool)
                mask[col] = False
                
                U_try = U[:, mask]
                V_try = V[:, mask]
                W_try = W[:, mask]
                
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    U, V, W = U_try, V_try, W_try
                    changed = True
                    break
        else:
            # Pairwise column removal
            # To avoid O(R^2) full verification which might be slow if R is large,
            # we can be smart, but R is small (~98).
            # 98*97/2 = 4753 pairs.
            # Verification is fast.
            for col1 in range(current_rank):
                if time.time() - start > time_limit:
                    break
                for col2 in range(col1 + 1, current_rank):
                    if time.time() - start > time_limit:
                        break
                    mask = np.ones(current_rank, dtype=bool)
                    mask[col1] = False
                    mask[col2] = False
                    
                    U_try = U[:, mask]
                    V_try = V[:, mask]
                    W_try = W[:, mask]
                    
                    if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                        U, V, W = U_try, V_try, W_try
                        changed = True
                        break
                if changed:
                    break
                    
    return U, V, W

def construct():
    n, m, p = _DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Strategy 1: Strassen-based structured decomposition (rank 98)
    U2, V2, W2 = _strassen_222()
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    U, V, W = _direct_sum_decomposition(U444, V444, W444)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        R = n * m * p
        U_s = np.zeros((n * m, R), dtype=float)
        V_s = np.zeros((m * p, R), dtype=float)
        W_s = np.zeros((p * n, R), dtype=float)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_s[ii * m + jj, col] = 1
                    V_s[jj * p + kk, col] = 1
                    W_s[kk * n + ii, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try greedy single column removal
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 10:
        U_try, V_try, W_try = _greedy_remove_columns(U, V, W, n, m, p, remaining_time, pair_mode=False)
        if _verify_decomposition(U_try, V_try, W_try, n, m, p):
            if U_try.shape[1] < best_rank:
                best_rank = U_try.shape[1]
                best_U, best_V, best_W = U_try, V_try, W_try
    
    # Try greedy pairwise column removal on the best result so far
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 10:
        U_try, V_try, W_try = _greedy_remove_columns(best_U, best_V, best_W, n, m, p, remaining_time, pair_mode=True)
        if _verify_decomposition(U_try, V_try, W_try, n, m, p):
            if U_try.shape[1] < best_rank:
                best_rank = U_try.shape[1]
                best_U, best_V, best_W = U_try, V_try, W_try
                
    # If still high, try random restarts with greedy removal?
    # If we have time, we can shuffle columns and try again.
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 20 and best_rank > 96:
        for _ in range(5): # 5 random restarts
            if time.time() > deadline - 5:
                break
            U_shuf = best_U[:, np.random.permutation(best_rank)]
            V_shuf = best_V[:, np.random.permutation(best_rank)]
            W_shuf = best_W[:, np.random.permutation(best_rank)]
            
            # Note: Shuffling columns independently breaks the decomposition!
            # We must shuffle the SAME permutation for U, V, W.
            perm = np.random.permutation(best_rank)
            U_shuf = best_U[:, perm]
            V_shuf = best_V[:, perm]
            W_shuf = best_W[:, perm]
            
            U_try, V_try, W_try = _greedy_remove_columns(U_shuf, V_shuf, W_shuf, n, m, p, 10.0, pair_mode=False)
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                if U_try.shape[1] < best_rank:
                    best_rank = U_try.shape[1]
                    best_U, best_V, best_W = U_try, V_try, W_try
                    # Try pairwise on the new best
                    remaining_time = deadline - time.time() - 5.0
                    if remaining_time > 10:
                        U_try2, V_try2, W_try2 = _greedy_remove_columns(best_U, best_V, best_W, n, m, p, remaining_time, pair_mode=True)
                        if _verify_decomposition(U_try2, V_try2, W_try2, n, m, p):
                            if U_try2.shape[1] < best_rank:
                                best_rank = U_try2.shape[1]
                                best_U, best_V, best_W = U_try2, V_try2, W_try2

    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        R = n * m * p
        U_s = np.zeros((n * m, R), dtype=float)
        V_s = np.zeros((m * p, R), dtype=float)
        W_s = np.zeros((p * n, R), dtype=float)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_s[ii * m + jj, col] = 1
                    V_s[jj * p + kk, col] = 1
                    W_s[kk * n + ii, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
