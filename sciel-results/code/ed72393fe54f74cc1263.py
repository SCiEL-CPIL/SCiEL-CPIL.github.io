import time
import numpy as np

_DIMENSIONS = [4, 4, 8]
_TIME_LIMIT = 140.0

def _verify_decomposition(U, V, W, n, m, p):
    R = U.shape[1]
    tol = 1e-9
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _greedy_remove_columns_v2(U, V, W, n, m, p, time_limit):
    """
    Improved greedy column removal:
    1. Score each column by its contribution norm
    2. Try removing from least important first
    3. Use direct verification
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    start = time.time()
    
    changed = True
    while changed:
        changed = False
        current_rank = U.shape[1]
        if current_rank <= 1:
            break
        if time.time() - start > time_limit:
            break
        
        # Score columns by contribution magnitude
        scores = np.zeros(current_rank)
        for r in range(current_rank):
            scores[r] = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
        
        # Try removing columns from least important to most important
        order = np.argsort(scores)
        
        for col in order:
            if time.time() - start > time_limit:
                break
            
            mask = np.ones(current_rank, dtype=bool)
            mask[col] = False
            
            U_try = U[:, mask]
            V_try = V[:, mask]
            W_try = W[:, mask]
            
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                U = U_try
                V = V_try
                W = W_try
                changed = True
                break
    
    return U, V, W

def _randomized_rank_reduction(U, V, W, n, m, p, time_limit):
    """
    Randomized approach: combine greedy removal with random linear
    combinations to find lower-rank equivalent decompositions.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    start = time.time()
    rng = np.random.default_rng(42)
    
    # Phase 1: Greedy removal
    U_g, V_g, W_g = _greedy_remove_columns_v2(U, V, W, n, m, p, min(25.0, time_limit * 0.3))
    best_U, best_V, best_W = U_g, V_g, W_g
    best_rank = U_g.shape[1]
    
    # Phase 2: Random linear combination attempts
    remaining = time_limit - (time.time() - start)
    if remaining < 10:
        return best_U, best_V, best_W
    
    U = best_U.copy()
    V = best_V.copy()
    W = best_W.copy()
    
    max_iters = int(remaining * 8)
    for _ in range(max_iters):
        if time.time() - start > time_limit:
            break
        if U.shape[1] <= 1:
            break
        
        R_cur = U.shape[1]
        r1, r2 = rng.choice(R_cur, size=2, replace=False)
        alpha = rng.uniform(-2, 2)
        
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        U_try[:, r2] = U_try[:, r1] + alpha * U_try[:, r2]
        V_try[:, r2] = V_try[:, r1] + alpha * V_try[:, r2]
        W_try[:, r2] = W_try[:, r1] + alpha * W_try[:, r2]
        
        mask = np.ones(R_cur, dtype=bool)
        mask[r1] = False
        U_try2 = U_try[:, mask]
        V_try2 = V_try[:, mask]
        W_try2 = W_try[:, mask]
        
        if _verify_decomposition(U_try2, V_try2, W_try2, n, m, p):
            U, V, W = U_try2, V_try2, W_try2
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = U.shape[1]
    
    return best_U, best_V, best_W

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

def _build_2x2x4_decomposition():
    """
    Build a rank-14 decomposition for 2x2x4 matrix multiplication
    (multiply 2x2 by 2x4 to get 2x4) via direct sum of two Strassen 2x2x2.
    """
    U_s, V_s, W_s = _strassen_222()
    U = np.zeros((4, 14), dtype=float)
    V = np.zeros((8, 14), dtype=float)
    W = np.zeros((8, 14), dtype=float)
    
    # First 2x2 block (columns 0-1 of the 2x4 matrix B)
    U[:, :7] = U_s
    V[:4, :7] = V_s
    W[:4, :7] = W_s
    
    # Second 2x2 block (columns 2-3 of the 2x4 matrix B)
    U[:, 7:] = U_s
    V[4:, 7:] = V_s
    W[4:, 7:] = W_s
    
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

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    R = n * m * p
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj, col] = 1
                V[jj * p + kk, col] = 1
                W[kk * n + ii, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = _DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Strategy 1: Kronecker product of 2x2x4 and 2x2x2 (rank 14*7=98)
    U_224, V_224, W_224 = _build_2x2x4_decomposition()
    U_222, V_222, W_222 = _strassen_222()
    U, V, W = _kron_decomposition(U_224, V_224, W_224, U_222, V_222, W_222)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U_s, V_s, W_s = _schoolbook_decomposition(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy 2: Randomized rank reduction on the structured decomposition
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 15:
        U_try, V_try, W_try = _randomized_rank_reduction(U, V, W, n, m, p, remaining_time)
        if _verify_decomposition(U_try, V_try, W_try, n, m, p):
            if U_try.shape[1] < best_rank:
                best_rank = U_try.shape[1]
                best_U, best_V, best_W = U_try, V_try, W_try
    
    # Strategy 3: Schoolbook + greedy removal as alternative starting point
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 15:
        U_s, V_s, W_s = _schoolbook_decomposition(n, m, p)
        U_try, V_try, W_try = _greedy_remove_columns_v2(U_s, V_s, W_s, n, m, p, remaining_time)
        if _verify_decomposition(U_try, V_try, W_try, n, m, p):
            if U_try.shape[1] < best_rank:
                best_rank = U_try.shape[1]
                best_U, best_V, best_W = U_try, V_try, W_try
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        U_s, V_s, W_s = _schoolbook_decomposition(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
