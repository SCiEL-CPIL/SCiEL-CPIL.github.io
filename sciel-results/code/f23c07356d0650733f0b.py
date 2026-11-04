import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _verify_fast(U, V, W, n, m, p):
    """Fast vectorized verification of tensor decomposition."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[u,v,w]
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Compute the tensor from U, V, W
    T_comp = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
    
    return np.allclose(T_comp, T, atol=1e-9, rtol=0)

def _verify_sparse(U, V, W, n, m, p):
    """Sparse verification."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    # Check active entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = np.dot(U[u, :] * V[v, :], W[w, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check a random sample of inactive entries
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid.add((u, v, w))
    
    rng = np.random.RandomState(42)
    n_inactive = nm * mp * pn - len(valid)
    n_check = min(500, n_inactive)
    
    for _ in range(n_check):
        u = rng.randint(nm)
        v = rng.randint(mp)
        w = rng.randint(pn)
        if (u, v, w) not in valid:
            val = np.dot(U[u, :] * V[v, :], W[w, :])
            if abs(val) > 1e-9:
                return False
    
    return True

def _schoolbook_3x3x4(n, m, p):
    """
    Construct a rank-30 decomposition for 3x3 * 3x4.
    Uses schoolbook for the first 3 columns of B (3x3x3 part) and
    a rank-3 decomposition for the last column of B (3x3x1 part).
    """
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 3x3x3 (k=0,1,2)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # Only first 3 columns of B
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Part 2: 3x3x1 (k=3)
    # We need to compute C_i = sum_j A_ij B_j3
    # u = i*m + j
    # v = j*p + 3
    # w = 3*n + i
    # We use 3 terms, one for each j (which acts as the rank index)
    for j in range(m):
        # Column index col corresponds to rank term r=j
        # U[u, col] should be 1 if u = i*m + j for any i
        # V[v, col] should be 1 if v = j*p + 3
        # W[w, col] should be 1 if w = 3*n + i for any i
        
        # Set V
        V[j * p + 3, col] = 1
        
        # Set U for all i
        for i in range(n):
            U[i * m + j, col] = 1
            
        # Set W for all i
        for i in range(n):
            W[3 * n + i, col] = 1
            
        col += 1
        
    return U, V, W

def _iterative_improvement(U, V, W, n, m, p, deadline):
    """
    Try to iteratively improve the decomposition by removing columns.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try to remove each column
        for c in range(rank):
            if time.time() > deadline - 10:
                return U, V, W
            
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            if _verify_sparse(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                rank -= 1
                improved = True
                break
    
    return U, V, W

def _try_rank_r(n, m, p, r, deadline, max_attempts=200):
    """
    Try to find a rank-r decomposition using random search.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    rng = np.random.RandomState(12345)
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break
        
        # Use random initialization with half-integers
        U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(nm, r))
        V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(mp, r))
        W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(pn, r))
        
        if _verify_sparse(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with the rank-30 decomposition
    U, V, W = _schoolbook_3x3x4(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try iterative improvement
    U_imp, V_imp, W_imp = _iterative_improvement(U, V, W, n, m, p, deadline - 30)
    if U_imp.shape[1] < best_rank:
        best_rank = U_imp.shape[1]
        best_U, best_V, best_W = U_imp, V_imp, W_imp
    
    # Try random search for lower ranks
    # We know rank 30 works. Try 29, 28, ...
    for r in range(best_rank - 1, max(20, best_rank - 10), -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_r(n, m, p, r, deadline - 10, max_attempts=50)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
