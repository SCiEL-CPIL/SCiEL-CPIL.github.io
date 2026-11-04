import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 5, 5]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification of tensor decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    U_T = U_arr.T  # (rank, n*m)
    V_T = V_arr.T  # (rank, m*p)
    W_T = W_arr.T  # (rank, p*n)
    
    # Build the full tensor T[a,b,c,d] = sum_r W[k*n+i,r] * U[a*m+b,r] * V[c*p+d,r]
    # where k=d, i=a (for the diagonal condition)
    # Actually: T[i,k,a,b,c,d] should be 1 iff a==i and b==c and d==k
    
    # Reshape for efficient computation
    # U_T is (rank, n*m), V_T is (rank, m*p), W_T is (rank, p*n)
    
    # For each output position (i, k) where i in [0,n), k in [0,p):
    #   For each input (a, b, c, d):
    #     term = W_T[:, k*n+i] dot (U_T[:, a*m+b] * V_T[:, c*p+d])
    
    # Vectorize over all (a,b,c,d) for each (i,k)
    for i in range(n):
        for k in range(p):
            w_col = W_T[:, k * n + i]  # (rank,)
            
            # Build expected: 1 if a==i and b==c and d==k
            # Compute actual for all (a,b,c,d)
            for a in range(n):
                for b in range(m):
                    u_col = U_T[:, a * m + b]  # (rank,)
                    for c in range(m):
                        v_col = V_T[:, c * p + k]  # (rank,) - d=k fixed
                        actual = np.dot(w_col, u_col * v_col)
                        expected = 1.0 if (a == i and b == c) else 0.0
                        if abs(actual - expected) > tol:
                            return False
                        # Also check d != k cases should be 0
                        for d in range(p):
                            if d == k:
                                continue
                            v_col_d = V_T[:, c * p + d]  # (rank,)
                            actual_d = np.dot(w_col, u_col * v_col_d)
                            if abs(actual_d) > tol:
                                return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured_pairing(n, m, p, R):
    """
    Structured approach: pair up the two rows (n=2) to reduce rank.
    For each column j in [0,m) and row pair, try to represent the 2*p terms
    using fewer rank-1 terms.
    
    Key insight: For fixed j, we need to represent:
      (i=0, j, k) for k in [0,p)  AND  (i=1, j, k) for k in [0,p)
    These share V[j*p+k]. 
    
    We can try: for each j, use p terms where each term handles both i=0 and i=1
    by having U have entries in both row 0 and row 1, and W have entries in both
    output positions.
    """
    if n != 2:
        return None
    
    # Strategy: For each j in [0,m), use p rank terms.
    # Total rank = m * p = 25 (same as schoolbook for n=2)
    # But we can try to do better by sharing across j's.
    
    # Alternative: Try rank = m*p with structured entries
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Fill with structured pattern
    for j in range(m):
        for k in range(p):
            r = j * p + k
            if r >= R:
                break
            # U: activate row (0,j) and (1,j)
            U[0 * m + j, r] = 1.0
            U[1 * m + j, r] = 1.0
            # V: activate (j,k)
            V[j * p + k, r] = 1.0
            # W: activate (k,0) and (k,1)
            W[k * n + 0, r] = 1.0
            W[k * n + 1, r] = 1.0
    
    # Verify
    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return U, V, W
    return None

def _try_random_sparse(n, m, p, R, seed):
    """Try a random sparse decomposition of rank R."""
    rng = random.Random(seed)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r in range(R):
        nnz_u = rng.randint(1, 4)
        for _ in range(nnz_u):
            idx = rng.randint(0, n * m - 1)
            U[idx, r] = rng.choice([-1, 1])
        
        nnz_v = rng.randint(1, 4)
        for _ in range(nnz_v):
            idx = rng.randint(0, m * p - 1)
            V[idx, r] = rng.choice([-1, 1])
        
        nnz_w = rng.randint(1, 4)
        for _ in range(nnz_w):
            idx = rng.randint(0, p * n - 1)
            W[idx, r] = rng.choice([-1, 1])
    
    return U, V, W

def _try_row_pairing(n, m, p, R):
    """
    Try to decompose by pairing rows. For n=2, group terms by column j.
    Each group of 2*p schoolbook terms might be representable in fewer terms.
    """
    if n != 2:
        return None
    
    # Try different sparsity patterns
    results = []
    
    # Pattern 1: For each j, use p terms that handle both rows
    # This gives rank = m*p = 25
    if R >= m * p:
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        for j in range(m):
            for k in range(p):
                r = j * p + k
                if r >= R:
                    break
                U[0 * m + j, r] = 1.0
                U[1 * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + 0, r] = 1.0
                W[k * n + 1, r] = 1.0
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            results.append((U, V, W))
    
    # Pattern 2: Try to share across columns
    # For each k, use m terms that handle all columns
    if R >= m * p:
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        for k in range(p):
            for j in range(m):
                r = k * m + j
                if r >= R:
                    break
                U[0 * m + j, r] = 1.0
                U[1 * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + 0, r] = 1.0
                W[k * n + 1, r] = 1.0
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            results.append((U, V, W))
    
    return results[0] if results else None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    random.seed(12345)
    
    # First try structured approaches
    for R in range(25, best_rank):
        if not still_going():
            break
        
        # Try structured row pairing
        result = _try_row_pairing(n, m, p, R)
        if result is not None:
            U, V, W = result
            best_rank = R
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            continue
        
        # Try random sparse with more attempts
        max_attempts = 100
        for attempt in range(max_attempts):
            if not still_going():
                break
            
            U, V, W = _try_random_sparse(n, m, p, R, seed=attempt * 1000 + R)
            
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = R
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best
