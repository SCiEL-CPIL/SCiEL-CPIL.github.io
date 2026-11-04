import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7).
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1; W[3, 3] = 1
    
    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition for n x m x p."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2):
    """Combine two decompositions side-by-side (block diagonal in the rank dimension)."""
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def _best_split(n, m, p, memo={}):
    """Find the optimal recursive decomposition using block splits.
    
    We can split any of the three dimensions into two parts.
    The rank is the sum of the ranks of the sub-problems.
    """
    key = (n, m, p)
    if key in memo:
        return memo[key]
    
    if n == 0 or m == 0 or p == 0:
        return 0
    
    # Schoolbook rank
    best_rank = n * m * p
    
    # Try splitting each dimension
    # Split m (inner dimension): m = m1 + m2
    # This splits A into (n x m1) and (n x m2), B into (m1 x p) and (m2 x p)
    # The subproblems are (n, m1, p) and (n, m2, p)
    for m1 in range(1, m):
        m2 = m - m1
        r = _best_split(n, m1, p, memo) + _best_split(n, m2, p, memo)
        if r < best_rank:
            best_rank = r
    
    # Split n (rows of A, rows of C): n = n1 + n2
    for n1 in range(1, n):
        n2 = n - n1
        r = _best_split(n1, m, p, memo) + _best_split(n2, m, p, memo)
        if r < best_rank:
            best_rank = r
    
    # Split p (cols of B, cols of C): p = p1 + p2
    for p1 in range(1, p):
        p2 = p - p1
        r = _best_split(n, m, p1, memo) + _best_split(n, m, p2, memo)
        if r < best_rank:
            best_rank = r
    
    memo[key] = best_rank
    return best_rank

def _build_decomposition(n, m, p, memo={}):
    """Build the optimal recursive decomposition.
    
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank).
    """
    key = (n, m, p)
    if key in memo:
        return memo[key]
    
    # Base case: small enough to use Strassen or schoolbook directly
    if n == 2 and m == 2 and p == 2:
        result = _strassen_2x2()
        memo[key] = result
        return result
    
    # Schoolbook for very small cases
    if n * m * p <= 8:
        result = _schoolbook(n, m, p)
        memo[key] = result
        return result
    
    # Try all splits and pick the best
    best_rank = n * m * p
    best_result = None
    
    # Split m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _build_decomposition(n, m1, p, memo)
        U2, V2, W2 = _build_decomposition(n, m2, p, memo)
        
        # Need to remap indices
        # For subproblem 1: A is n x m1, B is m1 x p
        # U1 rows: i*m1 + j for i in [0,n), j in [0,m1)
        # V1 rows: j*p + k for j in [0,m1), k in [0,p)
        # W1 rows: k*n + i for k in [0,p), i in [0,n)
        
        # For the full problem:
        # U rows: i*m + j for i in [0,n), j in [0,m)
        # V rows: j*p + k for j in [0,m), k in [0,p)
        # W rows: k*n + i for k in [0,p), i in [0,n)
        
        # Remap U1: row i*m1 + j -> i*m + j (j in [0, m1))
        # Remap U2: row i*m2 + j -> i*m + (m1 + j) (j in [0, m2))
        
        U1_full = np.zeros((n * m, U1.shape[1]), dtype=float)
        for i in range(n):
            for j in range(m1):
                U1_full[i * m + j, :] = U1[i * m1 + j, :]
        
        U2_full = np.zeros((n * m, U2.shape[1]), dtype=float)
        for i in range(n):
            for j in range(m2):
                U2_full[i * m + (m1 + j), :] = U2[i * m2 + j, :]
        
        # Remap V1: row j*p + k for j in [0, m1) -> same (j is already in [0, m1))
        V1_full = np.zeros((m * p, V1.shape[1]), dtype=float)
        for j in range(m1):
            for k in range(p):
                V1_full[j * p + k, :] = V1[j * p + k, :]
        
        # Remap V2: row j*p2 + k for j in [0, m2) -> (m1 + j)*p + k
        V2_full = np.zeros((m * p, V2.shape[1]), dtype=float)
        for j in range(m2):
            for k in range(p):
                V2_full[(m1 + j) * p + k, :] = V2[j * p + k, :]
        
        # W is unchanged since n and p are the same
        W1_full = W1
        W2_full = W2
        
        U_full = np.hstack([U1_full, U2_full])
        V_full = np.hstack([V1_full, V2_full])
        W_full = np.hstack([W1_full, W2_full])
        
        r = U_full.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = (U_full, V_full, W_full)
    
    # Split n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _build_decomposition(n1, m, p, memo)
        U2, V2, W2 = _build_decomposition(n2, m, p, memo)
        
        # Remap U1: row i*m + j for i in [0, n1) -> same
        U1_full = np.zeros((n * m, U1.shape[1]), dtype=float)
        for i in range(n1):
            for j in range(m):
                U1_full[i * m + j, :] = U1[i * m + j, :]
        
        # Remap U2: row i*m + j for i in [0, n2) -> (n1 + i)*m + j
        U2_full = np.zeros((n * m, U2.shape[1]), dtype=float)
        for i in range(n2):
            for j in range(m):
                U2_full[(n1 + i) * m + j, :] = U2[i * m + j, :]
        
        # V is unchanged
        V1_full = V1
        V2_full = V2
        
        # Remap W1: row k*n1 + i for k in [0,p), i in [0,n1) -> k*n + i
        W1_full = np.zeros((p * n, W1.shape[1]), dtype=float)
        for k in range(p):
            for i in range(n1):
                W1_full[k * n + i, :] = W1[k * n1 + i, :]
        
        # Remap W2: row k*n2 + i for k in [0,p), i in [0,n2) -> k*n + (n1 + i)
        W2_full = np.zeros((p * n, W2.shape[1]), dtype=float)
        for k in range(p):
            for i in range(n2):
                W2_full[k * n + (n1 + i), :] = W2[k * n2 + i, :]
        
        U_full = np.hstack([U1_full, U2_full])
        V_full = np.hstack([V1_full, V2_full])
        W_full = np.hstack([W1_full, W2_full])
        
        r = U_full.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = (U_full, V_full, W_full)
    
    # Split p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _build_decomposition(n, m, p1, memo)
        U2, V2, W2 = _build_decomposition(n, m, p2, memo)
        
        # U is unchanged
        U1_full = U1
        U2_full = U2
        
        # Remap V1: row j*p1 + k for j in [0,m), k in [0,p1) -> j*p + k
        V1_full = np.zeros((m * p, V1.shape[1]), dtype=float)
        for j in range(m):
            for k in range(p1):
                V1_full[j * p + k, :] = V1[j * p1 + k, :]
        
        # Remap V2: row j*p2 + k for j in [0,m), k in [0,p2) -> j*p + (p1 + k)
        V2_full = np.zeros((m * p, V2.shape[1]), dtype=float)
        for j in range(m):
            for k in range(p2):
                V2_full[j * p + (p1 + k), :] = V2[j * p2 + k, :]
        
        # Remap W1: row k*n + i for k in [0,p1) -> same (k in [0, p1))
        W1_full = np.zeros((p * n, W1.shape[1]), dtype=float)
        for k in range(p1):
            for i in range(n):
                W1_full[k * n + i, :] = W1[k * n + i, :]
        
        # Remap W2: row k*n + i for k in [0,p2) -> (p1 + k)*n + i
        W2_full = np.zeros((p * n, W2.shape[1]), dtype=float)
        for k in range(p2):
            for i in range(n):
                W2_full[(p1 + k) * n + i, :] = W2[k * n + i, :]
        
        U_full = np.hstack([U1_full, U2_full])
        V_full = np.hstack([V1_full, V2_full])
        W_full = np.hstack([W1_full, W2_full])
        
        r = U_full.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = (U_full, V_full, W_full)
    
    if best_result is None:
        # Fall back to schoolbook
        best_result = _schoolbook(n, m, p)
    
    memo[key] = best_result
    return best_result

def _to_list(mat):
    """Convert numpy array to list with integer/half-integer entries."""
    result = []
    for row in mat:
        r = []
        for val in row:
            if abs(val - round(val)) < 1e-10:
                r.append(int(round(val)))
            elif abs(val * 2 - round(val * 2)) < 1e-10:
                r.append(val)  # keep as float if half-integer
            else:
                r.append(val)
        result.append(r)
    return result

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # seconds, safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to build the optimal decomposition
    try:
        U, V, W = _build_decomposition(n, m, p, memo={})
        
        # Verify
        if not _verify(U, V, W, n, m, p):
            # Fall back to schoolbook
            U, V, W = _schoolbook(n, m, p)
        
        rank = U.shape[1]
        
    except Exception as e:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        rank = U.shape[1]
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Shouldn't happen, but just in case, return what we have
        pass
    
    U_list = _to_list(U)
    V_list = _to_list(V)
    W_list = _to_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
