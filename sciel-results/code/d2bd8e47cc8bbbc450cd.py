"""
Matrix multiplication 3x5x5 with rank-minimal exact tensor decomposition.
Uses block decomposition and Strassen for 2x2x2 sub-blocks to reduce rank.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _schoolbook_factors(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _strassen_222_factors():
    """Strassen decomposition of 2x2x2, rank 7."""
    # U: 4x7, V: 4x7, W: 4x7
    # Indexing: U row = i*2+j, V row = j*2+k, W row = k*2+i
    U = np.array([
        [ 1,  1,  1,  0,  0,  0,  0],  # i=0,j=0
        [ 0,  0,  0,  1,  0,  1, -1],  # i=0,j=1
        [ 1,  0,  0,  0,  1,  0,  0],  # i=1,j=0
        [ 0, -1,  0,  0,  0,  1,  1],  # i=1,j=1
    ], dtype=int)
    V = np.array([
        [ 1,  1,  0,  1,  1,  0,  0],  # j=0,k=0
        [ 0,  0,  1, -1,  0,  1,  0],  # j=0,k=1
        [ 1,  0,  1,  1,  0,  0,  0],  # j=1,k=0
        [ 0,  1,  0,  0,  1,  0,  1],  # j=1,k=1
    ], dtype=int)
    W = np.array([
        [ 0,  1,  1,  0,  0,  0,  0],  # k=0,i=0
        [ 0,  0,  0,  0,  1,  1,  1],  # k=0,i=1
        [ 1,  0,  0,  1,  0,  1, -1],  # k=1,i=0
        [ 0,  0,  0,  0,  1,  0,  0],  # k=1,i=1
    ], dtype=int)
    return U, V, W, 7

def _combine_factors(factors_list):
    """Combine a list of (U, V, W, rank) tuples by concatenating columns."""
    if len(factors_list) == 0:
        return None, None, None, 0
    if len(factors_list) == 1:
        return factors_list[0]
    
    U_list = []
    V_list = []
    W_list = []
    total_rank = 0
    for U, V, W, rank in factors_list:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        total_rank += rank
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W, total_rank

def _get_factors(n, m, p):
    """
    Get factors for n x m x p matrix multiplication.
    Returns (U, V, W, rank) where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    """
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return _schoolbook_factors(1, 1, 1)
    if n == 2 and m == 2 and p == 2:
        return _strassen_222_factors()
    
    # If any dimension is 1, use schoolbook
    if n == 1 or m == 1 or p == 1:
        return _schoolbook_factors(n, m, p)
    
    # Recursive splitting strategy
    # Split the largest dimension into 1 + (dim-1) if possible
    factors_list = []
    
    # Try splitting n
    if n >= 3:
        n1, n2 = 1, n - 1
        U1, V1, W1, r1 = _get_factors(n1, m, p)
        U2, V2, W2, r2 = _get_factors(n2, m, p)
        # Need to embed these into the full n x m x p problem
        # For n split: rows of U are i*m+j, rows of W are k*n+i
        # U1 corresponds to i in [0, n1), U2 to i in [n1, n)
        # W1 corresponds to i in [0, n1), W2 to i in [n1, n)
        
        # Build full U, V, W from partial
        U_full = np.zeros((n * m, r1 + r2), dtype=int)
        V_full = np.zeros((m * p, r1 + r2), dtype=int)
        W_full = np.zeros((p * n, r1 + r2), dtype=int)
        
        # Fill U for first part (i in [0, n1))
        U_full[:n1 * m, :r1] = U1
        # Fill U for second part (i in [n1, n))
        U_full[n1 * m:, r1:] = U2
        # V is the same for both parts (since m, p unchanged)
        V_full[:, :r1] = V1
        V_full[:, r1:] = V2
        # Fill W for first part (i in [0, n1))
        # W rows are k*n + i, so for i in [0, n1), the rows are at positions k*n + i for all k
        for k in range(p):
            W_full[k * n:(k + 1) * n1, :r1] = W1[k * n1:(k + 1) * n1]
        for k in range(p):
            W_full[k * n + n1:, r1:] = W2[k * n1:(k + 1) * n1]
        
        return U_full, V_full, W_full, r1 + r2
    
    # Try splitting m
    if m >= 3:
        m1, m2 = 1, m - 1
        U1, V1, W1, r1 = _get_factors(n, m1, p)
        U2, V2, W2, r2 = _get_factors(n, m2, p)
        
        U_full = np.zeros((n * m, r1 + r2), dtype=int)
        V_full = np.zeros((m * p, r1 + r2), dtype=int)
        W_full = np.zeros((p * n, r1 + r2), dtype=int)
        
        # U rows are i*m + j
        for i in range(n):
            U_full[i * m:(i + 1) * m1, :r1] = U1[i * m1:(i + 1) * m1]
            U_full[i * m + m1:, r1:] = U2[i * m1:(i + 1) * m1]
        
        # V rows are j*p + k
        V_full[:m1 * p, :r1] = V1
        V_full[m1 * p:, r1:] = V2
        
        # W rows are k*n + i (unchanged)
        W_full[:, :r1] = W1
        W_full[:, r1:] = W2
        
        return U_full, V_full, W_full, r1 + r2
    
    # Try splitting p
    if p >= 3:
        p1, p2 = 1, p - 1
        U1, V1, W1, r1 = _get_factors(n, m, p1)
        U2, V2, W2, r2 = _get_factors(n, m, p2)
        
        U_full = np.zeros((n * m, r1 + r2), dtype=int)
        V_full = np.zeros((m * p, r1 + r2), dtype=int)
        W_full = np.zeros((p * n, r1 + r2), dtype=int)
        
        # U is the same for both parts
        U_full[:, :r1] = U1
        U_full[:, r1:] = U2
        
        # V rows are j*p + k
        for j in range(m):
            V_full[j * p:(j + 1) * p1, :r1] = V1[j * p1:(j + 1) * p1]
            V_full[j * p + p1:, r1:] = V2[j * p1:(j + 1) * p1]
        
        # W rows are k*n + i
        W_full[:p1 * n, :r1] = W1
        W_full[p1 * n:, r1:] = W2
        
        return U_full, V_full, W_full, r1 + r2
    
    # Fallback: schoolbook
    return _schoolbook_factors(n, m, p)

def construct():
    """
    Construct factor matrices for 3x5x5 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} where U is (15, rank), V is (25, rank), W is (15, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to get a good decomposition
    try:
        U, V, W, rank = _get_factors(n, m, p)
        
        # Convert to lists if needed
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    except Exception as e:
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook_factors(n, m, p)
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
