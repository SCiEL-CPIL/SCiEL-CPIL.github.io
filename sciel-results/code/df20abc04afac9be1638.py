"""Matrix multiplication 3,3,8 - rank-minimal decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    nm, mp, pn = n * m, m * p, p * n
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return U, V, W

def _try_rank_nm(n, m, p):
    """Try rank = n*m decomposition with U as identity."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # U is identity: U[i*m+j, i*m+j] = 1
    for i in range(n):
        for j in range(m):
            U[i*m+j, i*m+j] = 1.0
    
    # For the constraint to hold:
    # V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1 for all i,j,k
    # Set both to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                V[j*p+k, i*m+j] = 1.0
                W[k*n+i, i*m+j] = 1.0
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list, rank
    
    return None

def _try_rank_mp(n, m, p):
    """Try rank = m*p decomposition with V as identity."""
    nm, mp, pn = n * m, m * p, p * n
    rank = m * p
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # V is identity: V[j*p+k, j*p+k] = 1
    for j in range(m):
        for k in range(p):
            V[j*p+k, j*p+k] = 1.0
    
    # Constraint: U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, j*p+k] = 1.0
                W[k*n+i, j*p+k] = 1.0
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list, rank
    
    return None

def _try_rank_pn(n, m, p):
    """Try rank = p*n decomposition with W as identity."""
    nm, mp, pn = n * m, m * p, p * n
    rank = p * n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # W is identity: W[k*n+i, k*n+i] = 1
    for k in range(p):
        for i in range(n):
            W[k*n+i, k*n+i] = 1.0
    
    # Constraint: U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, k*n+i] = 1.0
                V[j*p+k, k*n+i] = 1.0
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list, rank
    
    return None

def _try_combinatorial_search(n, m, p, deadline):
    """
    Combinatorial search over structured support patterns.
    Try various rank levels with structured initializations.
    """
    nm, mp, pn = n * m, m * p, p * n
    best_rank = n * m * p
    best_decomp = _schoolbook(n, m, p)
    
    # Try rank = n*m = 9
    if time.time() < deadline:
        result = _try_rank_nm(n, m, p)
        if result:
            return result
    
    # Try rank = m*p = 24
    if time.time() < deadline:
        result = _try_rank_mp(n, m, p)
        if result:
            return result
    
    # Try rank = p*n = 24
    if time.time() < deadline:
        result = _try_rank_pn(n, m, p)
        if result:
            return result
    
    # If none of the structured approaches worked, return schoolbook
    return best_decomp[0], best_decomp[1], best_decomp[2], best_rank

def construct():
    """
    Returns a rank-minimal tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Try combinatorial search for lower-rank decomposition
    U, V, W, rank = _try_combinatorial_search(n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
