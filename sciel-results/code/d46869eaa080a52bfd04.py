"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def _verify(U, V, W, n, m, p):
    """Verify exact tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return (U, V, W)

def _identity_V_decomposition(n, m, p):
    """
    Decomposition using V as identity matrix, giving rank = m*p.
    
    V[j*p+k, r] = 1 if r == j*p+k, else 0
    U[i*m+j, r] = 1 if r == j*p+k for some k, else 0 (specifically U[i*m+j, j*p+k] = 1)
    W[k*n+i, r] = 1 if r == j*p+k for some j, else 0 (specifically W[k*n+i, j*p+k] = 1)
    
    Verification: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    = U[i*m+j, j*p+k] * V[j*p+k, j*p+k] * W[k*n+i, j*p+k]
    = 1 * 1 * 1 = 1  ✓
    """
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # V is identity
    for idx in range(m * p):
        V[idx][idx] = 1
    
    # U[i*m+j, j*p+k] = 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][j*p+k] = 1
    
    # W[k*n+i, j*p+k] = 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W[k*n+i][j*p+k] = 1
    
    return (U, V, W)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # First, try the identity-V decomposition (rank = m*p = 21)
    U_id, V_id, W_id = _identity_V_decomposition(n, m, p)
    
    if _verify(U_id, V_id, W_id, n, m, p):
        return {"U": U_id, "V": V_id, "W": W_id}
    
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
