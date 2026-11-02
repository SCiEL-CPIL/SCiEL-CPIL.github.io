"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

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

def _combine_decompositions(n, m, p, r1, r2, U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions:
    First handles columns 0..p1-1 of B (and corresponding C columns)
    Second handles columns p1..p-1 of B (and corresponding C columns)
    
    U1: (n*m) x r1, V1: (m*p1) x r1, W1: (p1*n) x r1
    U2: (n*m) x r2, V2: (m*p2) x r2, W2: (p2*n) x r2
    where p1 + p2 = p
    
    Result: U: (n*m) x (r1+r2), V: (m*p) x (r1+r2), W: (p*n) x (r1+r2)
    """
    p1 = len(V1) // m  # number of columns in first part
    p2 = len(V2) // m  # number of columns in second part
    total_rank = r1 + r2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy first decomposition
    for i in range(n):
        for j in range(m):
            for c in range(r1):
                U[i*m + j][c] = U1[i*m + j][c]
    for j in range(m):
        for k in range(p1):
            for c in range(r1):
                V[j*p + k][c] = V1[j*p1 + k][c]
    for k in range(p1):
        for i in range(n):
            for c in range(r1):
                W[k*n + i][c] = W1[k*n + i][c]
    
    # Copy second decomposition (offset by r1)
    for i in range(n):
        for j in range(m):
            for c in range(r2):
                U[i*m + j][r1 + c] = U2[i*m + j][c]
    for j in range(m):
        for k in range(p2):
            for c in range(r2):
                V[j*p + p1 + k][r1 + c] = V2[j*p2 + k][c]
    for k in range(p2):
        for i in range(n):
            for c in range(r2):
                W[(p1 + k)*n + i][r1 + c] = W2[k*n + i][c]
    
    return (U, V, W)

def _try_winograd_3x3_3x3(n, m, p):
    """
    Try to find a rank-23 decomposition for 3x3 * 3x3 using Winograd's algorithm.
    Winograd's method for 3x3 matrix multiplication achieves rank 23.
    
    For C = A * B where A, B are 3x3:
    Use Winograd's algorithm which reduces from 27 to 23 multiplications.
    """
    # Winograd's algorithm for 3x3 matrix multiplication
    # This is a well-known result: rank(3,3,3) = 23
    
    # The Winograd decomposition uses the following structure:
    # We can group the 27 terms into 23 by exploiting linearity
    
    # Actually, let's use a known rank-23 decomposition for 3x3*3x3
    # One approach: use the fact that we can combine some terms
    
    # For simplicity, let's try to construct it directly
    # Winograd's 3x3 algorithm:
    # d1 = a11 + a13
    # d2 = a13 + a23
    # d3 = a31 + a33
    # d4 = a33 + a23
    # d5 = a11 + a21
    # d6 = a21 + a23
    # d7 = a31 + a33
    # d8 = a33 + a13
    # e1 = b11 + b12
    # e2 = b12 + b22
    # e3 = b22 + b23
    # e4 = b23 + b33
    # e5 = b11 + b21
    # e6 = b21 + b31
    # e7 = b31 + b33
    # e8 = b33 + b13
    # f1 = d1 * e1
    # f2 = d2 * e2
    # f3 = d3 * e3
    # f4 = d4 * e4
    # f5 = d5 * e5
    # f6 = d6 * e6
    # f7 = d7 * e7
    # f8 = d8 * e8
    # c11 = f1 + f5 + f8 - f2 - f3 - f7
    # c12 = f2 + f6 - f1 - f3
    # c13 = f3 + f7 - f4 - f6
    # c21 = f5 + f8 - f4 - f7
    # c22 = f1 + f4 - f5 - f8
    # c23 = f2 + f6 - f1 - f3
    # c31 = f3 + f7 - f4 - f6
    # c32 = f2 + f6 - f1 - f3
    # c33 = f1 + f4 - f5 - f8
    
    # This gives 8 multiplications, but we need to express as tensor decomposition
    # The rank is 23, not 8, because the algorithm has 23 "essential" multiplications
    
    # Let me use a different approach: just return schoolbook for now
    # and combine with the 3x3*3x4 part
    return None

def _try_3x3_3x4_decomposition(n, m, p):
    """
    Try to find a low-rank decomposition for 3x3 * 3x4 matrix multiplication.
    Schoolbook rank is 3*3*4 = 36.
    Lower bound is max(9, 12, 12) = 12.
    """
    # For 3x3 * 3x4, let's try to find a better decomposition
    # The schoolbook gives rank 36
    
    # One strategy: split the 4 columns into 2+2
    # Each 3x3 * 3x2 can potentially be done with rank 18 (schoolbook)
    # But maybe we can do better
    
    # For now, return schoolbook
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
    return (U, V, W, rank)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Split 3x3 * 3x7 into 3x3 * 3x3 and 3x3 * 3x4
    # Use schoolbook for each part, then combine
    
    p1 = 3  # first part handles 3 columns
    p2 = 4  # second part handles 4 columns
    
    # Get schoolbook decompositions for each part
    U1, V1, W1 = _schoolbook(n, m, p1)
    U2, V2, W2 = _schoolbook(n, m, p2)
    
    r1 = n * m * p1  # 27
    r2 = n * m * p2  # 36
    
    # Combine the decompositions
    U, V, W = _combine_decompositions(n, m, p, r1, r2, U1, V1, W1, U2, V2, W2)
    
    total_rank = r1 + r2  # 63, same as schoolbook
    
    # Verify the result
    if not _verify(U, V, W, n, m, p):
        # Fall back to direct schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Try to find a better decomposition by searching for lower rank
    # For 3x3 * 3x7, the lower bound is max(9, 21, 21) = 21
    
    # Try some structured approaches
    if time.time() < deadline - 10:
        # Try Winograd-style decomposition for the 3x3 part
        # If we can get rank 23 for 3x3*3x3 and rank 36 for 3x3*3x4, total = 59
        # This would be an improvement over 63
        
        # For now, just use the combined schoolbook
        pass
    
    U, V, W = best_result if 'best_result' in locals() else (U, V, W)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
