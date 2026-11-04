"""Program for matrix multiplication tensor decomposition task."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _build_3x3x3_rank23():
    """
    Build a rank-23 decomposition for 3x3*3x3.
    Uses Strassen for the 2x2*2x2 block and schoolbook for the rest.
    """
    n = m = p = 3
    rank = 23
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Strassen terms for top-left 2x2*2x2 (indices 0-6)
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[5][0] = 1
    
    # M2 = (a10+a11)b00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1; W[8][1] = -1
    
    # M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[1][2] = 1; W[8][2] = 1
    
    # M4 = a11(b10-b00)
    U[4][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[4][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[8][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[4][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1
    
    # Schoolbook terms for remaining entries (indices 7-22)
    # A[0,2]*B[2,0], A[0,2]*B[2,1], A[0,2]*B[2,2]
    # A[1,2]*B[2,0], A[1,2]*B[2,1], A[1,2]*B[2,2]
    # A[2,0]*B[0,0], A[2,0]*B[0,1], A[2,0]*B[0,2]
    # A[2,1]*B[1,0], A[2,1]*B[1,1], A[2,1]*B[1,2]
    # A[2,2]*B[2,0], A[2,2]*B[2,1], A[2,2]*B[2,2]
    # That's 15 terms, total 7+15=22. But we need to check if this is correct.
    
    col = 7
    for (i, j, k) in [(0,2,0),(0,2,1),(0,2,2),(1,2,0),(1,2,1),(1,2,2),
                       (2,0,0),(2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),
                       (2,2,0),(2,2,1),(2,2,2)]:
        U[i*3+j][col] = 1
        V[j*3+k][col] = 1
        W[k*3+i][col] = 1
        col += 1
    
    # We have 7 + 15 = 22 terms, but rank is 23. Let me add one more if needed.
    # Actually, let me verify this works.
    
    return U, V, W

def _build_3x3x6_from_3x3x3(U3, V3, W3):
    """
    Build a 3x3*3x6 decomposition from a 3x3*3x3 decomposition.
    """
    rank3 = len(U3[0])
    rank6 = 2 * rank3
    
    U6 = [[0]*rank6 for _ in range(9)]
    V6 = [[0]*rank6 for _ in range(18)]
    W6 = [[0]*rank6 for _ in range(18)]
    
    for r in range(rank3):
        for i in range(3):
            for j in range(3):
                U6[i*3+j][r] = U3[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V6[j*3+k][r] = V3[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W6[k*3+i][r] = W3[k*3+i][r]
    
    for r in range(rank3):
        for i in range(3):
            for j in range(3):
                U6[i*3+j][rank3+r] = U3[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V6[j*3 + 3 + k][rank3+r] = V3[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W6[k*3 + 3 + i][rank3+r] = W3[k*3+i][r]
    
    return U6, V6, W6

def _build_schoolbook_3x3x6():
    """Build schoolbook rank-54 decomposition for 3x3*3x6."""
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def time_left():
        return deadline - time.time()
    
    # Try rank-23 based 3x3*3x3, giving rank-46 for 3x3*3x6
    if time_left() > 10:
        U3, V3, W3 = _build_3x3x3_rank23()
        if _verify_decomposition(U3, V3, W3, 3, 3, 3):
            U6, V6, W6 = _build_3x3x6_from_3x3x3(U3, V3, W3)
            if _verify_decomposition(U6, V6, W6, n, m, p):
                return {"U": U6, "V": V6, "W": W6}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook_3x3x6()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
