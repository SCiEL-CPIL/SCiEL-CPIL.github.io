"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct using vectorized NumPy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # M1 = (a11+a22)(b11+b22) -> contributes to c11, c22
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # M2 = (a21+a22)b11 -> contributes to c21
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0
    
    # M3 = a11(b12-b22) -> contributes to c12
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0
    
    # M4 = a22(b21-b11) -> contributes to c11, c21
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[2][3] = 1.0
    
    # M5 = (a11+a12)b22 -> contributes to c11, c12
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0
    
    # M6 = (a21-a11)(b11+b12) -> contributes to c22
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7 = (a12-a22)(b21+b22) -> contributes to c11, c22
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0; W[3][6] = -1.0 # Fixed sign for c22
    
    return U, V, W

def _build_2x2x1():
    """Return rank-4 schoolbook decomposition for 2x2x1 matrix multiplication."""
    n, m, p = 2, 2, 1
    rank = 4
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1.0
            V[j*p+0][col] = 1.0
            W[0*n+i][col] = 1.0
            col += 1
    
    return U, V, W

def _build_2x2x3():
    """Return rank-11 decomposition for 2x2x3 matrix multiplication."""
    n, m, p = 2, 2, 3
    # Use Strassen for cols 0,1 and schoolbook for col 2
    U_s, V_s, W_s = _build_strassen_2x2x2()
    U_b, V_b, W_b = _build_2x2x1()
    
    rank = 7 + 4  # = 11
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # Copy Strassen part (cols 0,1)
    for i in range(n*m):
        for r in range(7):
            U[i][r] = U_s[i][r]
            
    # Map V_s (2x2 B matrix) to V (2x3 B matrix, cols 0,1)
    # V_s row s: j=s//2, k=s%2. Target row: j*p + k
    for s in range(4):
        j = s // 2
        k = s % 2
        t = j * p + k
        for r in range(7):
            V[t][r] = V_s[s][r]
            
    # Map W_s (2x2 C matrix) to W (2x3 C matrix, cols 0,1)
    # W_s row s: k=s//2, i=s%2. Target row: k*n + i
    for s in range(4):
        k = s // 2
        i = s % 2
        t = k * n + i
        for r in range(7):
            W[t][r] = W_s[s][r]
    
    # Copy schoolbook part for col 2, offset by 7
    for i in range(n*m):
        for r in range(4):
            U[i][7+r] = U_b[i][r]
    for j in range(m):
        for r in range(4):
            V[j*p+2][7+r] = V_b[j*p+0][r]
    for i in range(n):
        for r in range(4):
            W[2*n+i][7+r] = W_b[0*n+i][r]
    
    return U, V, W

def _build_2x2x4():
    """Return rank-14 decomposition for 2x2x4 using two Strassen blocks."""
    n, m, p = 2, 2, 4
    U1, V1, W1 = _build_strassen_2x2x2()
    U2, V2, W2 = _build_strassen_2x2x2()
    
    rank = 14
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # First Strassen for cols 0,1
    for i in range(n*m):
        for r in range(7):
            U[i][r] = U1[i][r]
    for s in range(4):
        j = s // 2
        k = s % 2
        t = j * p + k
        for r in range(7):
            V[t][r] = V1[s][r]
    for s in range(4):
        k = s // 2
        i = s % 2
        t = k * n + i
        for r in range(7):
            W[t][r] = W1[s][r]
    
    # Second Strassen for cols 2,3
    for i in range(n*m):
        for r in range(7):
            U[i][7+r] = U2[i][r]
    for s in range(4):
        j = s // 2
        k = s % 2
        t = j * p + (2 + k)
        for r in range(7):
            V[t][7+r] = V2[s][r]
    for s in range(4):
        k = s // 2
        i = s % 2
        t = (2 + k) * n + i
        for r in range(7):
            W[t][7+r] = W2[s][r]
    
    return U, V, W

def _build_2x2x7():
    """Return decomposition for 2x2x7 using Strassen blocks."""
    n, m, p = 2, 2, 7
    # Split 7 cols into 4+3
    U4, V4, W4 = _build_2x2x4()   # rank 14
    U3, V3, W3 = _build_2x2x3()   # rank 11
    
    rank = 14 + 11  # = 25
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # Copy 2x2x4 part (cols 0-3)
    for i in range(n*m):
        for r in range(14):
            U[i][r] = U4[i][r]
    for j in range(m):
        for c in range(4):
            for r in range(14):
                V[j*p+c][r] = V4[j*p+c][r]
    for k in range(4):
        for i in range(n):
            for r in range(14):
                W[k*n+i][r] = W4[k*n+i][r]
    
    # Copy 2x2x3 part (cols 4-6), offset by 14
    for i in range(n*m):
        for r in range(11):
            U[i][14+r] = U3[i][r]
    for j in range(m):
        for c in range(3):
            for r in range(11):
                V[j*p+(4+c)][14+r] = V3[j*p+c][r]
    for k in range(3):
        for i in range(n):
            for r in range(11):
                W[(4+k)*n+i][14+r] = W3[k*n+i][r]
    
    return U, V, W

def _build_2x4x7_decomposition():
    """Build a decomposition for 2x4x7 using block Strassen methods."""
    n, m, p = 2, 4, 7
    
    # Split A (2x4) into A1 (2x2) cols 0-1, A2 (2x2) cols 2-3
    # Split B (4x7) into B1 (2x7) rows 0-1, B2 (2x7) rows 2-3
    # A@B = A1@B1 + A2@B2
    # Each is M_{2,2,7}
    
    U1, V1, W1 = _build_2x2x7()  # rank 25
    U2, V2, W2 = _build_2x2x7()  # rank 25
    
    rank = 25 + 25  # = 50
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # A1@B1: A1 is rows of A with cols 0-1, B1 is rows 0-1 of B
    for i in range(n):
        for j in range(2):  # cols 0-1 of A
            for r in range(25):
                U[i*m+j][r] = U1[i*2+j][r]
    
    for j in range(2):  # rows 0-1 of B
        for k in range(p):
            for r in range(25):
                V[j*p+k][r] = V1[j*2+k][r]
    
    for k in range(p):
        for i in range(n):
            for r in range(25):
                W[k*n+i][r] = W1[k*n+i][r]
    
    # A2@B2: A2 is cols 2-3, B2 is rows 2-3
    offset = 25
    for i in range(n):
        for j in range(2, 4):  # cols 2-3 of A
            for r in range(25):
                U[i*m+j][offset+r] = U2[i*2+(j-2)][r]
    
    for j in range(2, 4):  # rows 2-3 of B
        for k in range(p):
            for r in range(25):
                V[j*p+k][offset+r] = V2[(j-2)*2+k][r]
    
    for k in range(p):
        for i in range(n):
            for r in range(25):
                W[k*n+i][offset+r] = W2[k*n+i][r]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Try the Strassen-based decomposition (rank 50)
    try:
        U, V, W = _build_2x4x7_decomposition()
        rank = len(U[0])
        
        # Verify
        if _verify_decomposition(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 56)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
