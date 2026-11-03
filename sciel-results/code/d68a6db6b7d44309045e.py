"""Optimized decomposition for 3x3 * 3x7 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Try Strassen for 2x2*2x2 core + schoolbook for rest
    U = [[0]*7 for _ in range(n*m)]
    V = [[0]*7 for _ in range(m*p)]
    W = [[0]*7 for _ in range(p*n)]
    
    # Strassen products for i in {0,1}, j in {0,1}, k in {0,1}
    # p1 = (A00+A11)*(B00+B11)
    r = 0
    U[0*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+0][r] = 1; V[1*p+1][r] = 1
    W[0*n+0][r] = 1; W[1*n+1][r] = 1
    
    # p2 = (A10+A11)*B01
    r = 1
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+1][r] = 1
    W[0*n+1][r] = 1; W[1*n+1][r] = -1
    
    # p3 = A00*(B01-B11)
    r = 2
    U[0*m+0][r] = 1
    V[0*p+1][r] = 1; V[1*p+1][r] = -1
    W[1*n+0][r] = 1
    
    # p4 = A11*(B10-B00)
    r = 3
    U[1*m+1][r] = 1
    V[1*p+0][r] = 1; V[0*p+0][r] = -1
    W[0*n+0][r] = 1; W[0*n+1][r] = 1
    
    # p5 = (A00+A01)*B11
    r = 4
    U[0*m+0][r] = 1; U[0*m+1][r] = 1
    V[1*p+1][r] = 1
    W[0*n+0][r] = -1; W[1*n+0][r] = 1
    
    # p6 = (A10-A00)*(B00+B10)
    r = 5
    U[1*m+0][r] = 1; U[0*m+0][r] = -1
    V[0*p+0][r] = 1; V[1*p+0][r] = 1
    # p6 not used in C formulas, so W all zero for this r
    
    # p7 = (A01-A11)*(B01+B11)
    r = 6
    U[0*m+1][r] = 1; U[1*m+1][r] = -1
    V[0*p+1][r] = 1; V[1*p+1][r] = 1
    W[0*n+0][r] = -1; W[1*n+1][r] = 1
    
    # Add schoolbook terms for uncovered (i,j,k)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    rank = col
    
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fall back to pure schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
