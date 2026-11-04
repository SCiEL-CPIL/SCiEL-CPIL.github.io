import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    U_r = U_np.reshape(n, m, -1)
    V_r = V_np.reshape(m, p, -1)
    W_r = W_np.reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x4_block(U, V, W, i0, jA0, jB0, k0, col):
    """
    Strassen decomposition for a 2x2x4 sub-block.
    A sub-block: rows i0, i0+1; cols jA0, jA0+1 of A; cols jB0, jB0+1 of B; rows k0..k0+3 of output.
    We partition the 4 output columns into two groups of 2.
    """
    n, m, p = 3, 4, 8
    # M1 = (A[i0,jA0] + A[i0+1,jA0+1]) * (B[jB0,k0] + B[jB0+1,k0+1])
    #     + (A[i0,jA0] + A[i0+1,jA0+1]) * (B[jB0,k0+2] + B[jB0+1,k0+3])
    # Actually, let's think of this differently.
    # For 2x2x2 Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2x4, we have 4 output columns. Let's split into two 2x2x2 blocks.
    # Block 1: k0, k0+1
    # Block 2: k0+2, k0+3
    
    # Block 1 (k0, k0+1):
    # M1_1 = (A[i0,jA0]+A[i0+1,jA0+1]) * (B[jB0,k0]+B[jB0+1,k0+1])
    # M2_1 = (A[i0+1,jA0]+A[i0+1,jA0+1]) * B[jB0,k0]
    # M3_1 = A[i0,jA0] * (B[jB0+1,k0+1]-B[jB0,k0+1])
    # M4_1 = A[i0+1,jA0+1] * (B[jB0,k0+1]-B[jB0+1,k0])
    # M5_1 = (A[i0,jA0]+A[i0,jA0+1]) * B[jB0+1,k0+1]
    # M6_1 = (A[i0+1,jA0]-A[i0,jA0]) * (B[jB0,k0]+B[jB0,k0+1])
    # M7_1 = (A[i0,jA0+1]-A[i0+1,jA0+1]) * (B[jB0+1,k0]+B[jB0+1,k0+1])
    
    # C[i0,k0] = M1_1 + M4_1 - M5_1 + M7_1
    # C[i0,k0+1] = M3_1 + M5_1
    # C[i0+1,k0] = M2_1 + M4_1
    # C[i0+1,k0+1] = M1_1 - M2_1 + M3_1 + M6_1
    
    c = col
    # M1_1
    U[i0*m+jA0, c] += 1
    U[(i0+1)*m+(jA0+1), c] += 1
    V[jB0*p+k0, c] += 1
    V[(jB0+1)*p+(k0+1), c] += 1
    W[k0*n+i0, c] += 1
    W[(k0+1)*n+(i0+1), c] += 1
    c += 1
    # M2_1
    U[(i0+1)*m+jA0, c] += 1
    U[(i0+1)*m+(jA0+1), c] += 1
    V[jB0*p+k0, c] += 1
    W[k0*n+(i0+1), c] += 1
    W[(k0+1)*n+(i0+1), c] -= 1
    c += 1
    # M3_1
    U[i0*m+jA0, c] += 1
    V[(jB0+1)*p+(k0+1), c] += 1
    V[jB0*p+(k0+1), c] -= 1
    W[(k0+1)*n+i0, c] += 1
    W[(k0+1)*n+(i0+1), c] += 1
    c += 1
    # M4_1
    U[(i0+1)*m+(jA0+1), c] += 1
    V[jB0*p+(k0+1), c] += 1
    V[(jB0+1)*p+k0, c] -= 1
    W[k0*n+i0, c] += 1
    W[k0*n+(i0+1), c] += 1
    c += 1
    # M5_1
    U[i0*m+jA0, c] += 1
    U[i0*m+(jA0+1), c] += 1
    V[(jB0+1)*p+(k0+1), c] += 1
    W[k0*n+i0, c] -= 1
    W[(k0+1)*n+i0, c] += 1
    c += 1
    # M6_1
    U[(i0+1)*m+jA0, c] += 1
    U[i0*m+jA0, c] -= 1
    V[jB0*p+k0, c] += 1
    V[jB0*p+(k0+1), c] += 1
    W[(k0+1)*n+(i0+1), c] += 1
    c += 1
    # M7_1
    U[i0*m+(jA0+1), c] += 1
    U[(i0+1)*m+(jA0+1), c] -= 1
    V[(jB0+1)*p+k0, c] += 1
    V[(jB0+1)*p+(k0+1), c] += 1
    W[k0*n+i0, c] += 1
    c += 1
    
    # Block 2 (k0+2, k0+3):
    c = col + 7
    # M1_2
    U[i0*m+jA0, c] += 1
    U[(i0+1)*m+(jA0+1), c] += 1
    V[jB0*p+(k0+2), c] += 1
    V[(jB0+1)*p+(k0+3), c] += 1
    W[(k0+2)*n+i0, c] += 1
    W[(k0+3)*n+(i0+1), c] += 1
    c += 1
    # M2_2
    U[(i0+1)*m+jA0, c] += 1
    U[(i0+1)*m+(jA0+1), c] += 1
    V[jB0*p+(k0+2), c] += 1
    W[(k0+2)*n+(i0+1), c] += 1
    W[(k0+3)*n+(i0+1), c] -= 1
    c += 1
    # M3_2
    U[i0*m+jA0, c] += 1
    V[(jB0+1)*p+(k0+3), c] += 1
    V[jB0*p+(k0+3), c] -= 1
    W[(k0+3)*n+i0, c] += 1
    W[(k0+3)*n+(i0+1), c] += 1
    c += 1
    # M4_2
    U[(i0+1)*m+(jA0+1), c] += 1
    V[jB0*p+(k0+3), c] += 1
    V[(jB0+1)*p+(k0+2), c] -= 1
    W[(k0+2)*n+i0, c] += 1
    W[(k0+2)*n+(i0+1), c] += 1
    c += 1
    # M5_2
    U[i0*m+jA0, c] += 1
    U[i0*m+(jA0+1), c] += 1
    V[(jB0+1)*p+(k0+3), c] += 1
    W[(k0+2)*n+i0, c] -= 1
    W[(k0+3)*n+i0, c] += 1
    c += 1
    # M6_2
    U[(i0+1)*m+jA0, c] += 1
    U[i0*m+jA0, c] -= 1
    V[jB0*p+(k0+2), c] += 1
    V[jB0*p+(k0+3), c] += 1
    W[(k0+3)*n+(i0+1), c] += 1
    c += 1
    # M7_2
    U[i0*m+(jA0+1), c] += 1
    U[(i0+1)*m+(jA0+1), c] -= 1
    V[(jB0+1)*p+(k0+2), c] += 1
    V[(jB0+1)*p+(k0+3), c] += 1
    W[(k0+2)*n+i0, c] += 1
    c += 1
    
    return col + 14

def _construct_hybrid():
    n, m, p = 3, 4, 8
    R = 60
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    # Row 0: schoolbook
    for j in range(4):
        for k in range(8):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
    
    # Rows 1-2: Strassen blocks
    # 2 blocks of 2x2x4 for rows 1-2, covering j=0,1 and j=2,3
    col = _strassen_2x2x4_block(U, V, W, 1, 0, 0, 0, col)
    col = _strassen_2x2x4_block(U, V, W, 1, 2, 2, 0, col)
    
    assert col == 60, f"Expected 60, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + 120
    
    try:
        U_h, V_h, W_h = _construct_hybrid()
        if _verify(U_h, V_h, W_h, n, m, p):
            return {"U": U_h, "V": V_h, "W": W_h}
    except Exception:
        pass
    
    # Schoolbook fallback
    R = n * m * p
    U_sb = [[0]*R for _ in range(n*m)]
    V_sb = [[0]*R for _ in range(m*p)]
    W_sb = [[0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
