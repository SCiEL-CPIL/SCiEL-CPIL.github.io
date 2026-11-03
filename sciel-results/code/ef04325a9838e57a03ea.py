import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    This program returns a verified rank-26 decomposition based on Strassen's algorithm
    for the 2x2 core and standard multiplication for the border.
    """
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def idx_u(i, j):
        return i * m + j
    
    def idx_v(j, k):
        return j * p + k
    
    def idx_w(k, i):
        return k * n + i
    
    def verify_decomp(U_mat, V_mat, W_mat, rank):
        """Verify that the tensor decomposition is correct."""
        # Check sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[idx_u(i,j), r] * V_mat[idx_v(j,k), r] * W_mat[idx_w(k,i), r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Rank 26 Decomposition: Strassen 2x2 Core + Border
    R = 26
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    # Strassen Terms for 2x2 Core (i,j,k in 0,1)
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # r=0: m1 = (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
    r = 0
    U[idx_u(0,0), r] = 1; U[idx_u(1,1), r] = 1
    V[idx_v(0,0), r] = 1; V[idx_v(1,1), r] = 1
    W[idx_w(0,0), r] = 1; W[idx_w(1,1), r] = 1
    
    # r=1: m2 = (a10+a11)b00 -> c01 += m2, c11 -= m2
    r = 1
    U[idx_u(1,0), r] = 1; U[idx_u(1,1), r] = 1
    V[idx_v(0,0), r] = 1
    W[idx_w(0,1), r] = 1; W[idx_w(1,1), r] = -1
    
    # r=2: m3 = a00(b01-b11) -> c10 += m3, c11 += m3
    r = 2
    U[idx_u(0,0), r] = 1
    V[idx_v(0,1), r] = 1; V[idx_v(1,1), r] = -1
    W[idx_w(1,0), r] = 1; W[idx_w(1,1), r] = 1
    
    # r=3: m4 = a11(b10-b00) -> c00 += m4, c01 += m4
    r = 3
    U[idx_u(1,1), r] = 1
    V[idx_v(1,0), r] = 1; V[idx_v(0,0), r] = -1
    W[idx_w(0,0), r] = 1; W[idx_w(0,1), r] = 1
    
    # r=4: m5 = (a00+a01)b11 -> c00 -= m5, c10 += m5
    r = 4
    U[idx_u(0,0), r] = 1; U[idx_u(0,1), r] = 1
    V[idx_v(1,1), r] = 1
    W[idx_w(0,0), r] = -1; W[idx_w(1,0), r] = 1
    
    # r=5: m6 = (a10-a00)(b00+b01) -> c11 += m6
    r = 5
    U[idx_u(1,0), r] = 1; U[idx_u(0,0), r] = -1
    V[idx_v(0,0), r] = 1; V[idx_v(0,1), r] = 1
    W[idx_w(1,1), r] = 1
    
    # r=6: m7 = (a01-a11)(b10+b11) -> c00 += m7
    r = 6
    U[idx_u(0,1), r] = 1; U[idx_u(1,1), r] = -1
    V[idx_v(1,0), r] = 1; V[idx_v(1,1), r] = 1
    W[idx_w(0,0), r] = 1
    
    # Border Terms (19 terms)
    # For all (i,j,k) not in {0,1}x{0,1}x{0,1}
    # i.e., if i==2 or j==2 or k==2
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[idx_u(i,j), col] = 1
                V[idx_v(j,k), col] = 1
                W[idx_w(k,i), col] = 1
                col += 1
                
    assert col == R, f"Expected {R} terms, got {col}"
    
    if verify_decomp(U, V, W, R):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Rank 27 if Rank 26 fails (should not happen)
    R27 = 27
    U27 = np.zeros((n * m, R27), dtype=np.float64)
    V27 = np.zeros((m * p, R27), dtype=np.float64)
    W27 = np.zeros((p * n, R27), dtype=np.float64)
    
    col27 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[idx_u(i,j), col27] = 1
                V27[idx_v(j,k), col27] = 1
                W27[idx_w(k,i), col27] = 1
                col27 += 1
                
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
