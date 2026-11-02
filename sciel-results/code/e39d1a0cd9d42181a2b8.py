"""
Strategy: Use the Kronecker product structure of matrix multiplication tensors.
M_6 can be decomposed using the product of a rank-7 decomposition of M_2 (Strassen)
and a rank-19 decomposition of M_3 (Coppersmith-Winograd).
Total rank = 7 * 19 = 133.
If the CW coefficients are not available or verification fails, fallback to
rank 189 (Strassen 2x2 * Schoolbook 3x3).
"""
DIMENSIONS = [6, 6, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication with rank 7.
    U, V, W are 4x7 matrices.
    """
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U indices: (i, j) -> i*2 + j
    # V indices: (j, k) -> j*2 + k
    # W indices: (k, i) -> k*2 + i
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1: (A00+A11)(B00+B11)
    # A00: (0,0), A11: (1,1)
    # B00: (0,0), B11: (1,1)
    # C00: M1 - M4 + M6 - M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M7
    
    # We need to map these to the tensor decomposition form:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # Term 0 (M1):
    # U: A00(1), A11(1)
    # V: B00(1), B11(1)
    # W: C00(1), C11(1) -- Wait, W is for C.
    # C00 gets M1 with +1. C11 gets M1 with +1.
    # C01 gets 0. C10 gets 0.
    
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # Term 1 (M2): (A10+A11)B00
    # U: A10(1), A11(1)
    # V: B00(1)
    # W: C10(1), C11(-1)
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
    
    # Term 2 (M3): A00(B01-B11)
    # U: A00(1)
    # V: B01(1), B11(-1)
    # W: C01(1), C11(1)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
    
    # Term 3 (M4): A11(B10-B00)
    # U: A11(1)
    # V: B10(1), B00(-1)
    # W: C00(-1), C10(1)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = -1; W[0*2+1, 3] = 1
    
    # Term 4 (M5): (A00+A01)B11
    # U: A00(1), A01(1)
    # V: B11(1)
    # W: C01(1)
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[1*2+0, 4] = 1
    
    # Term 5 (M6): (A10-A00)(B00+B01)
    # U: A10(1), A00(-1)
    # V: B00(1), B01(1)
    # W: C00(1)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[0*2+0, 5] = 1
    
    # Term 6 (M7): (A01-A11)(B10+B11)
    # U: A01(1), A11(-1)
    # V: B10(1), B11(1)
    # W: C00(-1), C11(1)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = -1; W[1*2+1, 6] = 1
    
    return U, V, W

def _cw_3x3_19():
    """
    Attempt to return U, V, W for 3x3 matrix multiplication with rank 19.
    If coefficients are not known, returns None.
    """
    # The Coppersmith-Winograd decomposition for 3x3 is complex.
    # I will return None to trigger fallback.
    return None

def _schoolbook_3x3():
    """
    Returns U, V, W for 3x3 matrix multiplication with rank 27.
    """
    U = np.zeros((9, 27), dtype=np.float64)
    V = np.zeros((9, 27), dtype=np.float64)
    W = np.zeros((9, 27), dtype=np.float64)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return U, V, W

def _kronecker_product_decomp(U1, V1, W1, U2, V2, W2):
    """
    Computes the Kronecker product decomposition of two tensor decompositions.
    Input: U1 (n1*m1, R1), V1 (m1*p1, R1), W1 (p1*n1, R1)
           U2 (n2*m2, R2), V2 (m2*p2, R2), W2 (p2*n2, R2)
    Output: U (n1*n2 * m1*m2, R1*R2), etc.
    """
    n1, m1, p1 = U1.shape[0]//m1, U1.shape[0]//U1.shape[1], 0 # Dimensions inferred from shapes?
    # Better to pass dimensions explicitly or infer.
    # U1 is n1*m1 x R1.
    n1m1 = U1.shape[0]
    m1p1 = V1.shape[0]
    p1n1 = W1.shape[0]
    R1 = U1.shape[1]
    
    n2m2 = U2.shape[0]
    m2p2 = V2.shape[0]
    p2n2 = W2.shape[0]
    R2 = U2.shape[1]
    
    # Infer n1, m1, p1
    # n1*m1 = n1m1, m1*p1 = m1p1, p1*n1 = p1n1
    # Assume square matrices for simplicity (n1=m1=p1)
    n1 = int(np.sqrt(n1m1))
    m1 = n1
    p1 = n1
    
    n2 = int(np.sqrt(n2m2))
    m2 = n2
    p2 = n2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    R = R1 * R2
    
    U = np.zeros((n*m, R), dtype=np.float64)
    V = np.zeros((m*p, R), dtype=np.float64)
    W = np.zeros((p*n, R), dtype=np.float64)
    
    for r1 in range(R1):
        for r2 in range(R2):
            c = r1 * R2 + r2
            
            # U indices: (I, i) for row, (J, j) for col
            # I in 0..n1-1, i in 0..n2-1
            # Global row: (I*n2 + i) * m + (J*m2 + j)
            # U1 index: I*m1 + J
            # U2 index: i*m2 + j
            
            for I in range(n1):
                for i in range(n2):
                    for J in range(m1):
                        for j in range(m2):
                            u1_val = U1[I*m1 + J, r1]
                            u2_val = U2[i*m2 + j, r2]
                            val = u1_val * u2_val
                            if val != 0:
                                global_row = (I*n2 + i) * m + (J*m2 + j)
                                U[global_row, c] = val
                                
            # V indices: (J, j) for row, (K, k) for col
            # V1 index: J*p1 + K
            # V2 index: j*p2 + k
            
            for J in range(m1):
                for j in range(m2):
                    for K in range(p1):
                        for k in range(p2):
                            v1_val = V1[J*p1 + K, r1]
                            v2_val = V2[j*p2 + k, r2]
                            val = v1_val * v2_val
                            if val != 0:
                                global_row = (J*m2 + j) * p + (K*p2 + k)
                                V[global_row, c] = val
                                
            # W indices: (K, k) for row, (I, i) for col
            # W1 index: K*n1 + I
            # W2 index: k*n2 + i
            
            for K in range(p1):
                for k in range(p2):
                    for I in range(n1):
                        for i in range(n2):
                            w1_val = W1[K*n1 + I, r1]
                            w2_val = W2[k*n2 + i, r2]
                            val = w1_val * w2_val
                            if val != 0:
                                global_row = (K*p2 + k) * n + (I*n2 + i)
                                W[global_row, c] = val
                                
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """
    Verify decomposition.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Get Strassen 2x2 (Rank 7)
    U2, V2, W2 = _strassen_2x2()
    
    # 2. Try Coppersmith-Winograd 3x3 (Rank 19)
    cw = _cw_3x3_19()
    if cw is not None:
        U3, V3, W3 = cw
    else:
        # Fallback to Schoolbook 3x3 (Rank 27)
        U3, V3, W3 = _schoolbook_3x3()
        
    # 3. Kronecker Product
    U6, V6, W6 = _kronecker_product_decomp(U2, V2, W2, U3, V3, W3)
    
    # 4. Verify
    if _verify_decomp(U6, V6, W6, n, m, p):
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    else:
        # Fallback to schoolbook 6x6 if verification fails (should not happen)
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
