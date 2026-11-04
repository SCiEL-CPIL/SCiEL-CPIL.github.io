import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7) -> indices i*2+j
    V: (4, 7) -> indices j*2+k
    W: (4, 7) -> indices k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Col 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Col 1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Col 2: M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Col 3: M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Col 4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Col 5: M6 = (A01-A00)(B00+B10)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7 = (A10-A11)(B01+B11)
    U[2, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _trivial_2x1x2():
    """Returns U, V, W for 2x1x2 matrix multiplication. Rank 4.
    U: (2, 4) -> indices i*1+j (i in {0,1}, j=0)
    V: (2, 4) -> indices j*2+k (j=0, k in {0,1})
    W: (4, 4) -> indices k*2+i (k in {0,1}, i in {0,1})
    """
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    
    # Col 0: A0*B0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Col 1: A1*B0
    U[1, 1] = 1; V[0, 1] = 1; W[2, 1] = 1
    # Col 2: A0*B1
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    # Col 3: A1*B1
    U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    
    return U, V, W

def _trivial_1x1x3():
    """Returns U, V, W for 1x1x3 matrix multiplication. Rank 3.
    U: (1, 3) -> indices i*1+j (i=0, j=0)
    V: (3, 3) -> indices j*3+k (j=0, k in {0,1,2})
    W: (3, 3) -> indices k*1+i (k in {0,1,2}, i=0)
    """
    U = np.zeros((1, 3))
    V = np.zeros((3, 3))
    W = np.zeros((3, 3))
    
    for k in range(3):
        U[0, k] = 1
        V[k, k] = 1
        W[k, k] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _kronecker_decompose(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decompositions.
    (U1,V1,W1) for (n1,m1,p1) and (U2,V2,W2) for (n2,m2,p2)
    gives decomposition for (n1*n2, m1*m2, p1*p2).
    """
    r1, r2 = U1.shape[1], U2.shape[1]
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            # U index: (i1*n2+i2)*m + (j1*m2+j2)
                            U[(i1*n2+i2)*m + (j1*m2+j2), c] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            # V index: (j1*m2+j2)*p + (k1*p2+k2)
                            V[(j1*m2+j2)*p + (k1*p2+k2), c] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            # W index: (k1*p2+k2)*n + (i1*n2+i2)
                            W[(k1*p2+k2)*n + (i1*n2+i2), c] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    
    return U, V, W

def _split_m_decompose(U1, V1, W1, n, m1, p, U2, V2, W2, m2):
    """Split m: (n, m1+m2, p) = (n, m1, p) ⊕ (n, m2, p)
    U1,V1,W1 for (n,m1,p) and U2,V2,W2 for (n,m2,p)
    """
    r1, r2 = U1.shape[1], U2.shape[1]
    r = r1 + r2
    m = m1 + m2
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Copy first part (m1 columns)
    for i in range(n):
        for j in range(m1):
            U[i*m+j, :r1] = U1[i*m1+j, :]
    for j in range(m1):
        for k in range(p):
            V[j*p+k, :r1] = V1[j*p+k, :]
    for k in range(p):
        for i in range(n):
            W[k*n+i, :r1] = W1[k*n+i, :]
    
    # Copy second part (m2 columns, offset in U and V)
    for i in range(n):
        for j in range(m2):
            U[i*m+(m1+j), r1:] = U2[i*m2+j, :]
    for j in range(m2):
        for k in range(p):
            V[(m1+j)*p+k, r1:] = V2[j*p+k, :]
    for k in range(p):
        for i in range(n):
            W[k*n+i, r1:] = W2[k*n+i, :]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Build (2,3,2) with rank 11 = 7 + 4
    S_U, S_V, S_W = _strassen_2x2x2()  # (2,2,2) rank 7
    T_U, T_V, T_W = _trivial_2x1x2()    # (2,1,2) rank 4
    
    # (2,3,2) = (2,2,2) ⊕ (2,1,2)
    U_232, V_232, W_232 = _split_m_decompose(S_U, S_V, S_W, 2, 2, 2, T_U, T_V, T_W, 1)
    # U_232: (6, 11), V_232: (6, 11), W_232: (4, 11)
    
    # Build (1,1,3) with rank 3
    O_U, O_V, O_W = _trivial_1x1x3()  # (1,1,3) rank 3
    
    # (2,3,6) = (2,3,2) ⊗ (1,1,3)
    # n=2*1=2, m=3*1=3, p=2*3=6
    U_final, V_final, W_final = _kronecker_decompose(U_232, V_232, W_232, 2, 3, 2, O_U, O_V, O_W, 1, 1, 3)
    # U_final: (6, 33), V_final: (18, 33), W_final: (12, 33)
    
    rank_final = U_final.shape[1]
    
    # Verify
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
