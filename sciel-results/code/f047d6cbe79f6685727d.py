import numpy as np
import time

def _strassen_2x2():
    """Returns U, V, W for 2x2 Strassen (Rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4 = A11(B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    # M5 = (A00 + A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A01 - A00)(B00 + B10)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    # M7 = (A11 - A01)(B01 + B11)
    U[3, 6] = 1; U[1, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    return U, V, W

def _schoolbook_3x3():
    """Returns U, V, W for 3x3 Schoolbook (Rank 27)."""
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3 + j, r] = 1
                V[j*3 + k, r] = 1
                W[k*3 + i, r] = 1
                r += 1
    return U, V, W

def _kronecker_product(U1, V1, W1, U2, V2, W2):
    """
    Computes the Kronecker product of two tensor decompositions.
    If M1 is (n1, m1, p1) and M2 is (n2, m2, p2),
    the result is M1 x M2 which is (n1*n2, m1*m2, p1*p2).
    Rank is R1 * R2.
    """
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    R = r1 * r2
    N = n1 * n2
    
    U = np.zeros((N, R))
    V = np.zeros((N, R))
    W = np.zeros((N, R))
    
    # Note: For matrix multiplication, U, V, W shapes are (n*m, R), (m*p, R), (p*n, R)
    # The Kronecker product of two MM tensors M_A and M_B gives M_{A \otimes B}.
    # The dimensions are:
    # A: (n1, m1, p1), B: (n2, m2, p2)
    # Result: (n1*n2, m1*m2, p1*p2)
    # U index: (i1, i2) where i1 in 0..n1-1, i2 in 0..n2-1? 
    # Wait, the input U for MM is indexed by (i, j) where i is row of A, j is col of A.
    # So U shape is (n*m, R).
    # In Kronecker, U_new[(i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)]?
    # No, the tensor T[i, j, k] = A[i, j] B[j, k].
    # The Kronecker product of tensors T1 and T2 is T1[i1, j1, k1] * T2[i2, j2, k2].
    # The resulting tensor is T[i1, i2, j1, j2, k1, k2].
    # This corresponds to multiplying (A1 \otimes A2) by (B1 \otimes B2).
    # A1 is n1 x m1, A2 is n2 x m2.
    # A1 \otimes A2 is (n1*n2) x (m1*m2).
    # The index for A1 \otimes A2 is (i1, i2) for rows, (j1, j2) for cols.
    # Standard Kronecker product ordering: (i1, i2) -> i1 * n2 + i2.
    # So U_new row index = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2).
    # This is consistent with the "block" structure where A1 is the "outer" block.
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            for i1 in range(n1 // 3 if n1 % 3 == 0 else n1): # This logic is flawed, n1 is n*m.
                pass
            
    # Let's use the explicit loop from the previous code but generalized.
    # U1 shape (n1*m1, r1). U2 shape (n2*m2, r2).
    # This assumes U1 is for a matrix of size n1 x m1?
    # In our case, U1 is for 2x2, so n1=2, m1=2. U1 shape (4, 7).
    # U2 is for 3x3, so n2=3, m2=3. U2 shape (9, 27).
    # Result is 6x6. n=6, m=6.
    # U_new shape (36, 189).
    
    n1, m1 = int(np.sqrt(n1)), int(np.sqrt(n1)) # This is wrong if n1 != m1.
    # But for square matrices, n1 == m1.
    # Let's assume square matrices for simplicity, as in the task.
    # n1 = dim1, n2 = dim2.
    # U1 is (n1*n1, r1). U2 is (n2*n2, r2).
    
    n1 = int(np.sqrt(U1.shape[0]))
    n2 = int(np.sqrt(U2.shape[0]))
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            for i1 in range(n1):
                for i2 in range(n2):
                    for j1 in range(n1):
                        for j2 in range(n2):
                            idx = (i1 * n2 + i2) * (n1 * n2) + (j1 * n2 + j2)
                            U[idx, r] = U1[i1 * n1 + j1, r1_idx] * U2[i2 * n2 + j2, r2_idx]
                            V[idx, r] = V1[i1 * n1 + j1, r1_idx] * V2[i2 * n2 + j2, r2_idx]
                            W[idx, r] = W1[i1 * n1 + j1, r1_idx] * W2[i2 * n2 + j2, r2_idx]
    return U, V, W

def _verify(n, m, p, U, V, W, tol=1e-6):
    """Verify the decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    np.random.seed(42)
    for _ in range(150):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0
    
    # Use Kronecker product of 2x2 Strassen and 3x3 Schoolbook
    U2, V2, W2 = _strassen_2x2()
    U3, V3, W3 = _schoolbook_3x3()
    
    U, V, W = _kronecker_product(U2, V2, W2, U3, V3, W3)
    
    if _verify(6, 6, 6, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook 216
    U_sb = np.zeros((36, 216))
    V_sb = np.zeros((36, 216))
    W_sb = np.zeros((36, 216))
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U_sb[i*6+j, col] = 1
                V_sb[j*6+k, col] = 1
                W_sb[k*6+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
