# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_factors():
    """
    Returns list of 7 tuples (u, v, w) for 2x2x2 Strassen decomposition.
    u: 2x2 matrix for A
    v: 2x2 matrix for B
    w: 2x2 matrix for C
    """
    # P1 = (A00+A11)(B00+B11)
    u1 = np.array([[1, 0], [1, 1]])
    v1 = np.array([[1, 0], [1, 0]]) # Wait, (B00+B11) means B00 * 1 + B11 * 1.
    # v[i][j] is coeff of B[i][j].
    # B00 is B[0][0], B11 is B[1][1].
    # So v1[0][0]=1, v1[1][1]=1.
    v1 = np.array([[1, 0], [0, 1]])
    # w1: C00 += P1, C11 += P1
    w1 = np.array([[1, 0], [0, 1]])
    
    # P2 = (A10+A11)B00
    u2 = np.array([[0, 0], [1, 1]])
    v2 = np.array([[1, 0], [0, 0]])
    # w2: C10 += P2, C11 -= P2
    w2 = np.array([[0, 0], [1, -1]])
    
    # P3 = A00(B10-B11)
    u3 = np.array([[1, 0], [0, 0]])
    v3 = np.array([[0, 0], [1, -1]])
    # w3: C01 += P3, C11 += P3
    w3 = np.array([[0, 1], [0, 1]])
    
    # P4 = A11(B01-B00)
    u4 = np.array([[0, 0], [0, 1]])
    v4 = np.array([[0, 1], [1, 0]]) # B01 - B00 => B[0][1]*1 + B[0][0]*(-1)
    # v4[0][1]=1, v4[0][0]=-1.
    v4 = np.array([[-1, 1], [0, 0]])
    # w4: C00 += P4, C10 += P4
    w4 = np.array([[1, 0], [1, 0]])
    
    # P5 = (A00+A01)B11
    u5 = np.array([[1, 1], [0, 0]])
    v5 = np.array([[0, 0], [0, 1]])
    # w5: C00 -= P5, C01 += P5
    w5 = np.array([[-1, 1], [0, 0]])
    
    # P6 = (A10-A00)(B00+B10)
    u6 = np.array([[-1, 0], [1, 0]])
    v6 = np.array([[1, 0], [1, 0]]) # B00+B10 => B[0][0]*1 + B[1][0]*1
    # w6: C11 += P6
    w6 = np.array([[0, 0], [0, 1]])
    
    # P7 = (A01-A11)(B11+B01)
    u7 = np.array([[0, 1], [-1, 1]]) # A01 - A11 => A[0][1]*1 + A[1][1]*(-1)
    v7 = np.array([[0, 1], [0, 1]]) # B11+B01 => B[1][1]*1 + B[0][1]*1
    # w7: C00 += P7
    w7 = np.array([[1, 0], [0, 0]])
    
    return [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]

def _construct_3x4x4():
    n, m, p = 3, 4, 4
    # Rank 44
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen = _strassen_factors()
    
    # Part 1: A1 * B (Rank 28)
    # A1 is rows 0,1.
    # Split A1 cols into [0,1] and [2,3].
    # Split B rows into [0,1] and [2,3].
    # Split B cols into [0,1] and [2,3].
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Global A indices: i in 0,1; j in 0,1
    # Global B indices: j in 0,1; k in 0,1
    # Global C indices: k in 0,1; i in 0,1
    
    for l, (u, v, w) in enumerate(strassen):
        c_idx = l
        for i in range(2):
            for j in range(2):
                U[i * m + j, c_idx] = u[i, j]
        for j in range(2):
            for k in range(2):
                V[j * p + k, c_idx] = v[j, k]
        for k in range(2):
            for i in range(2):
                W[k * n + i, c_idx] = w[i, k] # Note: W index is k*n+i. w[i,k] is coeff for C[i,k].
                # Wait, w[i,k] corresponds to C[i][k].
                # W index is k*n + i.
                # So W[k*n+i] should be w[i,k].
                
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Global A indices: i in 0,1; j in 2,3 (local j 0,1 -> global 2,3)
    # Global B indices: j in 2,3 (local j 0,1 -> global 2,3); k in 0,1
    # Global C indices: k in 0,1; i in 0,1
    
    for l, (u, v, w) in enumerate(strassen):
        c_idx = 7 + l
        for i in range(2):
            for j in range(2):
                U[i * m + (2 + j), c_idx] = u[i, j]
        for j in range(2):
            for k in range(2):
                V[(2 + j) * p + k, c_idx] = v[j, k]
        for k in range(2):
            for i in range(2):
                W[k * n + i, c_idx] = w[i, k]

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Global A indices: i in 0,1; j in 0,1
    # Global B indices: j in 0,1; k in 2,3 (local k 0,1 -> global 2,3)
    # Global C indices: k in 2,3; i in 0,1
    
    for l, (u, v, w) in enumerate(strassen):
        c_idx = 14 + l
        for i in range(2):
            for j in range(2):
                U[i * m + j, c_idx] = u[i, j]
        for j in range(2):
            for k in range(2):
                V[j * p + (2 + k), c_idx] = v[j, k]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i, c_idx] = w[i, k]

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # Global A indices: i in 0,1; j in 2,3
    # Global B indices: j in 2,3; k in 2,3
    # Global C indices: k in 2,3; i in 0,1
    
    for l, (u, v, w) in enumerate(strassen):
        c_idx = 21 + l
        for i in range(2):
            for j in range(2):
                U[i * m + (2 + j), c_idx] = u[i, j]
        for j in range(2):
            for k in range(2):
                V[(2 + j) * p + (2 + k), c_idx] = v[j, k]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i, c_idx] = w[i, k]

    # Part 2: A2 * B (Rank 16)
    # A2 is row 2.
    # C2 = A2 * B.
    # C[2, j] = sum_k A[2, k] B[k, j]
    # Factors for each k, j.
    
    c_idx = 28
    for k in range(4):
        for j in range(4):
            U[2 * m + k, c_idx] = 1.0
            V[k * p + j, c_idx] = 1.0
            W[j * n + 2, c_idx] = 1.0
            c_idx += 1
            
    return U, V, W

def construct():
    n, m, p = 3, 4, 4
    U, V, W = _construct_3x4x4()
    
    # Verify exactness
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U_sb = np.zeros((n * m, rank))
        V_sb = np.zeros((m * p, rank))
        W_sb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
