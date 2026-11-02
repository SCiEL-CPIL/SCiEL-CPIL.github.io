import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _get_strassen_4x4():
    """
    Returns U, V, W for 4x4 matrix multiplication using Strassen's algorithm.
    U: 16x49, V: 16x49, W: 16x49
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _get_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Coefficients for the 7 high-level Strassen factors
    # cA[h][br][bc]
    cA = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1, (1,1): 1 }, # M2
        { (0,0): 1 },           # M3
        { (1,1): 1 },           # M4
        { (0,0): 1, (0,1): 1 }, # M5
        { (1,0): 1, (0,0): -1 },# M6
        { (0,1): 1, (1,1): -1 } # M7
    ]
    
    # cB[h][br][bc]
    cB = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (0,0): 1 },           # M2
        { (0,1): 1, (1,1): -1 },# M3
        { (1,0): 1, (0,0): -1 },# M4
        { (1,1): 1 },           # M5
        { (0,0): 1, (0,1): 1 }, # M6
        { (1,0): 1, (1,1): 1 }  # M7
    ]
    
    # cC[h][br][bc]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    cC = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1, (1,1): -1 },# M2
        { (0,1): 1, (1,1): 1 }, # M3
        { (0,0): 1, (1,0): 1 }, # M4
        { (0,1): 1, (1,1): -1 },# M5
        { (1,1): 1 },           # M6
        { (0,0): 1, (1,1): 1 }  # M7
    ]
    
    for h in range(7):
        for l in range(7):
            r = 7 * h + l
            
            # Fill U
            for (br, bc), coeff_a in cA[h].items():
                for i_local in range(2):
                    for j_local in range(2):
                        u_idx = 4 * (br * 2 + i_local) + (bc * 2 + j_local)
                        u2_idx = 2 * i_local + j_local
                        U[u_idx, r] += coeff_a * U2[u2_idx, l]
                        
            # Fill V
            for (br, bc), coeff_b in cB[h].items():
                for i_local in range(2):
                    for j_local in range(2):
                        # V index is j*4+k. 
                        # Block (br, bc) of B. 
                        # Row j = br*2 + i_local? No, B rows are j.
                        # In Strassen 2x2 on blocks, B block (br, bc) corresponds to B[j, k] where j is row, k is col.
                        # The block B_br_bc contains B[br*2+i, bc*2+j].
                        # So row j = br*2 + i_local, col k = bc*2 + j_local.
                        v_idx = 4 * (br * 2 + i_local) + (bc * 2 + j_local)
                        v2_idx = 2 * i_local + j_local
                        V[v_idx, r] += coeff_b * V2[v2_idx, l]
                        
            # Fill W
            for (br, bc), coeff_c in cC[h].items():
                for i_local in range(2):
                    for j_local in range(2):
                        # W index is k*4+i.
                        # Block (br, bc) of C.
                        # C[br*2+i, bc*2+j].
                        # Row i = br*2 + i_local, Col k = bc*2 + j_local.
                        w_idx = 4 * (bc * 2 + j_local) + (br * 2 + i_local)
                        w2_idx = 2 * i_local + j_local
                        W[w_idx, r] += coeff_c * W2[w2_idx, l]
                        
    return U, V, W

def _construct_strassen_4x8():
    """
    Constructs the rank-98 decomposition for 4x4 * 4x8.
    """
    U44, V44, W44 = _get_strassen_4x4()
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Copy U
    U[:, 0:49] = U44
    U[:, 49:98] = U44
    
    # Fill V
    # V index: j*p + k. j in 0..3, k in 0..7.
    # For first 49 terms (B1, k=0..3):
    for j in range(4):
        for k in range(4):
            v_idx = j * 8 + k
            v44_idx = j * 4 + k
            V[v_idx, 0:49] = V44[v44_idx, 0:49]
            
    # For last 49 terms (B2, k=4..7):
    for j in range(4):
        for k in range(4, 8):
            v_idx = j * 8 + k
            v44_idx = j * 4 + (k - 4)
            V[v_idx, 49:98] = V44[v44_idx, 0:49]
            
    # Fill W
    # W index: k*n + i. k in 0..7, i in 0..3.
    # For first 49 terms (C1, k=0..3):
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            w44_idx = k * 4 + i
            W[w_idx, 0:49] = W44[w44_idx, 0:49]
            
    # For last 49 terms (C2, k=4..7):
    for k in range(4, 8):
        for i in range(4):
            w_idx = k * 4 + i
            w44_idx = (k - 4) * 4 + i
            W[w_idx, 49:98] = W44[w44_idx, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    # Use the Strassen 4x8 decomposition
    return _construct_strassen_4x8()

