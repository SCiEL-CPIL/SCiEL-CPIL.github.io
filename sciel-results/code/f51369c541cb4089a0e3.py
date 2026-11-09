import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 110.0
    
    # Use Laderman's rank-23 decomposition for 3x3x3, then extend to 3x3x6
    # by treating the 3x6 matrix as two 3x3 blocks and combining.
    
    # Laderman's 23-term decomposition for 3x3x3:
    # Indices: A[i,j] -> i*3+j, B[j,k] -> j*3+k, C[i,k] -> i*3+k
    # The 23 rank-1 terms are given by (u, v, w) triples of 9-element vectors.
    
    L23_terms = [
        # Term 0
        ([(0,1),(1,1)], [(0,1),(3,1)], [(0,1),(1,1)]),
        # Term 1
        ([(0,1),(4,1)], [(0,1),(6,1)], [(0,1),(3,1)]),
        # Term 2
        ([(1,1),(5,1)], [(1,1),(4,1)], [(1,1),(4,1)]),
        # Term 3
        ([(2,1),(6,1)], [(2,1),(7,1)], [(2,1),(6,1)]),
        # Term 4
        ([(0,1),(6,1)], [(0,1),(8,1)], [(0,1),(6,1)]),
        # Term 5
        ([(1,1),(7,1)], [(1,1),(5,1)], [(1,1),(5,1)]),
        # Term 6
        ([(2,1),(8,1)], [(2,1),(2,1)], [(2,1),(2,1)]),
        # Term 7
        ([(3,1),(4,1)], [(3,1),(0,1)], [(3,1),(0,1)]),
        # Term 8
        ([(3,1),(7,1)], [(3,1),(6,1)], [(3,1),(6,1)]),
        # Term 9
        ([(4,1),(8,1)], [(4,1),(3,1)], [(4,1),(3,1)]),
        # Term 10
        ([(5,1),(6,1)], [(5,1),(1,1)], [(5,1),(1,1)]),
        # Term 11
        ([(5,1),(8,1)], [(5,1),(7,1)], [(5,1),(7,1)]),
        # Term 12
        ([(6,1),(8,1)], [(6,1),(4,1)], [(6,1),(4,1)]),
        # Term 13
        ([(3,1)], [(1,1)], [(1,1)]),
        # Term 14
        ([(4,1)], [(4,1)], [(4,1)]),
        # Term 15
        ([(5,1)], [(7,1)], [(5,1)]),
        # Term 16
        ([(6,1)], [(0,1)], [(6,1)]),
        # Term 17
        ([(7,1)], [(3,1)], [(7,1)]),
        # Term 18
        ([(8,1)], [(6,1)], [(8,1)]),
        # Term 19
        ([(1,1)], [(2,1)], [(2,1)]),
        # Term 20
        ([(2,1)], [(5,1)], [(5,1)]),
        # Term 21
        ([(0,1)], [(7,1)], [(7,1)]),
        # Term 22
        ([(8,1)], [(8,1)], [(8,1)]),
    ]
    
    def make_vec(coords, size):
        v = [0.0] * size
        for idx, val in coords:
            v[idx] = val
        return v
    
    # Build 3x3x3 matrices
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    for r, (uc, vc, wc) in enumerate(L23_terms):
        U3[:, r] = make_vec(uc, 9)
        V3[:, r] = make_vec(vc, 9)
        W3[:, r] = make_vec(wc, 9)
    
    # Verify 3x3x3 decomposition
    T3 = np.einsum('ar,br,cr->abc', U3, V3, W3)
    T3_target = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T3_target[i*3+j, j*3+k, i*3+k] = 1.0
    
    if not np.allclose(T3, T3_target, atol=1e-9):
        # Fall back to trivial
        U_triv = np.zeros((9, 54))
        V_triv = np.zeros((18, 54))
        W_triv = np.zeros((18, 54))
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    col = i*18 + j*6 + k
                    U_triv[i*3+j, col] = 1
                    V_triv[j*6+k, col] = 1
                    W_triv[k*3+i, col] = 1
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
    
    # Build 3x3x6 from two 3x3x3 blocks
    # A is 3x3 (9 elements), B is 3x6 (18 elements), C is 6x3 (18 elements)
    # B[j,k] for j in 0..2, k in 0..5
    # C[k,i] for k in 0..5, i in 0..2
    # Split B into two 3x3 blocks: B[:,0:3] and B[:,3:6]
    # C is split into C[0:3,:] and C[3:6,:]
    
    U36 = np.zeros((9, 46))
    V36 = np.zeros((18, 46))
    W36 = np.zeros((18, 46))
    
    # Block 1: B[:,0:3] and C[0:3,:]
    U36[:, 0:23] = U3
    V36[0:9, 0:23] = V3
    W36[0:9, 0:23] = W3
    
    # Block 2: B[:,3:6] and C[3:6,:]
    U36[:, 23:46] = U3
    V36[9:18, 23:46] = V3
    W36[9:18, 23:46] = W3
    
    # Verify 3x3x6 decomposition
    T36 = np.einsum('ar,br,cr->abc', U36, V36, W36)
    T36_target = np.zeros((9, 18, 18))
    for i in range(3):
        for j in range(3):
            for k in range(6):
                T36_target[i*3+j, j*6+k, k*3+i] = 1.0
    
    if np.allclose(T36, T36_target, atol=1e-9):
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    # Final fallback: trivial rank-54 solution
    U_triv = np.zeros((9, 54))
    V_triv = np.zeros((18, 54))
    W_triv = np.zeros((18, 54))
    for i in range(3):
        for j in range(3):
            for k in range(6):
                col = i*18 + j*6 + k
                U_triv[i*3+j, col] = 1
                V_triv[j*6+k, col] = 1
                W_triv[k*3+i, col] = 1
    return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

# EVOLVE-BLOCK-END
