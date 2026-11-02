import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _verify_decomposition(U, V, W):
    n, m, p = 3, 3, 3
    rank = U.shape[1]
    
    T = np.zeros((9, 9, 9), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                T[idx_a, idx_b, idx_c] = 1

    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_recon)

def _construct_rank_22():
    """
    Rank-22 decomposition for 3x3 matrix multiplication.
    Uses Strassen for the 2x2 block plus cross terms.
    """
    terms = []
    
    def add_term(u_indices, v_indices, w_indices):
        u = [0]*9
        v = [0]*9
        w = [0]*9
        for i, val in u_indices:
            u[i] += val
        for i, val in v_indices:
            v[i] += val
        for i, val in w_indices:
            w[i] += val
        terms.append((u, v, w))

    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Using standard Strassen with correct indexing:
    # A22 = [[A00, A01], [A10, A11]] -> indices 0,1,3,4
    # B22 = [[B00, B01], [B10, B11]] -> indices 0,1,2,3
    # C22 = [[C00, C01], [C10, C11]] -> indices 0,1,2,3
    
    # M1 = (A00 + A11)(B00 + B11)
    add_term([(0,1),(4,1)], [(0,1),(3,1)], [(0,1),(3,1)])
    
    # M2 = (A10 + A11)B00
    add_term([(3,1),(4,1)], [(0,1)], [(2,1),(3,1)])
    
    # M3 = A00(B00 - B10)
    add_term([(0,1)], [(0,1),(2,-1)], [(1,1),(3,1)])
    
    # M4 = A11(B10 - B00)
    add_term([(4,1)], [(2,1),(0,-1)], [(0,1),(3,1)])
    
    # M5 = (A00 + A01)B11
    add_term([(0,1),(1,1)], [(3,1)], [(0,-1),(3,-1)])
    
    # M6 = (A10 - A00)(B00 + B10)
    add_term([(3,1),(0,-1)], [(0,1),(2,1)], [(2,1),(3,-1)])
    
    # M7 = (A01 - A11)(B10 + B11)
    add_term([(1,1),(4,-1)], [(2,1),(3,1)], [(1,1),(0,-1)])

    # Cross terms: A[0:2,0:2] * B[0:2,2] -> C[0:2,2]
    # A00*B02 -> C02
    add_term([(0,1)], [(2,1)], [(2,1)])
    # A01*B12 -> C02
    add_term([(1,1)], [(5,1)], [(2,1)])
    # A10*B02 -> C12
    add_term([(3,1)], [(2,1)], [(5,1)])
    # A11*B12 -> C12
    add_term([(4,1)], [(5,1)], [(5,1)])

    # A[0:2,2] * B[2,2] -> C[0:2,2]
    add_term([(2,1)], [(8,1)], [(2,1)])
    add_term([(5,1)], [(8,1)], [(5,1)])

    # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    add_term([(6,1)], [(0,1)], [(6,1)])
    add_term([(7,1)], [(2,1)], [(6,1)])
    add_term([(6,1)], [(1,1)], [(7,1)])
    add_term([(7,1)], [(3,1)], [(7,1)])

    # A[2,2] * B[2,0:2] -> C[2,0:2]
    add_term([(8,1)], [(6,1)], [(6,1)])
    add_term([(8,1)], [(7,1)], [(7,1)])

    # A[2,0:2] * B[0:2,2] -> C[2,2]
    add_term([(6,1)], [(2,1)], [(8,1)])
    add_term([(7,1)], [(5,1)], [(8,1)])

    # A[2,2] * B[2,2] -> C[2,2]
    add_term([(8,1)], [(8,1)], [(8,1)])

    rank = len(terms)
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    # Try rank 22
    U, V, W = _construct_rank_22()
    if _verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 27)
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    column = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx_a = i * 3 + j
                idx_b = j * 3 + k
                idx_c = k * 3 + i
                U[idx_a, column] = 1
                V[idx_b, column] = 1
                W[idx_c, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
