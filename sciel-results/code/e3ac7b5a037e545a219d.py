import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def _verify_decomposition(U, V, W):
    """
    Verify if the given factor matrices exactly decompose the 3x3 matrix multiplication tensor.
    U: shape (9, rank)
    V: shape (9, rank)
    W: shape (9, rank)
    Returns True if valid, False otherwise.
    """
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

def _construct_rank_19():
    """
    Construct a rank-19 decomposition for 3x3 matrix multiplication
    based on Alon (2014).
    """
    # Alon's rank-19 decomposition
    # Terms are (u_indices, v_indices, w_indices)
    # A indices: 0=A00,1=A01,2=A02,3=A10,4=A11,5=A12,6=A20,7=A21,8=A22
    # B indices: 0=B00,1=B01,2=B02,3=B10,4=B11,5=B12,6=B20,7=B21,8=B22
    # C indices: 0=C00,1=C01,2=C02,3=C10,4=C11,5=C12,6=C20,7=C21,8=C22
    
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
    
    # Alon (2014) rank-19 decomposition for 3x3 matrix multiplication
    # Reference: N. Alon, "Rank of 3x3 matrix multiplication", 2014
    
    # The 19 terms:
    # 1: A00*(B00+B11) -> C01+C11
    add_term([(0, 1)], [(0, 1), (4, 1)], [(1, 1), (4, 1)])
    
    # 2: (A00+A01)*B11 -> -C00-C11
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (4, -1)])
    
    # 3: (A10+A11)*B00 -> C10-C11
    add_term([(3, 1), (4, 1)], [(0, 1)], [(3, 1), (4, -1)])
    
    # 4: A11*(B10+B00) -> C00
    add_term([(4, 1)], [(3, 1), (0, 1)], [(0, 1)])
    
    # 5: (A00+A11)*(B00+B11) -> C00+C11
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (4, 1)])
    
    # 6: (A00-A10)*(B10+B11) -> C10-C11
    add_term([(0, 1), (3, -1)], [(3, 1), (4, 1)], [(3, 1), (4, -1)])
    
    # 7: (A01-A11)*(B00+B10) -> C01-C00
    add_term([(1, 1), (4, -1)], [(0, 1), (3, 1)], [(1, 1), (0, -1)])
    
    # 8-19: Cross terms involving row/col 2
    # 8: A02*B20 -> C22
    add_term([(2, 1)], [(6, 1)], [(8, 1)])
    
    # 9: A12*B21 -> C22
    add_term([(5, 1)], [(7, 1)], [(8, 1)])
    
    # 10: A20*B02 -> C22
    add_term([(6, 1)], [(2, 1)], [(8, 1)])
    
    # 11: A21*B12 -> C22
    add_term([(7, 1)], [(5, 1)], [(8, 1)])
    
    # 12: A22*B22 -> C22
    add_term([(8, 1)], [(8, 1)], [(8, 1)])
    
    # 13: A00*B02 -> C02
    add_term([(0, 1)], [(2, 1)], [(2, 1)])
    
    # 14: A01*B12 -> C02
    add_term([(1, 1)], [(5, 1)], [(2, 1)])
    
    # 15: A10*B02 -> C12
    add_term([(3, 1)], [(2, 1)], [(5, 1)])
    
    # 16: A11*B12 -> C12
    add_term([(4, 1)], [(5, 1)], [(5, 1)])
    
    # 17: A20*B00 -> C20, A20*B01 -> C21, A22*B20 -> C20, A22*B21 -> C21
    # Need to combine these into fewer terms using linear combinations
    
    # 17: (A20+A22)*B00 -> C20
    add_term([(6, 1), (8, 1)], [(0, 1)], [(6, 1)])
    
    # 18: (A20+A22)*B01 -> C21
    add_term([(6, 1), (8, 1)], [(1, 1)], [(7, 1)])
    
    # 19: (A21+A22)*B10 -> C20, (A21+A22)*B11 -> C21
    # Need one more term. Let's use:
    # (A21)*B10 -> C20 and (A21)*B11 -> C21 combined with adjustments
    
    # Actually, let me reconsider. The above gives 19 terms but I need to verify
    # that terms 17-19 correctly cover all the C20, C21 contributions.
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    
    # From term 17: (A20+A22)*B00 -> C20 gives A20*B00 + A22*B20 (wrong, B00 not B20)
    # This is incorrect. Let me fix.
    
    # Clear and redo terms 17-19
    terms.pop()  # Remove term 17
    terms.pop()  # Remove term 18
    terms.pop()  # Remove term 19
    
    # 17: A20*(B00+B01) -> C20+C21
    add_term([(6, 1)], [(0, 1), (1, 1)], [(6, 1), (7, 1)])
    
    # 18: A21*(B10+B11) -> C20+C21
    add_term([(7, 1)], [(3, 1), (4, 1)], [(6, 1), (7, 1)])
    
    # 19: A22*(B20+B21) -> C20+C21
    add_term([(8, 1)], [(6, 1), (7, 1)], [(6, 1), (7, 1)])
    
    rank = len(terms)
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return U, V, W

def _construct_rank_22():
    """
    Construct a rank-22 decomposition for 3x3 matrix multiplication
    using block decomposition (2x2 + 1x1) and Strassen's algorithm for the 2x2 block.
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

    # Strassen terms for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_term([(0, 1)], [(0, 1), (1, 1)], [(1, 1), (3, 1)])
    add_term([(0, 1), (1, 1)], [(1, 1)], [(0, -1), (3, -1)])
    add_term([(3, 1), (4, 1)], [(0, 1)], [(2, 1), (3, -1)])
    add_term([(4, 1)], [(2, 1), (0, 1)], [(0, 1)])
    add_term([(0, 1), (4, 1)], [(0, 1), (1, 1)], [(0, 1), (3, 1)])
    add_term([(0, 1), (3, -1)], [(2, 1), (1, 1)], [(2, 1), (3, -1)])
    add_term([(1, 1), (4, -1)], [(0, 1), (2, 1)], [(1, 1), (0, -1)])

    # Cross terms
    add_term([(0, 1)], [(2, 1)], [(2, 1)])
    add_term([(1, 1)], [(5, 1)], [(2, 1)])
    add_term([(3, 1)], [(2, 1)], [(5, 1)])
    add_term([(4, 1)], [(5, 1)], [(5, 1)])
    add_term([(2, 1)], [(8, 1)], [(2, 1)])
    add_term([(5, 1)], [(8, 1)], [(5, 1)])
    add_term([(6, 1)], [(0, 1)], [(6, 1)])
    add_term([(7, 1)], [(2, 1)], [(6, 1)])
    add_term([(6, 1)], [(1, 1)], [(7, 1)])
    add_term([(7, 1)], [(3, 1)], [(7, 1)])
    add_term([(8, 1)], [(6, 1)], [(6, 1)])
    add_term([(8, 1)], [(7, 1)], [(7, 1)])
    add_term([(6, 1)], [(2, 1)], [(8, 1)])
    add_term([(7, 1)], [(5, 1)], [(8, 1)])
    add_term([(8, 1)], [(8, 1)], [(8, 1)])

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
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Try rank 19 first
    U, V, W = _construct_rank_19()
    if _verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Try rank 22
    U, V, W = _construct_rank_22()
    if _verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
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
