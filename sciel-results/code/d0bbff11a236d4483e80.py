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
    
    # Build the target tensor T of shape (9, 9, 9)
    T = np.zeros((9, 9, 9), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                T[idx_a, idx_b, idx_c] = 1

    # Reconstruct T from U, V, W
    # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # Using einsum
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check if they match
    return np.allclose(T, T_recon)

def _construct_rank_22():
    """
    Construct a rank-22 decomposition for 3x3 matrix multiplication
    using block decomposition (2x2 + 1x1) and Strassen's algorithm for the 2x2 block.
    """
    # Indices:
    # A: 00, 01, 02, 10, 11, 12, 20, 21, 22
    # B: 00, 01, 02, 10, 11, 12, 20, 21, 22
    # C: 00, 01, 02, 10, 11, 12, 20, 21, 22
    
    # We will store the terms as lists of (index, value) for U, V, W
    terms = []
    
    # Helper to add a term
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
    # A22: 00, 01, 10, 11
    # B22: 00, 01, 10, 11
    # C22: 00, 01, 10, 11
    
    # P1 = A00 * (B00 + B11)
    # C01 += P1, C11 += P1
    add_term([(0, 1)], [(0, 1), (1, 1)], [(1, 1), (3, 1)])
    
    # P2 = (A00 + A01) * B11
    # C00 -= P2, C11 -= P2
    add_term([(0, 1), (1, 1)], [(1, 1)], [(0, -1), (3, -1)])
    
    # P3 = (A10 + A11) * B00
    # C10 += P3, C11 -= P3
    add_term([(3, 1), (4, 1)], [(0, 1)], [(2, 1), (3, -1)])
    
    # P4 = A11 * (B10 + B00)
    # C00 += P4
    add_term([(4, 1)], [(2, 1), (0, 1)], [(0, 1)])
    
    # P5 = (A00 + A11) * (B00 + B11)
    # C00 += P5, C11 += P5
    add_term([(0, 1), (4, 1)], [(0, 1), (1, 1)], [(0, 1), (3, 1)])
    
    # P6 = (A00 - A10) * (B10 + B11)
    # C10 += P6, C11 -= P6
    add_term([(0, 1), (3, -1)], [(2, 1), (1, 1)], [(2, 1), (3, -1)])
    
    # P7 = (A01 - A11) * (B00 + B10)
    # C01 += P7, C00 -= P7
    add_term([(1, 1), (4, -1)], [(0, 1), (2, 1)], [(1, 1), (0, -1)])

    # Cross terms: A22 * d -> g (C02, C12)
    # d = B[0:2, 2] -> B02 (2), B12 (5)
    # A22 = A[0:2, 0:2] -> A00 (0), A01 (1), A10 (3), A11 (4)
    
    # A00 * B02 -> C02 (2)
    add_term([(0, 1)], [(2, 1)], [(2, 1)])
    
    # A01 * B12 -> C02 (2)
    add_term([(1, 1)], [(5, 1)], [(2, 1)])
    
    # A10 * B02 -> C12 (5)
    add_term([(3, 1)], [(2, 1)], [(5, 1)])
    
    # A11 * B12 -> C12 (5)
    add_term([(4, 1)], [(5, 1)], [(5, 1)])

    # a * f -> g
    # a = A[0:2, 2] -> A02 (2), A12 (5)
    # f = B[2, 2] -> B22 (8)
    
    # A02 * B22 -> C02 (2)
    add_term([(2, 1)], [(8, 1)], [(2, 1)])
    
    # A12 * B22 -> C12 (5)
    add_term([(5, 1)], [(8, 1)], [(5, 1)])

    # bT * B22 -> hT (C20, C21)
    # bT = A[2, 0:2] -> A20 (6), A21 (7)
    # B22 = B[0:2, 0:2] -> B00 (0), B01 (1), B10 (2), B11 (3)
    
    # A20 * B00 -> C20 (6)
    add_term([(6, 1)], [(0, 1)], [(6, 1)])
    
    # A21 * B10 -> C20 (6)
    add_term([(7, 1)], [(2, 1)], [(6, 1)])
    
    # A20 * B01 -> C21 (7)
    add_term([(6, 1)], [(1, 1)], [(7, 1)])
    
    # A21 * B11 -> C21 (7)
    add_term([(7, 1)], [(3, 1)], [(7, 1)])

    # c * eT -> hT
    # c = A[2, 2] -> A22 (8)
    # eT = B[2, 0:2] -> B20 (6), B21 (7)
    
    # A22 * B20 -> C20 (6)
    add_term([(8, 1)], [(6, 1)], [(6, 1)])
    
    # A22 * B21 -> C21 (7)
    add_term([(8, 1)], [(7, 1)], [(7, 1)])

    # bT * d -> i (C22)
    # bT = A20 (6), A21 (7)
    # d = B02 (2), B12 (5)
    
    # A20 * B02 -> C22 (8)
    add_term([(6, 1)], [(2, 1)], [(8, 1)])
    
    # A21 * B12 -> C22 (8)
    add_term([(7, 1)], [(5, 1)], [(8, 1)])

    # c * f -> i
    # A22 (8) * B22 (8) -> C22 (8)
    add_term([(8, 1)], [(8, 1)], [(8, 1)])

    # Build matrices
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
