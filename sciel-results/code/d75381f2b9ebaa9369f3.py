"""Program implementing a rank-55 exact tensor decomposition for MM_{2,3,10}."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-55 exact decomposition of the 2x3 * 3x10 matrix multiplication tensor.
    Strategy:
      - Decompose the 2x3 * 3x10 problem into two independent parts:
          1. 2x2 * 2x10 part (rows 0,1 of A; cols 0,1 of B)
          2. 2x1 * 1x10 part (row 2 of A; cols 2 of B)
      - Use Strassen's algorithm (rank 7) for each 2x2 * 2x2 block of the first part.
        There are 5 such blocks (columns 0-1, 2-3, 4-5, 6-7, 8-9 of B).
        Total rank for part 1: 5 * 7 = 35.
      - Use schoolbook for the 2x1 * 1x10 part.
        Total rank for part 2: 2 * 1 * 10 = 20.
      - Total rank: 35 + 20 = 55.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # We'll build U, V, W as lists of lists (or numpy arrays) with rank columns.
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    # Strassen 2x2x2 decomposition terms.
    # For a 2x2 * 2x2 multiplication:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # Strassen terms (M1..M7):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b00)
    # M5 = (a00+a10)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # 
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # For each term, we have a product of a linear combination of A entries and a linear combination of B entries.
    # In the tensor decomposition, each term corresponds to a rank-1 term in the multiplication tensor.
    # The U, V, W matrices encode the coefficients.
    
    # For a single Strassen term in a 2x2*2x2 block at column offset k0 (0, 2, 4, 6, 8):
    # The A indices are (i, j) where i in {0,1}, j in {0,1}
    # The B indices are (j, k) where j in {0,1}, k in {k0, k0+1}
    # The C indices are (i, k) where i in {0,1}, k in {k0, k0+1}
    
    # Global indices:
    # A: row = i, col = j -> index in U: i*m + j = i*3 + j
    # B: row = j, col = k -> index in V: j*p + k = j*10 + k
    # C: row = i, col = k -> index in W: k*n + i = k*2 + i (trace convention)
    
    # Strassen coefficients:
    # Each term t has:
    #   U coefficients for A entries
    #   V coefficients for B entries
    #   W coefficients for C entries (with signs)
    
    # Let's define the 7 Strassen terms with their (U, V, W) contributions.
    # For a block at column offset k0:
    # A entries: (0,0), (0,1), (1,0), (1,1) -> U indices: 0, 1, 3, 4
    # B entries: (0,k0), (0,k0+1), (1,k0), (1,k0+1) -> V indices: k0, k0+1, 10+k0, 10+k0+1
    # C entries: (0,k0), (0,k0+1), (1,k0), (1,k0+1) -> W indices: 2*k0, 2*k0+1, 2*k0+2, 2*k0+3
    
    # Term 1: M1 = (a00+a11)(b00+b11)
    # U: {0:1, 4:1}, V: {k0:1, 10+k0:1}, W: {2*k0:1, 2*k0+3:1}
    # Term 2: M2 = (a10+a11)(b00)
    # U: {3:1, 4:1}, V: {k0:1}, W: {2*k0+2:1}
    # Term 3: M3 = a00(b10-b11)
    # U: {0:1}, V: {10+k0:1, 10+k0+1:-1}, W: {2*k0+1:1}
    # Term 4: M4 = a11(b01-b00)
    # U: {4:1}, V: {k0+1:1, k0:-1}, W: {2*k0:1, 2*k0+2:1}
    # Term 5: M5 = (a00+a10)(b11)
    # U: {0:1, 3:1}, V: {10+k0+1:1}, W: {2*k0:-1, 2*k0+1:1}
    # Term 6: M6 = (a10-a00)(b00+b01)
    # U: {3:1, 0:-1}, V: {k0:1, k0+1:1}, W: {2*k0+2:1}
    # Term 7: M7 = (a01-a11)(b10+b11)
    # U: {1:1, 4:-1}, V: {10+k0:1, 10+k0+1:1}, W: {2*k0:-1, 2*k0+3:1}
    
    def get_strassen_terms(k0):
        """Return the 7 Strassen terms for a 2x2*2x2 block at column offset k0."""
        terms = []
        # Term 1
        terms.append(({0:1, 4:1}, {k0:1, 10+k0:1}, {2*k0:1, 2*k0+3:1}))
        # Term 2
        terms.append(({3:1, 4:1}, {k0:1}, {2*k0+2:1}))
        # Term 3
        terms.append(({0:1}, {10+k0:1, 10+k0+1:-1}, {2*k0+1:1}))
        # Term 4
        terms.append(({4:1}, {k0+1:1, k0:-1}, {2*k0:1, 2*k0+2:1}))
        # Term 5
        terms.append(({0:1, 3:1}, {10+k0+1:1}, {2*k0:-1, 2*k0+1:1}))
        # Term 6
        terms.append(({3:1, 0:-1}, {k0:1, k0+1:1}, {2*k0+2:1}))
        # Term 7
        terms.append(({1:1, 4:-1}, {10+k0:1, 10+k0+1:1}, {2*k0:-1, 2*k0+3:1}))
        return terms
    
    rank = 55
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    
    col = 0
    # Part 1: 5 Strassen blocks for columns 0-1, 2-3, 4-5, 6-7, 8-9
    for k0 in [0, 2, 4, 6, 8]:
        terms = get_strassen_terms(k0)
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx, col] = val
            for idx, val in v_dict.items():
                V[idx, col] = val
            for idx, val in w_dict.items():
                W[idx, col] = val
            col += 1
    
    # Part 2: Schoolbook for 2x1 * 1x10
    # A row 2 (index 2 in the 3 rows), B col 2 (index 2 in the 3 cols)
    # For each i in {0,1} and k in {0,...,9}:
    #   A entry: (i, 2) -> U index: i*3 + 2 = 2 + i
    #   B entry: (2, k) -> V index: 2*10 + k = 20 + k
    #   C entry: (i, k) -> W index: k*2 + i
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to lists of lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
