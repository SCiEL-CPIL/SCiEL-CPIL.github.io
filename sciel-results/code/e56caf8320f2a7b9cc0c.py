import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The multiplication C = A @ B where A is 2x2, B is 2x14, C is 2x14.
    The tensor has dimensions n*m*p = 2*2*14 = 56.
    
    We can view this as 7 independent 2x2 * 2x2 sub-problems (one for each pair of columns of B).
    Each sub-problem has tensor rank 7 (Strassen's algorithm).
    Since the A factors are shared but the B and C factors are independent per column-pair,
    the total rank is 7 * 7 = 49.
    
    This is minimal because:
    - The rank of the 2x2 * 2x2 tensor is exactly 7 (proven by Strassen, and lower bound by 
      various results - the rank is at least 7 for 2x2 matrix multiplication).
    - The 7 column-pairs are independent in the sense that no linear combination of the 
      Strassen terms for one pair can cancel terms from another pair due to the support 
      structure in the B and C indices.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = DIMENSIONS
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7 blocks of 2 columns each

    # Strassen terms for 2x2 * 2x2
    # Indexing: A[i,j] -> i*m+j, B[j,k] -> j*2+k (within 2x2 block), C[i,k] -> i*2+k (within 2x2 block)
    # Strassen's 7 products for A @ B = C:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6
    
    # Each term: (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
    # indices are local to the 2x2 block
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11) -> contributes to C00 (+1), C11 (+1)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2 = (A10 + A11) * B00 -> contributes to C10 (+1), C11 (-1)
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # M3 = A00 * (B01 - B11) -> contributes to C01 (+1), C11 (+1)
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # M4 = A11 * (B10 - B00) -> contributes to C00 (+1), C10 (+1)
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # M5 = (A00 + A01) * B11 -> contributes to C00 (-1), C01 (+1)
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # M6 = (A10 - A00) * (B00 + B01) -> contributes to C11 (+1)
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7 = (A01 - A11) * (B10 + B11) -> contributes to C00 (+1)
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]

    rank = num_blocks * 7  # 49
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            a_idx_list, a_val_list, b_idx_list, b_val_list, c_idx_list, c_val_list = strassen_terms[s]

            # Fill U (A indices): a_idx is i*m + j where i,j in {0,1}
            for a_idx, coeff in zip(a_idx_list, a_val_list):
                U[a_idx, col] += coeff

            # Fill V (B indices): local (j, k) -> global row j, col 2*t + k
            for b_idx, coeff in zip(b_idx_list, b_val_list):
                b_j = b_idx // 2
                b_k = b_idx % 2
                global_b_idx = b_j * p + (2 * t + b_k)
                V[global_b_idx, col] += coeff

            # Fill W (C indices): local (i, k) -> global row 2*t + k, col i
            # W convention: W[k*n + i, r] where k is column of C, i is row of C
            for c_idx, coeff in zip(c_idx_list, c_val_list):
                c_i = c_idx // 2
                c_k = c_idx % 2
                global_w_idx = (2 * t + c_k) * n + c_i
                W[global_w_idx, col] += coeff

    # Deadline guard
    if time.time() > deadline:
        # Should never trigger for this fixed problem
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
