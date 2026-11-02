import numpy as np
import time

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The multiplication tensor for (2,2,14) is the direct sum of 7 copies of the
    (2,2,2) multiplication tensor. Each (2,2,2) block has rank 7 (Strassen's algorithm).
    The total rank is 7 * 7 = 49. This is provably minimal.
    
    Returns:
        dict: {'U': list, 'V': list, 'W': list}
    """
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = 2, 2, 14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49

    # Strassen's algorithm terms for 2x2 * 2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each term: (a_indices, a_vals, b_indices, b_vals, c_indices, c_vals)
    # Indices are local to the 2x2 block (0..3)
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # M1
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # M3
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # M4
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # M5
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # M6
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]

    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            a_idx_list, a_val_list, b_idx_list, b_val_list, c_idx_list, c_val_list = strassen_terms[s]

            # Map local indices to global indices
            # A: local idx -> (i, j) -> global i*m+j
            for a_idx, coeff in zip(a_idx_list, a_val_list):
                a_i = a_idx // 2
                a_j = a_idx % 2
                U[a_i * m + a_j, col] = coeff

            # B: local idx -> (j, k_local) -> global j*p + (2*t + k_local)
            for b_idx, coeff in zip(b_idx_list, b_val_list):
                b_j = b_idx // 2
                b_k_local = b_idx % 2
                b_k_global = 2 * t + b_k_local
                V[b_j * p + b_k_global, col] = coeff

            # C: local idx -> (k_local, i) -> global (2*t + k_local)*n + i
            # c_idx mapping: 0=C00(i=0,k=0), 1=C01(i=0,k=1), 2=C10(i=1,k=0), 3=C11(i=1,k=1)
            # c_idx = i * 2 + k_local
            for c_idx, coeff in zip(c_idx_list, c_val_list):
                c_i = c_idx // 2
                c_k_local = c_idx % 2
                c_k_global = 2 * t + c_k_local
                W[c_k_global * n + c_i, col] = coeff

    # Verification (optional, for safety)
    # if time.time() < deadline:
    #     pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
