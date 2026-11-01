import numpy as np

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The multiplication tensor for (2,2,14) is the direct sum of 7 copies of the
    (2,2,2) multiplication tensor. Each (2,2,2) block has rank 7 (Strassen's algorithm).
    The total rank is 7 * 7 = 49. This is provably minimal.
    
    Returns:
        dict: {'U': list, 'V': list, 'W': list}
    """
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
            # Note: W index is k*n + i. 
            # In the tensor definition T[i,j,k], the third index is k (col of B/C).
            # The code verification uses W[k*n+i]. 
            # Let's check the local C indices in strassen_terms.
            # c_idx 0 -> C00 (i=0, k=0)
            # c_idx 1 -> C01 (i=0, k=1)
            # c_idx 2 -> C10 (i=1, k=0)
            # c_idx 3 -> C11 (i=1, k=1)
            # So c_idx = k_local * 2 + i ? No.
            # C00: i=0, k=0. 
            # C01: i=0, k=1.
            # C10: i=1, k=0.
            # C11: i=1, k=1.
            # In the terms above:
            # M1 contributes to C00 (idx 0) and C11 (idx 3).
            # So c_idx 0 corresponds to (i=0, k=0).
            # c_idx 1 corresponds to (i=0, k=1).
            # c_idx 2 corresponds to (i=1, k=0).
            # c_idx 3 corresponds to (i=1, k=1).
            # This mapping is c_idx = k_local * 2 + i ?
            # 0 = 0*2 + 0
            # 1 = 0*2 + 1 ? No, 1 is k=1, i=0. So 1*2+0=2? No.
            # Let's re-read the indices in the terms.
            # M1: c_indices [0, 3]. C00 and C11.
            # C00 is i=0, k=0.
            # C11 is i=1, k=1.
            # So c_idx 0 -> (i=0, k=0).
            # c_idx 3 -> (i=1, k=1).
            # M2: c_indices [2, 3]. C10 and C11.
            # C10 is i=1, k=0.
            # C11 is i=1, k=1.
            # So c_idx 2 -> (i=1, k=0).
            # c_idx 3 -> (i=1, k=1).
            # M3: c_indices [1, 3]. C01 and C11.
            # C01 is i=0, k=1.
            # C11 is i=1, k=1.
            # So c_idx 1 -> (i=0, k=1).
            
            # Mapping:
            # 0: i=0, k=0
            # 1: i=0, k=1
            # 2: i=1, k=0
            # 3: i=1, k=1
            # This is c_idx = i * 2 + k_local ?
            # 0 = 0*2 + 0
            # 1 = 0*2 + 1
            # 2 = 1*2 + 0
            # 3 = 1*2 + 1
            # Yes, c_idx = i * 2 + k_local.
            
            for c_idx, coeff in zip(c_idx_list, c_val_list):
                c_i = c_idx // 2
                c_k_local = c_idx % 2
                c_k_global = 2 * t + c_k_local
                W[c_k_global * n + c_i, col] = coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
