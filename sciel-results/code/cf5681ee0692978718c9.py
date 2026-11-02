"""Matrix multiplication 2,2,14 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The multiplication tensor decomposes as a direct sum of 7 independent 2x2 * 2x2 tensors.
    Using Strassen's rank-7 decomposition for each block gives total rank 49.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7 blocks of 2x2

    # Strassen's rank-7 decomposition for 2x2 * 2x2
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),         # M7
    ]

    rank = num_blocks * 7  # 49
    U = np.zeros((nm, rank), dtype=np.int32)
    V = np.zeros((mp, rank), dtype=np.int32)
    W = np.zeros((pn, rank), dtype=np.int32)

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]

            # Fill U: index (i, j) -> i*m + j
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff

            # Fill V: index (j, k) -> j*p + k
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff

            # Fill W: index (k, i) -> k*n + i  [trace convention]
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Convert to plain lists with integer entries
    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
