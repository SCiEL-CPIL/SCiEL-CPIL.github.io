"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication
    using Strassen's algorithm on 7 blocks of 2x2.
    
    The multiplication is C = A * B where A is 2x2 and B is 2x14.
    We split B into 7 blocks of 2x2: B_0, ..., B_6.
    Then C = [A*B_0, ..., A*B_6].
    Each A*B_t is computed using Strassen's algorithm (rank 7).
    Total rank = 7 * 7 = 49.
    
    Tensor convention:
    U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] sums to 1.
    n=2, m=2, p=14.
    U shape: (4, 49)
    V shape: (28, 49)
    W shape: (28, 49)
    """
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    num_blocks = p // 2
    rank = num_blocks * 7

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen terms for 2x2 * 2x2
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11 (local to block)
    # C indices: 0:00, 1:01, 2:10, 3:11 (local to block)
    
    # Term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: {A_idx: coeff}
    # V_coeffs: {B_local_idx: coeff}
    # W_coeffs: {C_local_idx: coeff}
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10 + A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00 + A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            # Fill U
            for a_idx, coeff in u_coeffs.items():
                i = a_idx // m
                j = a_idx % m
                U[i * m + j, col] = coeff
            
            # Fill V
            # B global index: j*p + k
            # k = 2*t + b_col
            for b_idx, coeff in v_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff
            
            # Fill W
            # C global index: k*n + i
            # k = 2*t + c_col
            for c_idx, coeff in w_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Verify
    def tensor_check(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not tensor_check(U, V, W):
        # Fallback
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Convert to lists
    def to_list(arr):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in arr.tolist()]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
