import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact rank-49 tensor decomposition for 4x4 matrix multiplication
    using the Strassen-based construction (2x2x2 block recursion with rank-7
    inner blocks), with the W factor following the convention W[k*n+i, r].
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # 2x2 Strassen rank-7 decomposition
    # U2[a_idx, t]: coefficient of A sub-block in input linear form
    # V2[b_idx, t]: coefficient of B sub-block in input linear form
    # W2[c_idx, t]: coefficient of C sub-block in output
    # Sub-block ordering: 00, 01, 10, 11

    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # t=0: (A00+A11)(B00+B11) -> C00 += , C11 +=
    U2[[0, 3], 0] = 1
    V2[[0, 3], 0] = 1
    W2[0, 0] = 1
    W2[3, 0] = 1

    # t=1: (A10+A11)(B00) -> C10 += , C11 -=
    U2[[2, 3], 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1
    W2[3, 1] = -1

    # t=2: A00 * (B01-B11) -> C01 += , C11 +=
    U2[0, 2] = 1
    V2[1, 2] = 1
    V2[3, 2] = -1
    W2[2, 2] = 1
    W2[3, 2] = 1

    # t=3: A11 * (B10-B00) -> C00 += , C10 +=
    U2[3, 3] = 1
    V2[2, 3] = 1
    V2[0, 3] = -1
    W2[0, 3] = 1
    W2[1, 3] = 1

    # t=4: (A00+A01)(B11) -> C00 -= , C01 +=
    U2[[0, 1], 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1
    W2[2, 4] = 1

    # t=5: (A10-A00)(B00+B01) -> C11 +=
    U2[2, 5] = 1
    U2[0, 5] = -1
    V2[[0, 1], 5] = 1
    W2[3, 5] = 1

    # t=6: (A01-A11)(B10+B11) -> C00 +=
    U2[1, 6] = 1
    U2[3, 6] = -1
    V2[[2, 3], 6] = 1
    W2[0, 6] = 1

    # Strassen coefficients for the 2x2 outer structure
    # beta: A block row I * A block col J
    beta = np.zeros((7, 2, 2), dtype=int)
    # gamma: B block row J * B block col K
    gamma = np.zeros((7, 2, 2), dtype=int)
    # delta: C block row I * C block col K
    delta = np.zeros((7, 2, 2), dtype=int)

    # s=0
    beta[0, 0, 0] = 1
    beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1
    gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1
    delta[0, 1, 1] = 1

    # s=1
    beta[1, 1, 0] = 1
    beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1
    delta[1, 1, 1] = -1

    # s=2
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1
    gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1
    delta[2, 1, 1] = 1

    # s=3
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1
    gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1
    delta[3, 1, 0] = 1

    # s=4
    beta[4, 0, 0] = 1
    beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1
    delta[4, 0, 1] = 1

    # s=5
    beta[5, 1, 0] = 1
    beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1
    gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1

    # s=6
    beta[6, 0, 1] = 1
    beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1
    gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1

    # Build the full 4x4 decomposition via Kronecker structure
    for s in range(7):
        for t in range(7):
            c = s * 7 + t

            # U factor: A[i,j] where i,j in 0..3
            # I = i//2, i_prime = i%2, J = j//2, j_prime = j%2
            for I in range(2):
                for J in range(2):
                    b = beta[s, I, J]
                    if b != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_val = U2[i_prime * 2 + j_prime, t]
                                if u_val != 0:
                                    i = I * 2 + i_prime
                                    j = J * 2 + j_prime
                                    U[i * 4 + j, c] = b * u_val

            # V factor: B[j,k]
            for J in range(2):
                for K in range(2):
                    g = gamma[s, J, K]
                    if g != 0:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v_val = V2[j_prime * 2 + k_prime, t]
                                if v_val != 0:
                                    j = J * 2 + j_prime
                                    k = K * 2 + k_prime
                                    V[j * 4 + k, c] = g * v_val

            # W factor: C[i,k] stored as W[k*n + i] = W[k*4 + i]
            for I in range(2):
                for K in range(2):
                    d = delta[s, I, K]
                    if d != 0:
                        for i_prime in range(2):
                            for k_prime in range(2):
                                w_val = W2[k_prime * 2 + i_prime, t]
                                if w_val != 0:
                                    i = I * 2 + i_prime
                                    k = K * 2 + k_prime
                                    W[k * 4 + i, c] = d * w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
