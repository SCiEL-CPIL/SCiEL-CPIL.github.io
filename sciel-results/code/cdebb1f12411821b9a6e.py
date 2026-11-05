"""Matrix multiplication 6x6x6: 3x3 blocks of 2x2 with Strassen inner (rank 189)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 140

    n, m, p = 6, 6, 6
    rank = 189
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)

    # Strassen 2x2: 7 terms
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b10-b11)
    # P4 = a11*(b01-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # For each Strassen term, define:
    # - U coefficients: which A entries (local 2x2) contribute and with what sign
    # - V coefficients: which B entries (local 2x2) contribute and with what sign
    # - W coefficients: which C entries (local 2x2) receive contribution and with what sign

    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # P1 = (a00+a11)*(b00+b11) -> C00 += 1, C11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2 = (a10+a11)*b00 -> C10 += 1, C11 += -1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # P3 = a00*(b10-b11) -> C00 += 1, C01 += 1
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}),
        # P4 = a11*(b01-b00) -> C00 += 1, C10 += 1
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # P5 = (a00+a01)*b11 -> C01 += 1, C00 += -1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # P6 = (a10-a00)*(b00+b01) -> C11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7 = (a01-a11)*(b10+b11) -> C00 += 1, C01 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}),
    ]

    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for uc, vc, wc_ in strassen_terms:
                    for (r, c), coeff in uc.items():
                        U[(2*I+r)*6 + (2*J+c), col] += coeff
                    for (r, c), coeff in vc.items():
                        V[(2*J+r)*6 + (2*K+c), col] += coeff
                    for (r, c), coeff in wc_.items():
                        W[(2*K+c)*6 + (2*I+r), col] += coeff
                    col += 1

    # Verify the tensor decomposition
    # T[i,j,k] = sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r] should equal 1 for all i,j,k
    valid = True
    if time.time() < DEADLINE - 10:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

    if not valid:
        # Fallback to schoolbook 216
        rank = 216
        U = np.zeros((36, rank), dtype=float)
        V = np.zeros((36, rank), dtype=float)
        W = np.zeros((36, rank), dtype=float)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    col = i * 36 + j * 6 + k
                    U[i*6 + j, col] = 1
                    V[j*6 + k, col] = 1
                    W[k*6 + i, col] = 1

    U_list = [[U[r, c] for c in range(rank)] for r in range(36)]
    V_list = [[V[r, c] for c in range(rank)] for r in range(36)]
    W_list = [[W[r, c] for c in range(rank)] for r in range(36)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
