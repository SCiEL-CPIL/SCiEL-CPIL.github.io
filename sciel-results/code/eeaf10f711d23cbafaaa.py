import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    This is the Kronecker product of Strassen's rank-7 algorithm for 2x2
    matrix multiplication with itself, producing the standard rank-49
    decomposition.
    """
    n, m, p = DIMENSIONS
    rank = 49

    # Strassen rank-7 decomposition for 2x2 matrix multiplication
    # U_local[t][a_idx] where a_idx = i*2+j for A[i,j]
    U_local = np.array([
        [1, 0, 0, 1],   # M1 = (A00+A11)(B00+B11)
        [0, 0, 1, 1],   # M2 = (A10+A11)B00
        [1, 0, 0, 0],   # M3 = A00(B01-B11)
        [0, 0, 0, 1],   # M4 = A11(B10-B00)
        [1, 1, 0, 0],   # M5 = (A00+A01)B11
        [-1, 0, 1, 0],  # M6 = (A10-A00)(B00+B01)
        [0, 1, 0, -1]   # M7 = (A01-A11)(B10+B11)
    ], dtype=int)

    # V_local[t][b_idx] where b_idx = k*2+l for B[k,l]
    V_local = np.array([
        [1, 0, 0, 1],   # (A00+A11)(B00+B11)
        [1, 0, 0, 0],   # (A10+A11)B00
        [0, 1, 0, -1],  # A00(B01-B11)
        [-1, 0, 1, 0],  # A11(B10-B00)
        [0, 0, 0, 1],   # (A00+A01)B11
        [1, 1, 0, 0],   # (A10-A00)(B00+B01)
        [0, 0, 1, 1]    # (A01-A11)(B10+B11)
    ], dtype=int)

    # W_local[t][c_idx] where c_idx = i*2+j for C[i,j]
    W_local = np.array([
        [1, 0, 0, 1],   # M1: C00 += M1, C11 += M1
        [0, 0, 1, -1],  # M2: C10 += M2, C11 -= M2
        [0, 1, 0, 1],   # M3: C01 += M3, C11 += M3
        [1, 0, 1, 0],   # M4: C00 += M4, C10 += M4
        [-1, 1, 0, 0],  # M5: C00 -= M5, C01 += M5
        [0, 0, 0, 1],   # M6: C11 += M6
        [1, 0, 0, 0]    # M7: C00 += M7
    ], dtype=int)

    # Build the rank-49 decomposition via Kronecker product
    # Outer Strassen for 4x4 blocks (each block is 2x2)
    O_A = U_local.copy()  # Same structure for outer level
    O_B = V_local.copy()
    O_C = W_local.copy()

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Precompute nonzero entries for efficiency
    U_nz = {}
    for t in range(7):
        for a_idx in range(4):
            val = U_local[t, a_idx]
            if val != 0:
                U_nz[(t, a_idx)] = val

    V_nz = {}
    for t in range(7):
        for b_idx in range(4):
            val = V_local[t, b_idx]
            if val != 0:
                V_nz[(t, b_idx)] = val

    W_nz = {}
    for t in range(7):
        for c_idx in range(4):
            val = W_local[t, c_idx]
            if val != 0:
                W_nz[(t, c_idx)] = val

    for s in range(7):
        for t in range(7):
            col = s * 7 + t

            # Fill U: global index (2*i1+i2)*m + (2*j1+j2)
            for i1 in range(2):
                for j1 in range(2):
                    outer_a_idx = i1 * 2 + j1
                    coeff_outer = O_A[s, outer_a_idx]
                    if coeff_outer == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            inner_a_idx = i2 * 2 + j2
                            coeff_inner = U_nz.get((t, inner_a_idx), 0)
                            if coeff_inner == 0:
                                continue
                            global_idx = (2 * i1 + i2) * m + (2 * j1 + j2)
                            U[global_idx, col] += coeff_outer * coeff_inner

            # Fill V: global index (2*k1+k2)*p + (2*l1+l2)
            for k1 in range(2):
                for l1 in range(2):
                    outer_b_idx = k1 * 2 + l1
                    coeff_outer = O_B[s, outer_b_idx]
                    if coeff_outer == 0:
                        continue
                    for k2 in range(2):
                        for l2 in range(2):
                            inner_b_idx = k2 * 2 + l2
                            coeff_inner = V_nz.get((t, inner_b_idx), 0)
                            if coeff_inner == 0:
                                continue
                            global_idx = (2 * k1 + k2) * p + (2 * l1 + l2)
                            V[global_idx, col] += coeff_outer * coeff_inner

            # Fill W: W[k*n+i, r] convention
            # global output index: (2*c_col1+c_col2)*n + (2*c_row1+c_row2)
            for c_row1 in range(2):
                for c_col1 in range(2):
                    outer_c_idx = c_row1 * 2 + c_col1
                    coeff_outer = O_C[outer_c_idx, s]
                    if coeff_outer == 0:
                        continue
                    for c_row2 in range(2):
                        for c_col2 in range(2):
                            inner_c_idx = c_row2 * 2 + c_col2
                            coeff_inner = W_nz.get((t, inner_c_idx), 0)
                            if coeff_inner == 0:
                                continue
                            global_idx = (2 * c_col1 + c_col2) * n + (2 * c_row1 + c_row2)
                            W[global_idx, col] += coeff_outer * coeff_inner

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
