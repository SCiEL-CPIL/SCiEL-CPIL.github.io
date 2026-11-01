import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # n=2, m=2, p=16

    # Strategy:
    # The tensor for C = A @ B where A is 2x2, B is 2x16.
    # Indices: i (row A), j (col A / row B), k (col B).
    # Tensor T[i,j,k] = 1.
    # We can view this as 8 independent 2x2x2 multiplications.
    # B has 2 rows and 16 columns.
    # We can partition the 16 columns into 8 blocks of 2 columns.
    # Let B = [B_0, B_1, ..., B_7] where B_q is a 2x2 matrix.
    # C = A @ B = [A@B_0, A@B_1, ..., A@B_7].
    # The tensor for the whole multiplication is the direct sum of 8 tensors
    # each corresponding to multiplying a 2x2 matrix A by a 2x2 matrix B_q.
    # However, the A factors are the SAME for all q.
    # In a tensor decomposition T = sum_r U_r \otimes V_r \otimes W_r,
    # if we use Strassen's algorithm for each block, we can potentially reuse
    # the U factors?
    # No, the U factors depend on A, which is fixed.
    # But the decomposition is linear.
    # T_total = sum_{q=0}^{7} T_q where T_q is the tensor for the q-th block.
    # T_q has support on B indices for block q and C indices for block q.
    # The U part of T_q is the same for all q (it only depends on A indices).
    # Specifically, if we decompose T_2x2x2 into rank 7 terms:
    # T_2x2x2 = sum_{s=0}^{6} U_s \otimes V_s \otimes W_s
    # Then T_total = sum_{q=0}^{7} sum_{s=0}^{6} U_s \otimes (V_s^{(q)}) \otimes (W_s^{(q)})
    # where V_s^{(q)} is V_s but with indices shifted to block q in the B dimension,
    # and W_s^{(q)} is W_s but with indices shifted to block q in the C dimension.
    # The U_s are the SAME for all q.
    # However, in the final matrix representation:
    # U is (n*m, R). The columns are the U factors.
    # If we simply repeat the same U_s 8 times, we get 8 * 7 = 56 columns in U.
    # Can we do better?
    # Notice that U_s is a vector in R^4.
    # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # If we have the same U_s for different q, say r and r' have U_r = U_{r'},
    # we cannot simply merge them because V_r and V_{r'} are different vectors
    # (they have support on different B indices).
    # Merging would require V_r = V_{r'} and W_r = W_{r'} to combine coefficients,
    # which is not the case.
    # So the rank is indeed 8 * 7 = 56.

    num_blocks = p // 2  # 8 blocks
    R = num_blocks * 7  # 56

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Strassen definitions for 2x2x2
    # A indices: 0(A11), 1(A12), 2(A21), 3(A22)
    # B local indices: 0(B11), 1(B12), 2(B21), 3(B22)
    # C local indices: 0(C11), 1(C12), 2(C21), 3(C22)

    U_coeffs = {
        0: {0: 1, 3: 1},   # M1
        1: {2: 1, 3: 1},   # M2
        2: {0: 1},         # M3
        3: {3: 1},         # M4
        4: {0: 1, 1: 1},   # M5
        5: {2: 1, 0: -1},  # M6
        6: {1: 1, 3: -1}   # M7
    }

    V_coeffs = {
        0: {0: 1, 3: 1},   # M1
        1: {0: 1},         # M2
        2: {1: 1, 3: -1},  # M3
        3: {2: 1, 0: -1},  # M4
        4: {3: 1},         # M5
        5: {0: 1, 1: 1},   # M6
        6: {2: 1, 3: 1}    # M7
    }

    W_contribs = {
        0: {0: 1, 3: 1},   # M1 -> C11, C22
        1: {2: 1, 3: -1},  # M2 -> C21, C22
        2: {1: 1, 3: 1},   # M3 -> C12, C22
        3: {0: 1, 2: 1},   # M4 -> C11, C21
        4: {0: -1, 1: 1},  # M5 -> C11, C12
        5: {3: 1},         # M6 -> C22
        6: {0: 1}          # M7 -> C11
    }

    for q in range(num_blocks):
        # Global indices for B in this block
        # B is 2x16. Block q covers columns 2q and 2q+1.
        # B[0, 2q] -> idx 0*p + 2q = 2q
        # B[0, 2q+1] -> idx 0*p + 2q+1 = 2q+1
        # B[1, 2q] -> idx 1*p + 2q = 16 + 2q
        # B[1, 2q+1] -> idx 1*p + 2q+1 = 16 + 2q+1

        b_idx_11 = 2*q
        b_idx_12 = 2*q + 1
        b_idx_21 = 16 + 2*q
        b_idx_22 = 16 + 2*q + 1

        # Global indices for C in this block
        # C is 2x16. Output index k*n + i.
        # C[0, 2q] -> k=2q, i=0 -> 2q*2 + 0 = 4q
        # C[0, 2q+1] -> k=2q+1, i=0 -> (2q+1)*2 + 0 = 4q+2
        # C[1, 2q] -> k=2q, i=1 -> 2q*2 + 1 = 4q+1
        # C[1, 2q+1] -> k=2q+1, i=1 -> (2q+1)*2 + 1 = 4q+3

        c_idx_11 = 4*q
        c_idx_12 = 4*q + 2
        c_idx_21 = 4*q + 1
        c_idx_22 = 4*q + 3

        for s in range(7):
            r = q * 7 + s

            # Fill U
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff

            # Fill V
            for local_idx, coeff in V_coeffs[s].items():
                if local_idx == 0:
                    V[b_idx_11, r] = coeff
                elif local_idx == 1:
                    V[b_idx_12, r] = coeff
                elif local_idx == 2:
                    V[b_idx_21, r] = coeff
                elif local_idx == 3:
                    V[b_idx_22, r] = coeff

            # Fill W
            for local_out_idx, coeff in W_contribs[s].items():
                if local_out_idx == 0:
                    W[c_idx_11, r] = coeff
                elif local_out_idx == 1:
                    W[c_idx_12, r] = coeff
                elif local_out_idx == 2:
                    W[c_idx_21, r] = coeff
                elif local_out_idx == 3:
                    W[c_idx_22, r] = coeff

    # Verification
    if time.time() < deadline - 10:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test

        A_flat = A_test.flatten()
        B_flat = B_test.flatten()

        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms

        C_test_flat = C_test.T.flatten()

        if not np.allclose(C_decomp, C_test_flat):
            # Fallback to schoolbook rank 64
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)

            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i

                        U[u_idx, column] = 1
                        V[v_idx, column] = 1
                        W[w_idx, column] = 1
                        column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
