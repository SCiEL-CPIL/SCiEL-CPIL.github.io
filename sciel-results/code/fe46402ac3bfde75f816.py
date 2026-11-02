import numpy as np
import time

def construct():
    """
    Finds a low-rank exact tensor decomposition for 3x3x6 matrix multiplication.
    Strategy:
    1. Use block decomposition: 3x3x6 = two 3x3x3 blocks.
       If we find a rank-R decomposition for 3x3x3, we get rank 2R for 3x3x6.
    2. Search for 3x3x3 cores of rank 23-27 using random U,V with {-1,0,1} entries.
       For each (i,k) output pair, solve a 3xR linear system for W[i,k,:].
    3. Extend the core to 3x3x6 by duplicating U and splitting V,W into two halves.
    4. Fallback to schoolbook rank-54 if no better found within time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    n, m, p = 3, 3, 6
    dim_A = n * m  # 9
    dim_B = m * p  # 18
    dim_C = p * n  # 18

    # Build the target tensor T as a dictionary of non-zero entries
    T_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_nonzero[(a, b, c)] = 1.0

    # Schoolbook fallback
    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = generate_schoolbook()
    best_rank = n * m * p

    # Precompute the 3x3x3 core structure
    # For 3x3x3: dim_A_core=9, dim_B_core=9, dim_C_core=9
    # T_core[(i,j),(j,k),(i,k)] = 1
    # For fixed (i,k), we need: sum_r U[i*3+j, r] * V[j*3+k, r] * W[i*3+k, r] = 1 for all j
    # This is: M_{(i,k)} @ W[i*3+k, :] = ones(3)
    # where M_{(i,k)}[j, r] = U[i*3+j, r] * V[j*3+k, r]

    # Search for 3x3x3 cores
    found_core = False
    core_U = core_V = core_W = None
    best_core_rank = 28  # Will use 2*28=56 if no better found, but we start from 23

    for R in range(23, 28):
        if time.time() > deadline:
            break

        # Number of attempts per rank
        attempts = 3000

        for attempt in range(attempts):
            if time.time() > deadline:
                break

            # Generate random U, V with entries in {-1, 0, 1}
            U_core = np.random.randint(-1, 2, size=(9, R))
            V_core = np.random.randint(-1, 2, size=(9, R))

            W_core = np.zeros((9, R))
            valid = True

            # Solve for each output (i, k)
            for i in range(3):
                for k in range(3):
                    # Build 3 x R matrix M where M[j, r] = U[i*3+j, r] * V[j*3+k, r]
                    M = np.zeros((3, R))
                    for j in range(3):
                        M[j, :] = U_core[i * 3 + j, :] * V_core[j * 3 + k, :]

                    # Solve M @ w_vec = ones(3)
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(3), rcond=None)
                    except:
                        valid = False
                        break

                    # Check if solution is close to integers or half-integers
                    w_doubled = w_vec * 2
                    closest_ints = np.round(w_doubled)
                    error = np.max(np.abs(w_doubled - closest_ints))

                    if error < 1e-9:
                        W_core[i * 3 + k, :] = w_vec
                    else:
                        valid = False
                        break

            if valid:
                # Verify the core solution
                is_valid = True
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = np.sum(U_core[i * 3 + j, :] * V_core[j * 3 + k, :] * W_core[i * 3 + k, :])
                            if abs(s - 1.0) > 1e-9:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    if not is_valid:
                        break

                if is_valid:
                    # Check that non-matching terms are zero
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                for i2 in range(3):
                                    for k2 in range(3):
                                        if (i2, k2) != (i, k):
                                            s = np.sum(U_core[i * 3 + j, :] * V_core[j * 3 + k, :] * W_core[i2 * 3 + k2, :])
                                            if abs(s) > 1e-9:
                                                is_valid = False
                                                break
                                        if not is_valid:
                                            break
                                    if not is_valid:
                                        break
                                if not is_valid:
                                    break
                            if not is_valid:
                                break
                        if not is_valid:
                            break
                    if not is_valid:
                        continue

                found_core = True
                core_U = U_core.tolist()
                core_V = V_core.tolist()
                core_W = W_core.tolist()
                best_core_rank = R
                break

        if found_core:
            break

    if found_core:
        # Construct 3x3x6 solution from core
        # 3x3x6 is two 3x3x3 blocks (k=0,1,2 and k=3,4,5)
        # U is shared: U_total[i*3+j, r] = core_U[i*3+j, r] for first half
        #                            U_total[i*3+j, R+r] = core_U[i*3+j, r] for second half
        # V_total[j*6+k, r] = core_V[j*3+k, r] for k=0,1,2 (first half)
        # V_total[j*6+k, R+r] = core_V[j*3+(k-3), r] for k=3,4,5 (second half)
        # W_total[k*3+i, r] = core_W[k*3+i, r] for k=0,1,2 (first half)
        # W_total[k*3+i, R+r] = core_W[(k-3)*3+i, r] for k=3,4,5 (second half)

        R_total = 2 * best_core_rank
        U_total = [[0] * R_total for _ in range(dim_A)]
        V_total = [[0] * R_total for _ in range(dim_B)]
        W_total = [[0] * R_total for _ in range(dim_C)]

        for i in range(3):
            for j in range(3):
                for r in range(best_core_rank):
                    # U is duplicated for both halves
                    U_total[i * 3 + j][r] = core_U[i * 3 + j][r]
                    U_total[i * 3 + j][best_core_rank + r] = core_U[i * 3 + j][r]

                    # V for first half (k=0,1,2)
                    for k in range(3):
                        V_total[j * 6 + k][r] = core_V[j * 3 + k][r]

                    # V for second half (k=3,4,5)
                    for k in range(3):
                        V_total[j * 6 + (k + 3)][best_core_rank + r] = core_V[j * 3 + k][r]

        for k in range(6):
            for i in range(3):
                for r in range(best_core_rank):
                    if k < 3:
                        # First half
                        W_total[k * 3 + i][r] = core_W[k * 3 + i][r]
                    else:
                        # Second half
                        W_total[k * 3 + i][best_core_rank + r] = core_W[(k - 3) * 3 + i][r]

        # Verify the full 3x3x6 solution
        is_valid = True
        for (a, b, c), val in T_nonzero.items():
            s = sum(U_total[a][r] * V_total[b][r] * W_total[c][r] for r in range(R_total))
            if abs(s - val) > 1e-9:
                is_valid = False
                break

        if is_valid:
            # Check a few zero entries
            for _ in range(200):
                a = np.random.randint(0, dim_A)
                b = np.random.randint(0, dim_B)
                c = np.random.randint(0, dim_C)
                if (a, b, c) not in T_nonzero:
                    s = sum(U_total[a][r] * V_total[b][r] * W_total[c][r] for r in range(R_total))
                    if abs(s) > 1e-9:
                        is_valid = False
                        break

        if is_valid:
            best_result = {"U": U_total, "V": V_total, "W": W_total}
            best_rank = R_total
            return best_result

    return best_result
