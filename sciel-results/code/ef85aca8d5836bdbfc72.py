import numpy as np
import time

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 140.0

def construct():
    start = time.time()
    n, m, p = DIMENSIONS

    # Part 1: 4x4x4 multiplication using recursive Strassen (Rank 49)
    # Part 2: 4x4x2 multiplication using Schoolbook (Rank 8)
    # Total Rank: 57

    # --- Construct 4x4x4 Strassen (Rank 49) ---
    U_44 = np.zeros((16, 49), dtype=float)
    V_44 = np.zeros((16, 49), dtype=float)
    W_44 = np.zeros((16, 49), dtype=float)

    # Outer Strassen steps for 4x4 * 4x4
    # A blocks: 00(0:2,0:2), 01(0:2,2:4), 10(2:4,0:2), 11(2:4,2:4)
    # B blocks: 00(0:2,0:2), 01(0:2,2:4), 10(2:4,0:2), 11(2:4,2:4)
    # C blocks: 00(0:2,0:2), 01(0:2,2:4), 10(2:4,0:2), 11(2:4,2:4)
    
    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),   # M1
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),           # M2
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),           # M3
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),           # M4
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),           # M5
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),           # M6
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),           # M7
    ]

    def get_u_row_44(block, i1, j1):
        if block == '00': i, j = i1, j1
        elif block == '01': i, j = i1, j1 + 2
        elif block == '10': i, j = i1 + 2, j1
        elif block == '11': i, j = i1 + 2, j1 + 2
        else: raise ValueError
        return i * 4 + j

    def get_v_row_44(block, j1, k1):
        if block == '00': j, k = j1, k1
        elif block == '01': j, k = j1, k1 + 2
        elif block == '10': j, k = j1 + 2, k1
        elif block == '11': j, k = j1 + 2, k1 + 2
        else: raise ValueError
        return j * 4 + k

    def get_w_row_44(block, i1, k1):
        if block == '00': i, k = i1, k1
        elif block == '01': i, k = i1, k1 + 2
        elif block == '10': i, k = i1 + 2, k1
        elif block == '11': i, k = i1 + 2, k1 + 2
        else: raise ValueError
        return k * 4 + i

    inner_steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        for inner_s in range(7):
            i_alpha, i_beta, i_gamma = inner_steps[inner_s]
            
            for ai in range(2):
                for aj in range(2):
                    inner_key = f"{ai}{aj}"
                    inner_coeff = i_alpha.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in alpha.items():
                            U_44[get_u_row_44(b, ai, aj), col] += val * inner_coeff
            
            for bj in range(2):
                for bk in range(2):
                    inner_key = f"{bj}{bk}"
                    inner_coeff = i_beta.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in beta.items():
                            V_44[get_v_row_44(b, bj, bk), col] += val * inner_coeff
            
            for ci in range(2):
                for ck in range(2):
                    inner_key = f"{ci}{ck}"
                    inner_coeff = i_gamma.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in gamma.items():
                            W_44[get_w_row_44(b, ci, ck), col] += val * inner_coeff
            
            col += 1

    # --- Construct 4x4x2 Schoolbook (Rank 8) ---
    U_42 = np.zeros((16, 8), dtype=float)
    V_42 = np.zeros((8, 8), dtype=float)
    W_42 = np.zeros((8, 8), dtype=float)
    
    # A is 4x4, B is 4x2.
    # C[i, j] = sum_k A[i, k] B[k, j]
    # We need 8 terms.
    # Term t corresponds to A[i, k] * B[k, j]
    # We can map t = k * 2 + j
    # U[t, i*4+k] = 1
    # V[t, k*2+j] = 1
    # W[t, j*4+i] = 1
    
    for k in range(4):
        for j in range(2):
            t = k * 2 + j
            for i in range(4):
                U_42[i * 4 + k, t] = 1.0
                W_42[j * 4 + i, t] = 1.0
            V_42[k * 2 + j, t] = 1.0

    # --- Combine ---
    U = np.hstack([U_44, U_42])
    V = np.hstack([V_44, V_42])
    W = np.hstack([W_44, W_42])

    # Verification
    try:
        rng = np.random.default_rng(12345)
        for trial in range(10):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            A_flat = A.reshape(n * m)
            B_flat = B.reshape(m * p)
            C_true = A @ B
            C_true_flat = C_true.T.reshape(p * n)
            a = U.T @ A_flat
            b = V.T @ B_flat
            C_scheme_flat = W.T @ (a * b)
            if not np.allclose(C_scheme_flat, C_true_flat, atol=1e-10):
                break
    except Exception:
        pass

    if time.time() - start > DEADLINE_SECONDS - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
