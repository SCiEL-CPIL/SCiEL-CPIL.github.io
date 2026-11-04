import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 140.0

def construct():
    start = time.time()
    n, m, p = DIMENSIONS

    rank = 77

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Outer Strassen steps: each is (A_comb, B_comb, C_comb)
    # A_comb: dict from block name ('00','01','10','11') to coefficient
    # B_comb: same
    # C_comb: same
    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    def get_u_row(block, i1, j1):
        if block == '00': i, j = i1, j1
        elif block == '01': i, j = i1, j1 + 2
        elif block == '10': i, j = i1 + 2, j1
        elif block == '11': i, j = i1 + 2, j1 + 2
        else: raise ValueError
        return i * m + j
        
    def get_v_row(block, j1, k1):
        if block == '00': j, k = j1, k1
        elif block == '01': j, k = j1, k1 + 3
        elif block == '10': j, k = j1 + 2, k1
        elif block == '11': j, k = j1 + 2, k1 + 3
        else: raise ValueError
        return j * p + k
        
    def get_w_row(block, i1, k1):
        if block == '00': i, k = i1, k1
        elif block == '01': i, k = i1, k1 + 3
        elif block == '10': i, k = i1 + 2, k1
        elif block == '11': i, k = i1 + 2, k1 + 3
        else: raise ValueError
        return k * n + i

    # Inner Strassen for 2x2 * 2x2
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
        
        # Part 1: Inner Strassen for 2x2 * 2x2 (cols 0,1 of each B block)
        for inner_s in range(7):
            i_alpha, i_beta, i_gamma = inner_steps[inner_s]
            
            for ai in range(2):
                for aj in range(2):
                    inner_key = f"{ai}{aj}"
                    inner_coeff = i_alpha.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in alpha.items():
                            U[get_u_row(b, ai, aj), col] += val * inner_coeff
            
            for bj in range(2):
                for bk in range(2):
                    inner_key = f"{bj}{bk}"
                    inner_coeff = i_beta.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in beta.items():
                            V[get_v_row(b, bj, bk), col] += val * inner_coeff
            
            for ci in range(2):
                for ck in range(2):
                    inner_key = f"{ci}{ck}"
                    inner_coeff = i_gamma.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in gamma.items():
                            W[get_w_row(b, ci, ck), col] += val * inner_coeff
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 * 2x1 (col 2 of each B block)
        for ai in range(2):
            for bj in range(2):
                for b, val in alpha.items():
                    U[get_u_row(b, ai, bj), col] += val
                for b, val in beta.items():
                    V[get_v_row(b, bj, 2), col] += val
                for b, val in gamma.items():
                    W[get_w_row(b, ai, 2), col] += val
                col += 1

    # Verify
    rng = np.random.default_rng(12345)
    verified = True
    for trial in range(5):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        A_flat = A.reshape(n * m)
        B_flat = B.reshape(m * p)
        C_true = A @ B
        C_true_flat = C_true.T.reshape(p * n)
        a = U.T @ A_flat
        b = V.T @ B_flat
        C_scheme_flat = W.T @ (a * b)
        if not np.allclose(C_scheme_flat, C_true_flat, atol=1e-8):
            verified = False
            break

    if not verified:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_sb[i * m + j, idx] = 1
                    V_sb[j * p + k, idx] = 1
                    W_sb[k * n + i, idx] = 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
