"""Matrix multiplication tensor decomposition for 3x4 by 4x5."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 138.0
    rank_schoolbook = n * m * p

    def build_schoolbook():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = build_schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    num_eqs = n * m * p

    def max_residual(U, V, W, rank):
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(W[k * n + i, :rank], U[i * m + j, :rank] * V[j * p + k, :rank])
                    res = abs(s - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res

    def to_output(U, V, W):
        U_l = [[float(x) for x in row] for row in U]
        V_l = [[float(x) for x in row] for row in V]
        W_l = [[float(x) for x in row] for row in W]
        for L in (U_l, V_l, W_l):
            for row in L:
                for idx in range(len(row)):
                    half = round(row[idx] * 2.0) * 0.5
                    if abs(row[idx] - half) < 1e-7:
                        row[idx] = half
        return {"U": U_l, "V": V_l, "W": W_l}

    def build_A_U(W, V, rank):
        A = np.zeros((num_eqs, n * m * rank))
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        A[eq, (i * m + j) * rank + r] = W[k * n + i, r] * V[j * p + k, r]
                    eq += 1
        return A

    def build_A_V(U, W, rank):
        A = np.zeros((num_eqs, m * p * rank))
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        A[eq, (j * p + k) * rank + r] = U[i * m + j, r] * W[k * n + i, r]
                    eq += 1
        return A

    def build_A_W(U, V, rank):
        A = np.zeros((num_eqs, p * n * rank))
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        A[eq, (k * n + i) * rank + r] = U[i * m + j, r] * V[j * p + k, r]
                    eq += 1
        return A

    b_vec = np.ones(num_eqs)

    def als_solve(U, V, W, rank, num_rounds=150):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        for rnd in range(num_rounds):
            if time.time() > deadline - 2:
                break
            A_U = build_A_U(W, V, rank)
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b_vec, rcond=None)
            U = U_flat.reshape(n * m, rank)
            A_V = build_A_V(U, W, rank)
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b_vec, rcond=None)
            V = V_flat.reshape(m * p, rank)
            A_W = build_A_W(U, V, rank)
            W_flat, _, _, _ = np.linalg.lstsq(A_W, b_vec, rcond=None)
            W = W_flat.reshape(p * n, rank)
        return U, V, W

    np.random.seed(54321)

    for target_rank in range(8, 60):
        if time.time() > deadline - 5:
            break
        if target_rank >= best_rank:
            continue
        found = False
        num_trials = 25 if target_rank <= 20 else 12
        for trial in range(num_trials):
            if time.time() > deadline - 3:
                break
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            U_sol, V_sol, W_sol = als_solve(U, V, W, target_rank, num_rounds=100)
            res = max_residual(U_sol, V_sol, W_sol, target_rank)
            if res < 1e-10:
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break
        if found:
            continue

    return to_output(best_U, best_V, best_W)
# EVOLVE-BLOCK-END
