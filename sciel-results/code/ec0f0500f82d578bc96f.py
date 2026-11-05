import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0  # safety margin for 120s limit

    num_u = n * m  # 8
    num_v = m * p  # 20
    num_w = p * n  # 10

    # ---------- Build the target tensor T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,n+i) ----------
    def build_target():
        T = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j][j * p + k][k * n + i] = Fraction(1)
        return T

    # ---------- Strassen block solution (rank 36) ----------
    def strassen_block_solution():
        rank = 36
        U = [[0] * rank for _ in range(num_u)]
        V = [[0] * rank for _ in range(num_v)]
        W = [[0] * rank for _ in range(num_w)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        standard_terms_2x2x1 = [
            ({0: 1}, {0: 1}, {0: 1}),
            ({1: 1}, {1: 1}, {0: 1}),
            ({2: 1}, {0: 1}, {1: 1}),
            ({3: 1}, {1: 1}, {1: 1}),
        ]

        for q in range(2):
            for s in range(3):
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj, lk = idx // 2, idx % 2
                            v_g[(2 * q + lj) * p + (2 * s + lk)] = val
                        for idx, val in w_t.items():
                            li, lk = idx // 2, idx % 2
                            w_g[(2 * s + lk) * n + li] = val
                        add_term(u_g, v_g, w_g)
                else:
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj = idx
                            v_g[(2 * q + lj) * p + 4] = val
                        for idx, val in w_t.items():
                            li = idx
                            w_g[4 * n + li] = val
                        add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    # ---------- Exact verification ----------
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = Fraction(0)
                    for r in range(rank):
                        ua = Fraction(U[a_idx][r])
                        vb = Fraction(V[b_idx][r])
                        wc = Fraction(W[c_idx][r])
                        if ua and vb and wc:
                            s += ua * vb * wc
                    if s != 1:
                        return False
        return True

    # ---------- Greedy rank-1 extraction with better heuristics ----------
    def greedy_extract(target_rank, time_budget, T):
        num_u, num_v, num_w = 8, 20, 10

        # Start from Strassen solution
        base = strassen_block_solution()
        U0 = [[Fraction(x) for x in row] for row in base["U"]]
        V0 = [[Fraction(x) for x in row] for row in base["V"]]
        W0 = [[Fraction(x) for x in row] for row in base["W"]]

        # Compute residual R = T - sum of Strassen terms
        R = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for r in range(36):
            for a_idx in range(num_u):
                ua = U0[a_idx][r]
                if ua:
                    for b_idx in range(num_v):
                        vb = V0[b_idx][r]
                        if vb:
                            uv = ua * vb
                            if uv:
                                for c_idx in range(num_w):
                                    wc = W0[c_idx][r]
                                    if wc:
                                        R[a_idx][b_idx][c_idx] -= uv * wc

        for a_idx in range(num_u):
            for b_idx in range(num_v):
                for c_idx in range(num_w):
                    R[a_idx][b_idx][c_idx] += T[a_idx][b_idx][c_idx]

        U_list, V_list, W_list = [], [], []

        def try_extract(a_idx, b_idx, c_idx):
            """Try to extract a rank-1 term centered at (a_idx, b_idx, c_idx)."""
            val = R[a_idx][b_idx][c_idx]
            if val == 0:
                return False
            inv_val = Fraction(1) / val

            # Build u vector: u[i] = R[i][b_idx][c_idx] / val
            u = [Fraction(0)] * num_u
            u[a_idx] = Fraction(1)
            for u_i in range(num_u):
                if u_i == a_idx:
                    continue
                f = R[u_i][b_idx][c_idx]
                if f:
                    u[u_i] = f * inv_val

            # Build v vector: v[j] = R[a_idx][j][c_idx]
            v = [Fraction(0)] * num_v
            v[b_idx] = val
            for b_i in range(num_v):
                if b_i == b_idx:
                    continue
                f = R[a_idx][b_i][c_idx]
                if f:
                    v[b_i] = f

            # Build w vector: w[k] = R[a_idx][b_idx][k] / val
            w = [Fraction(0)] * num_w
            w[c_idx] = Fraction(1)
            for w_i in range(num_w):
                if w_i == c_idx:
                    continue
                f = R[a_idx][b_idx][w_i]
                if f:
                    w[w_i] = f * inv_val

            # Verify that the rank-1 tensor u⊗v⊗w is contained in R
            for u_i in range(num_u):
                ui = u[u_i]
                if not ui:
                    continue
                for b_i in range(num_v):
                    bi = v[b_i]
                    if not bi:
                        continue
                    expected_coeff = ui * bi
                    if expected_coeff == 0:
                        continue
                    row_r = R[u_i][b_i]
                    for w_i in range(num_w):
                        wi = w[w_i]
                        if wi:
                            if row_r[w_i] != expected_coeff * wi:
                                return False
                        else:
                            if row_r[w_i] != 0:
                                return False

            # Subtract from R
            for u_i in range(num_u):
                ui = u[u_i]
                if not ui:
                    continue
                for b_i in range(num_v):
                    bi = v[b_i]
                    if not bi:
                        continue
                    base_val = ui * bi
                    if base_val == 0:
                        continue
                    row_r = R[u_i][b_i]
                    for w_i in range(num_w):
                        wi = w[w_i]
                        if wi:
                            row_r[w_i] -= base_val * wi

            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            return True

        def is_residual_zero():
            for a_idx in range(num_u):
                for b_idx in range(num_v):
                    row = R[a_idx][b_idx]
                    for c_idx in range(num_w):
                        if row[c_idx] != 0:
                            return False
            return True

        def score_candidate(a_idx, b_idx, c_idx):
            """Score a candidate: lower is better. Prefers sparse fibers."""
            val = R[a_idx][b_idx][c_idx]
            if val == 0:
                return float('inf')

            # Count non-zeros in the three fibers
            nz_u = 0
            for u_i in range(num_u):
                if R[u_i][b_idx][c_idx] != 0:
                    nz_u += 1

            nz_v = 0
            for b_i in range(num_v):
                if R[a_idx][b_i][c_idx] != 0:
                    nz_v += 1

            nz_w = 0
            for w_i in range(num_w):
                if R[a_idx][b_idx][w_i] != 0:
                    nz_w += 1

            # Prefer small total support
            score = nz_u * nz_v + nz_v * nz_w + nz_u * nz_w
            # Bonus for being at a "corner" (fewer non-zeros)
            score += (nz_u - 1) * 10 + (nz_v - 1) * 5 + (nz_w - 1) * 20
            return score

        def find_best_candidate():
            best_score = float('inf')
            best_cand = None
            for a_idx in range(num_u):
                for b_idx in range(num_v):
                    for c_idx in range(num_w):
                        if R[a_idx][b_idx][c_idx] == 0:
                            continue
                        s = score_candidate(a_idx, b_idx, c_idx)
                        if s < best_score:
                            best_score = s
                            best_cand = (a_idx, b_idx, c_idx)
            return best_cand

        def find_any_valid():
            """Search all non-zero positions for a valid extraction."""
            for a_idx in range(num_u):
                for b_idx in range(num_v):
                    for c_idx in range(num_w):
                        if R[a_idx][b_idx][c_idx] != 0:
                            if try_extract(a_idx, b_idx, c_idx):
                                return True
            return False

        # Main extraction loop
        for step in range(target_rank):
            if time.time() > time_budget:
                return None

            if is_residual_zero():
                break

            cand = find_best_candidate()
            if cand is None:
                break

            a_idx, b_idx, c_idx = cand
            if not try_extract(a_idx, b_idx, c_idx):
                # Try all positions
                if not find_any_valid():
                    return None

        if not is_residual_zero():
            return None

        rank = len(U_list)
        Uf = [[0] * rank for _ in range(num_u)]
        Vf = [[0] * rank for _ in range(num_v)]
        Wf = [[0] * rank for _ in range(num_w)]
        for r in range(rank):
            for u_i in range(num_u):
                if U_list[r][u_i]:
                    Uf[u_i][r] = float(U_list[r][u_i])
            for b_i in range(num_v):
                if V_list[r][b_i]:
                    Vf[b_i][r] = float(V_list[r][b_i])
            for w_i in range(num_w):
                if W_list[r][w_i]:
                    Wf[w_i][r] = float(W_list[r][w_i])
        return {"U": Uf, "V": Vf, "W": Wf}

    # ---------- Main: try to find rank < 36 ----------
    best = strassen_block_solution()
    best_rank = 36

    T = build_target()

    # Try ranks from 35 down to 25, with time budget
    for tr in range(best_rank - 1, max(best_rank - 20, 1), -1):
        if time.time() > deadline - 5:
            break
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
        # Give more time to higher ranks (more likely to succeed)
        time_budget = time.time() + remaining * 0.7
        res = greedy_extract(tr, time_budget, T)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                best = res
                best_rank = len(res["U"][0])
                break

    return best
# EVOLVE-BLOCK-END
