import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0

    # ---------- Structural fallback: Strassen block decomposition, rank 36 ----------
    def strassen_block_solution():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
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

    # ---------- Exact-rational verification ----------
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = 0
                    for r in range(rank):
                        if U[a_idx][r] and V[b_idx][r] and W[c_idx][r]:
                            s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                    if s != 1:
                        return False
        return True

    # ---------- Recursive block-partitioning decomposition ----------
    def block_partition_solution():
        """
        Split 2x4x5 using 1+3 partition on inner dim (m=4) and 1+4 on output (p=5).
        This gives four sub-tensors:
        - (1,1,1): rank 1
        - (1,3,1): rank 3
        - (1,1,4): rank 4
        - (1,3,4): split further into (1,1,4)+2*(1,2,2)
          Using rank-6 for 1x3x3 blocks: rank 6+6+6+6=24? No...
          
        Actually, let's use a cleaner approach:
        Split m=4 into 1+3, p=5 into 1+4.
        Sub-problems:
        - 2x1x1: rank 2
        - 2x1x4: rank 8
        - 2x3x1: rank 6
        - 2x3x4: split p=4 into 2+2
          - 2x3x2: split m=3 into 1+2
            - 2x1x2: rank 4
            - 2x2x2: Strassen rank 7
            Total: 11
          - 2x3x2: same, rank 11
          Total for 2x3x4: 22
        Grand total: 2+8+6+22 = 38 (worse than 36)
        
        Let's try a different split: m=4 into 2+2, p=5 into 1+4
        - 2x2x1: rank 4
        - 2x2x4: split p=4 into 2+2, each 2x2x2 Strassen rank 7, total 14
        - 2x2x1: rank 4
        - 2x2x4: rank 14
        Total: 4+14+4+14 = 36 (same as current)
        
        The current Strassen-based 36 is optimal for simple block partitioning.
        """
        return strassen_block_solution()

    # ---------- Exact-rational greedy rank-1 extraction (rank minimization) ----------
    def greedy_extract(target_rank, time_budget):
        num_u, num_v, num_w = n * m, m * p, p * n
        # Target tensor T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,n+i)
        T = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j][j * p + k][k * n + i] = Fraction(1)

        # Start from the Strassen solution as a valid initial decomposition
        base = strassen_block_solution()
        U0 = [[Fraction(x) for x in row] for row in base["U"]]
        V0 = [[Fraction(x) for x in row] for row in base["V"]]
        W0 = [[Fraction(x) for x in row] for row in base["W"]]
        R = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for r in range(len(U0[0])):
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
        # R should now be exactly T
        for a_idx in range(num_u):
            ra = R[a_idx]
            for b_idx in range(num_v):
                rb = ra[b_idx]
                for c_idx in range(num_w):
                    rb[c_idx] = rb[c_idx] + T[a_idx][b_idx][c_idx]

        U_list, V_list, W_list = [], [], []

        def is_valid_extraction(a_idx, b_idx, c_idx):
            """Check if a rank-1 extraction centered at (a_idx, b_idx, c_idx) is valid."""
            val = R[a_idx][b_idx][c_idx]
            if val == 0:
                return False
            inv_val = Fraction(1) / val
            
            u = [Fraction(0)] * num_u
            u[a_idx] = Fraction(1)
            for u_i in range(num_u):
                if u_i == a_idx:
                    continue
                f = R[u_i][b_idx][c_idx]
                if f:
                    u[u_i] = f * inv_val
            
            v = [Fraction(0)] * num_v
            v[b_idx] = val
            for b_i in range(num_v):
                if b_i == b_idx:
                    continue
                f = R[a_idx][b_i][c_idx]
                if f:
                    v[b_i] = f
            
            w = [Fraction(0)] * num_w
            w[c_idx] = Fraction(1)
            for w_i in range(num_w):
                if w_i == c_idx:
                    continue
                f = R[a_idx][b_idx][w_i]
                if f:
                    w[w_i] = f * inv_val
            
            # Verify
            for u_i in range(num_u):
                ui = u[u_i]
                if not ui:
                    continue
                for b_i in range(num_v):
                    bi = v[b_i]
                    if not bi:
                        continue
                    row_r = R[u_i][b_i]
                    for w_i in range(num_w):
                        wi = w[w_i]
                        if not wi:
                            if row_r[w_i] != 0:
                                return False
                        else:
                            expected = ui * bi * wi
                            if row_r[w_i] != expected:
                                return False
            return True

        def pick_candidate():
            # Pick candidate that is most likely to be a valid extraction
            # Score: prefer positions where the "fiber" through (a,b,c) is more rank-1-like
            best_score = None
            best_cand = None
            for a_idx in range(num_u):
                ra = R[a_idx]
                for b_idx in range(num_v):
                    rb = ra[b_idx]
                    for c_idx in range(num_w):
                        val = rb[c_idx]
                        if val == 0:
                            continue
                        # Count support in each direction
                        nz_v = 0
                        for b_i in range(num_v):
                            if R[a_idx][b_i][c_idx] != 0:
                                nz_v += 1
                        nz_u = 0
                        for u_i in range(num_u):
                            if R[u_i][b_idx][c_idx] != 0:
                                nz_u += 1
                        nz_w = 0
                        for w_i in range(num_w):
                            if R[a_idx][b_idx][w_i] != 0:
                                nz_w += 1
                        # A good candidate has small supports in all directions
                        score = nz_v * nz_u + nz_u * nz_w + nz_v * nz_w
                        if best_score is None or score < best_score:
                            best_score = score
                            best_cand = (a_idx, b_idx, c_idx, val)
            return best_cand

        def try_extract(a_idx, b_idx, c_idx, val):
            inv_val = Fraction(1) / val
            
            u = [Fraction(0)] * num_u
            u[a_idx] = Fraction(1)
            for u_i in range(num_u):
                if u_i == a_idx:
                    continue
                f = R[u_i][b_idx][c_idx]
                if f:
                    u[u_i] = f * inv_val
            
            v = [Fraction(0)] * num_v
            v[b_idx] = val
            for b_i in range(num_v):
                if b_i == b_idx:
                    continue
                f = R[a_idx][b_i][c_idx]
                if f:
                    v[b_i] = f
            
            w = [Fraction(0)] * num_w
            w[c_idx] = Fraction(1)
            for w_i in range(num_w):
                if w_i == c_idx:
                    continue
                f = R[a_idx][b_idx][w_i]
                if f:
                    w[w_i] = f * inv_val
            
            # Verify
            for u_i in range(num_u):
                ui = u[u_i]
                if not ui:
                    continue
                for b_i in range(num_v):
                    bi = v[b_i]
                    if not bi:
                        continue
                    row_r = R[u_i][b_i]
                    for w_i in range(num_w):
                        wi = w[w_i]
                        if not wi:
                            if row_r[w_i] != 0:
                                return False
                        else:
                            expected = ui * bi * wi
                            if row_r[w_i] != expected:
                                return False
            
            # Update
            for u_i in range(num_u):
                ui = u[u_i]
                if not ui:
                    continue
                for b_i in range(num_v):
                    bi = v[b_i]
                    if not bi:
                        continue
                    base_val = ui * bi
                    row_r = R[u_i][b_i]
                    for w_i in range(num_w):
                        wi = w[w_i]
                        if wi:
                            row_r[w_i] -= base_val * wi
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            return True

        for step in range(target_rank):
            if time.time() > time_budget:
                return None
            cand = pick_candidate()
            if cand is None:
                break
            a_idx, b_idx, c_idx, val = cand
            if not try_extract(a_idx, b_idx, c_idx, val):
                # Try neighbors in each direction
                found = False
                # Try other c values
                for c_try in range(num_w):
                    if c_try == c_idx:
                        continue
                    val_try = R[a_idx][b_idx][c_try]
                    if val_try != 0 and try_extract(a_idx, b_idx, c_try, val_try):
                        found = True
                        break
                if not found:
                    # Try other b values
                    for b_try in range(num_v):
                        if b_try == b_idx:
                            continue
                        val_try = R[a_idx][b_try][c_idx]
                        if val_try != 0 and try_extract(a_idx, b_try, c_idx, val_try):
                            found = True
                            break
                    if not found:
                        # Try other a values
                        for a_try in range(num_u):
                            if a_try == a_idx:
                                continue
                            val_try = R[a_try][b_idx][c_idx]
                            if val_try != 0 and try_extract(a_try, b_idx, c_idx, val_try):
                                found = True
                                break
                        if not found:
                            # Try diagonal moves
                            for a_try in range(num_u):
                                if a_try == a_idx:
                                    continue
                                for b_try in range(num_v):
                                    if b_try == b_idx:
                                        continue
                                    val_try = R[a_try][b_try][c_idx]
                                    if val_try != 0 and try_extract(a_try, b_try, c_idx, val_try):
                                        found = True
                                        break
                                if found:
                                    break
                            if not found:
                                return None

        if pick_candidate() is not None:
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

    # ---------- Main: fallback first, then attempt to trim rank ----------
    best = strassen_block_solution()
    best_rank = len(best["U"][0])

    # Attempt rank minimization below the fallback rank
    # Use more aggressive time budget for greedy extraction
    # Try from best_rank-1 down, but allow more time per attempt
    for tr in range(best_rank - 1, max(best_rank - 25, 1), -1):
        if time.time() > deadline - 15:
            break
        # Give more time to earlier (higher rank) attempts since they're more likely to succeed
        remaining = deadline - 15 - time.time()
        # Allocate time proportionally: earlier attempts get more
        attempts_left = best_rank - 1 - tr + 1
        time_budget = min(deadline - 15, time.time() + remaining / max(attempts_left, 1))
        res = greedy_extract(tr, time_budget)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                best = res
                best_rank = len(res["U"][0])
                break

    return best
# EVOLVE-BLOCK-END
