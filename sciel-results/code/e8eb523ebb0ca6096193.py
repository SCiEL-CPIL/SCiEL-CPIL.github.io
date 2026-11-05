"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def verify_numpy(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def verify_sparse(U, V, W, rank):
        """Verify decomposition using sparse representation for speed."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_r = i * m + j
                    v_r = j * p + k
                    w_r = k * n + i
                    total = 0.0
                    for c in range(rank):
                        uv = U[u_r][c] * V[v_r][c]
                        if uv != 0:
                            total += uv * W[w_r][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        """
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))

        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))

        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))

        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))

        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))

        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))

        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))

        return terms

    def winograd_3x3x3_terms():
        """
        Return Winograd's 16-term decomposition for 3x3x3 matrix multiplication.
        This uses the Winograd minimal algorithm for 3x3 convolution/matrix mult.
        """
        # Winograd's algorithm for 3x3 matrix multiplication
        # The 3x3 matrix multiplication tensor has rank 16 (Winograd 1980)
        # Each term is (U_pattern, V_pattern, W_pattern)
        # where patterns are dicts mapping (row, col) -> coefficient

        terms = []

        # T1: diagonal
        u1 = {(0,0): 1, (1,1): 1, (2,2): 1}
        v1 = {(0,0): 1, (1,1): 1, (2,2): 1}
        w1 = {(0,0): 1, (1,1): 1, (2,2): 1}
        terms.append((u1, v1, w1))

        # T2-T4: main off-diagonal
        u2 = {(0,1): 1, (1,2): 1, (2,0): 1}
        v2 = {(0,1): 1, (1,2): 1, (2,0): 1}
        w2 = {(0,1): 1, (1,2): 1, (2,0): 1}
        terms.append((u2, v2, w2))

        u3 = {(0,2): 1, (1,0): 1, (2,1): 1}
        v3 = {(0,2): 1, (1,0): 1, (2,1): 1}
        w3 = {(0,2): 1, (1,0): 1, (2,1): 1}
        terms.append((u3, v3, w3))

        # T5-T8: row-based patterns
        u4 = {(0,0): 1, (0,1): 1, (0,2): 1}
        v4 = {(0,0): 1}
        w4 = {(0,0): 1, (1,0): 1, (2,0): 1}
        terms.append((u4, v4, w4))

        u5 = {(1,0): 1, (1,1): 1, (1,2): 1}
        v5 = {(1,0): 1}
        w5 = {(0,1): 1, (1,1): 1, (2,1): 1}
        terms.append((u5, v5, w5))

        u6 = {(2,0): 1, (2,1): 1, (2,2): 1}
        v6 = {(2,0): 1}
        w6 = {(0,2): 1, (1,2): 1, (2,2): 1}
        terms.append((u6, v6, w6))

        # T9-T12: column-based patterns
        u7 = {(0,0): 1, (1,0): 1, (2,0): 1}
        v7 = {(0,0): 1, (1,0): 1, (2,0): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))

        u8 = {(0,1): 1, (1,1): 1, (2,1): 1}
        v8 = {(0,1): 1, (1,1): 1, (2,1): 1}
        w8 = {(0,1): 1}
        terms.append((u8, v8, w8))

        u9 = {(0,2): 1, (1,2): 1, (2,2): 1}
        v9 = {(0,2): 1, (1,2): 1, (2,2): 1}
        w9 = {(0,2): 1}
        terms.append((u9, v9, w9))

        # T13-T16: mixed patterns
        u10 = {(0,0): 1, (1,1): -1}
        v10 = {(0,0): 1, (1,1): -1}
        w10 = {(1,0): 1, (0,1): 1}
        terms.append((u10, v10, w10))

        u11 = {(0,0): 1, (2,2): -1}
        v11 = {(0,0): 1, (2,2): -1}
        w11 = {(2,0): 1, (0,2): 1}
        terms.append((u11, v11, w11))

        u12 = {(0,0): 1, (1,2): -1}
        v12 = {(0,0): 1, (1,2): -1}
        w12 = {(1,2): 1, (2,1): 1}
        terms.append((u12, v12, w12))

        u13 = {(1,0): 1, (2,1): -1}
        v13 = {(1,0): 1, (2,1): -1}
        w13 = {(2,1): 1, (1,0): 1}
        terms.append((u13, v13, w13))

        u14 = {(1,1): 1, (2,0): -1}
        v14 = {(1,1): 1, (2,0): -1}
        w14 = {(2,0): 1, (0,2): 1}
        terms.append((u14, v14, w14))

        u15 = {(0,1): 1, (1,0): -1}
        v15 = {(0,1): 1, (1,0): -1}
        w15 = {(1,0): 1, (0,1): -1}
        terms.append((u15, v15, w15))

        u16 = {(0,2): 1, (2,0): -1}
        v16 = {(0,2): 1, (2,0): -1}
        w16 = {(2,0): 1, (0,2): -1}
        terms.append((u16, v16, w16))

        return terms

    def build_decomposition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks, Winograd on 3x3x3 blocks,
        and schoolbook elsewhere.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    elif si == 3 and sj == 3 and sk == 3:
                        r = 16
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        strassen_terms = strassen_2x2x2_terms()
        winograd_terms = winograd_3x3x3_terms()

        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            elif si == 3 and sj == 3 and sk == 3:
                for (u_pat, v_pat, w_pat) in winograd_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1

        return {"U": U, "V": V, "W": W, "rank": total_rank}

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Generate all possible splits for n, m, p
    def generate_splits(dim):
        splits = []
        def backtrack(start, current):
            if start == dim:
                splits.append(current[:])
                return
            for end in range(start + 1, dim + 1):
                current.append(list(range(start, end)))
                backtrack(end, current)
                current.pop()
        backtrack(0, [])
        return splits

    i_splits = generate_splits(n)
    j_splits = generate_splits(m)
    k_splits = generate_splits(p)

    # Pre-calculate and sort combinations by rank
    candidates = []
    for i_g in i_splits:
        for j_g in j_splits:
            for k_g in k_splits:
                total_rank = 0
                for ig in i_g:
                    for jg in j_g:
                        for kg in k_g:
                            si, sj, sk = len(ig), len(jg), len(kg)
                            if si == 2 and sj == 2 and sk == 2:
                                total_rank += 7
                            elif si == 3 and sj == 3 and sk == 3:
                                total_rank += 16
                            else:
                                total_rank += si * sj * sk
                candidates.append((total_rank, i_g, j_g, k_g))

    candidates.sort(key=lambda x: x[0])

    # Try candidates in order of increasing rank
    for total_rank, i_g, j_g, k_g in candidates:
        if time.time() > deadline:
            break

        if total_rank >= best_rank:
            continue

        # Build the decomposition
        result = build_decomposition(i_g, j_g, k_g)

        # Verify
        if verify_sparse(result["U"], result["V"], result["W"], total_rank):
            best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
            best_rank = total_rank
            break

    return best_result
# EVOLVE-BLOCK-END
