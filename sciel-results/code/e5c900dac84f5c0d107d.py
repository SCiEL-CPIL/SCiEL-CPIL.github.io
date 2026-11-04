"""Matrix multiplication 2x3x8 with optimized Strassen-based decomposition.
Attempts to reduce rank below 44 by merging terms across the 2x2 and 2x1 splits."""
import time
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build the rank-44 decomposition as baseline
    def _build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8), split B1 into 4 blocks of 2 columns
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li*2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] += u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la*2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] += v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk*2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] += w_val

                col += 1
                if not _elapsed_ok():
                    return None

        # Part 2: A2 (2x1) * B2 (1x8)
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                if not _elapsed_ok():
                    return None

        return {"U": U, "V": V, "W": W}

    baseline = _build_rank44()
    if baseline is None or not _validate(baseline):
        return _schoolbook()

    # Try to reduce rank by merging Part 2 terms with Part 1 terms
    U_work = [row[:] for row in baseline["U"]]
    V_work = [row[:] for row in baseline["V"]]
    W_work = [row[:] for row in baseline["W"]]
    
    part2_cols = list(range(28, 44))
    part1_cols = list(range(0, 28))
    
    removed_count = 0
    for p2_col in part2_cols:
        if not _elapsed_ok():
            break
            
        idx_in_part2 = p2_col - 28
        i_val = idx_in_part2 // 8
        k_val = idx_in_part2 % 8
        
        absorbed = False
        for p1_col in part1_cols:
            if not _elapsed_ok():
                break
                
            trial_U = [row[:] for row in U_work]
            trial_V = [row[:] for row in V_work]
            trial_W = [row[:] for row in W_work]
            
            u_idx = i_val * m + 2
            v_idx = 2 * p + k_val
            w_idx = k_val * n + i_val
            
            trial_U[u_idx][p1_col] += 1
            trial_V[v_idx][p1_col] += 1
            trial_W[w_idx][p1_col] += 1
            
            for row in trial_U:
                row[p2_col] = 0
            for row in trial_V:
                row[p2_col] = 0
            for row in trial_W:
                row[p2_col] = 0
            
            trial_candidate = {"U": trial_U, "V": trial_V, "W": trial_W}
            if _validate(trial_candidate):
                U_work = trial_U
                V_work = trial_V
                W_work = trial_W
                removed_count += 1
                absorbed = True
                break
    
    if removed_count > 0 and _elapsed_ok():
        final_candidate = {"U": U_work, "V": V_work, "W": W_work}
        if _validate(final_candidate):
            return final_candidate

    return baseline

