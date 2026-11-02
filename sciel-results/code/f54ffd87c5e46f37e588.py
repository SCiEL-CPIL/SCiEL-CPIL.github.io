import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 118.0


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8

    # Strassen 2x2x2 decomposition (7 terms)
    # Local index: A[i*2+j], B[j*2+k], C[k*2+i]
    strassen_decomp = [
        ({0: 1, 3: 1},   {0: 1, 3: 1},   {0: 1, 3: 1}),    # M1
        ({2: 1, 3: 1},   {0: 1},         {2: 1, 3: -1}),   # M2
        ({0: 1},         {1: 1, 3: -1},  {1: 1, 3: 1}),    # M3
        ({3: 1},         {2: 1, 0: -1},  {0: 1, 2: 1}),    # M4
        ({0: 1, 1: 1},   {3: 1},         {1: 1, 0: -1}),   # M5
        ({2: 1, 0: -1},  {0: 1, 1: 1},   {3: 1}),          # M6
        ({1: 1, 3: -1},  {2: 1, 3: 1},   {0: 1}),          # M7
    ]

    def _build_block_strassen():
        """Rank-88: n=3 -> 2+1, m=4 -> 2+2, p=8 -> 4x2.
        Top (rows 0,1): Strassen per 2x2x2 block. Bottom (row 2): schoolbook."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        for j1 in range(2):
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[li_i * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + li_i][col] += cf
                    col += 1
            for k in range(8):
                for j2 in range(2):
                    gj = 2 * j1 + j2
                    U[2 * m + gj][col] = 1
                    V[gj * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_alt_split():
        """Alt: n=3 -> 1+2. Top row 0 schoolbook (32); bottom rows 1,2 Strassen (56)."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j in range(4):
            for k in range(8):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k * n][col] = 1
                col += 1
        for j1 in range(2):
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[(1 + li_i) * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + (1 + li_i)][col] += cf
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W, tol=1e-9):
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j] * Va[j * p + k], Wa[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _verify_exact(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for col in range(r):
                        s += Fraction(U[i * m + j][col]) * Fraction(V[j * p + k][col]) * Fraction(W[k * n + i][col])
                    if s != 1:
                        return False
        return True

    def _tensor_matrix(U, V, W):
        """Build tensor matrix: each column is U_c ⊗ V_c ⊗ W_c flattened."""
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T = np.zeros((n * m * p, r))
        pos = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for col in range(r):
                        T[pos, col] = Ua[i * m + j, col] * Va[j * p + k, col] * Wa[k * n + i, col]
                    pos += 1
        return T

    def _reduce(U, V, W, budget):
        """Tensor-span reduction: remove rank terms whose tensor is in span of others."""
        r = len(U[0])
        tensors = []
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for idx in range(r):
            t = np.zeros(n * m * p)
            pos = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t[pos] = Ua[i * m + j, idx] * Va[j * p + k, idx] * Wa[k * n + i, idx]
                        pos += 1
            tensors.append(t)

        T_mat = np.column_stack(tensors)
        keep = list(range(r))

        for _pass in range(5):
            if time.time() > budget:
                break
            removed_any = False
            current_r = len(keep)
            T_cur = np.column_stack([tensors[i] for i in keep])
            cur_rank = np.linalg.matrix_rank(T_cur, tol=1e-9)
            for idx_pos in range(current_r):
                if time.time() > budget:
                    break
                idx = keep[idx_pos]
                cols = [keep[c] for c in range(current_r) if c != idx_pos]
                if not cols:
                    continue
                sub_rank = np.linalg.matrix_rank(np.column_stack([tensors[i] for i in cols]), tol=1e-9)
                if sub_rank == cur_rank:
                    keep = [c for c in keep if c != idx]
                    cur_rank = sub_rank
                    removed_any = True
            if not removed_any:
                break

        U_red = [[U[i][c] for c in keep] for i in range(len(U))]
        V_red = [[V[i][c] for c in keep] for i in range(len(V))]
        W_red = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_red, V_red, W_red, len(keep)

    def _local_search_merge(U, V, W, budget):
        """Try to merge pairs of terms that sum to a single rank-1 tensor."""
        r = len(U[0])
        keep = list(range(r))
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # For each pair (a, b), check if U_a⊗V_a⊗W_a + U_b⊗V_b⊗W_b = α·(U_c⊗V_c⊗W_c)
        # for some existing c, or if the sum is itself rank-1.
        # Simpler: check if removing one of the pair and adding their sum works.
        
        tensors = []
        for idx in range(r):
            t = np.zeros(n * m * p)
            pos = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t[pos] = Ua[i * m + j, idx] * Va[j * p + k, idx] * Wa[k * n + i, idx]
                        pos += 1
            tensors.append(t)
        
        for _pass in range(3):
            if time.time() > budget:
                break
            changed = False
            current_keep = list(keep)
            T_cur = np.column_stack([tensors[i] for i in current_keep])
            cur_rank = np.linalg.matrix_rank(T_cur, tol=1e-9)
            
            for a in range(len(current_keep)):
                if time.time() > budget:
                    break
                for b in range(a + 1, len(current_keep)):
                    ta = tensors[current_keep[a]]
                    tb = tensors[current_keep[b]]
                    tsum = ta + tb
                    # Check if tsum is in span of remaining tensors
                    others = [current_keep[c] for c in range(len(current_keep)) if c != a and c != b]
                    if not others:
                        # tsum must be zero
                        if np.allclose(tsum, 0, atol=1e-9):
                            keep = [c for c in current_keep if c != a and c != b]
                            changed = True
                            break
                        continue
                    T_others = np.column_stack([tensors[i] for i in others])
                    T_with_sum = np.column_stack([T_others, tsum.reshape(-1, 1)])
                    rank_others = np.linalg.matrix_rank(T_others, tol=1e-9)
                    rank_with = np.linalg.matrix_rank(T_with_sum, tol=1e-9)
                    if rank_with == rank_others:
                        # Replace a and b with their sum
                        new_keep = [c for c in current_keep if c != a and c != b]
                        # Need to add new term: we'll track it separately
                        # For simplicity, just note that rank didn't increase
                        # But we need to actually construct the new term...
                        # This is complex; skip for now
                        pass
                if changed:
                    break
            if not changed:
                break
        
        U_red = [[U[i][c] for c in keep] for i in range(len(U))]
        V_red = [[V[i][c] for c in keep] for i in range(len(V))]
        W_red = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_red, V_red, W_red, len(keep)

    best = None
    best_rank = 10 ** 9

    builders = [_build_block_strassen, _build_alt_split]
    for builder in builders:
        if time.time() > DEADLINE:
            break
        sol = builder()
        if sol is None:
            continue
        U, V, W = sol["U"], sol["V"], sol["W"]
        if not _verify(U, V, W):
            continue
        # Try reduction methods
        for reducer in (_reduce, _local_search_merge):
            if time.time() > DEADLINE:
                break
            try:
                U2, V2, W2, rk = reducer(U, V, W, DEADLINE)
                if rk < best_rank and _verify(U2, V2, W2):
                    best = {"U": U2, "V": V2, "W": W2}
                    best_rank = rk
            except Exception:
                pass
        if len(U[0]) < best_rank and _verify(U, V, W):
            best = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])

    if best is None:
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
        best = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best
