"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0: V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0: W[r][c] += W1[r][c]

        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0: 1, 1: 2, 2: 4, 3: 5}
        w2_map = {0: 2, 1: 4, 2: 3, 3: 5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base + c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base + c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base + c] += W2[r][c]
        return U, V, W

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        strassen_terms = [
            ([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], [(0,0,1), (0,2,1), (2,0,1), (2,2,-1)]),
            ([(2,0,1), (2,2,1)], [(0,0,1)], [(2,0,1), (2,2,-1)]),
            ([(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1), (2,2,1)]),
            ([(2,2,1)], [(2,0,1), (0,0,-1)], [(0,0,1), (2,0,1)]),
            ([(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,-1), (0,2,1)]),
            ([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(2,2,1)]),
            ([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,1), (2,2,1)]),
        ]

        for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    for a_block in a_blocks:
                        a_ro, a_co, coeff = a_block
                        g_u = (i_local + a_ro) * m + (j_local + a_co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3
                    k_local = v_local % 3
                    for b_block in b_blocks:
                        b_ro, b_co, coeff = b_block
                        g_v = (j_local + b_ro) * p + (k_local + b_co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    for c_block in c_blocks:
                        c_ro, c_co, coeff = c_block
                        g_w = (k_local + c_ro) * n + (i_local + c_co)
                        W[g_w][base_col + c] += w_val * coeff
        return U, V, W

    def make_trace(U, V, W, n, m, p):
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                u_row = Ua[i*m + j, :]
                for k in range(p):
                    T[i, j, k] = np.dot(u_row * Va[j*p + k, :], Wa[k*n + i, :])
        return T

    def verify(U, V, W, n, m, p, tol=1e-7):
        T = make_trace(U, V, W, n, m, p)
        return bool(np.all(np.abs(T - 1.0) <= tol))

    def try_reduce_rank(U, V, W, n, m, p, time_budget):
        """Aggressive rank reduction using linear algebra."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        R = Ua.shape[1]
        if R == 0: return U, V, W, R

        def column_contribution(r):
            u_col = Ua[:, r]
            v_col = Va[:, r]
            w_col = Wa[:, r]
            c = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    uv = u_col[i*m + j]
                    if uv == 0: continue
                    for k in range(p):
                        c[i, j, k] = uv * v_col[j*p + k] * w_col[k*n + i]
            return c

        keep = list(range(R))
        contribs = {}
        for r in range(R):
            if time.time() > time_budget: break
            contribs[r] = column_contribution(r)

        changed = True
        while changed and time.time() < time_budget:
            changed = False
            for r in list(keep):
                if time.time() > time_budget: break
                c = contribs.get(r)
                if c is None: continue
                if np.all(np.abs(c) <= 1e-9):
                    keep.remove(r)
                    changed = True
                    break

        if len(keep) < R:
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            R = len(keep)

        if time.time() < time_budget - 5 and R > 1:
            nmp = n * m * p
            M = np.zeros((nmp, R), dtype=float)
            for r in range(R):
                c = contribs.get(r)
                if c is None:
                    c = column_contribution(r)
                    contribs[r] = c
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M[i*m*p + j*p + k, r] = c[i, j, k]
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            except:
                pass

        if len(keep) == Ua.shape[1]:
            return U, V, W, R
        return Ua.tolist(), Va.tolist(), Wa.tolist(), R

    U, V, W = build_4x4x6()
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    time_budget = min(deadline - 10, time.time() + 90)
    U_r, V_r, W_r, r_rank = try_reduce_rank(U, V, W, n, m, p, time_budget)
    if verify(U_r, V_r, W_r, n, m, p):
        return {"U": U_r, "V": V_r, "W": W_r}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
