"""Matrix multiplication 2x2 by 2x6: rank-21 Strassen-on-blocks decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # explicit safety margin before the 120 s wall-clock kill
    n, m, p = DIMENSIONS

    # ---- Schoolbook fallback (rank n*m*p) ----
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # ---- Strassen on three independent 2x2 blocks of B (cols 0:2, 2:4, 4:6) ----
    # A (2x2) is shared across blocks; each block's B is a 2x2 slice.
    # Each block contributes 7 rank-1 terms -> total rank = 3 * 7 = 21.
    nblocks = p // 2  # 3
    rank = nblocks * 7
    U = [[0] * rank for _ in range(n * m)]   # row = i*m + j
    V = [[0] * rank for _ in range(m * p)]   # row = j*p + k
    W = [[0] * rank for _ in range(p * n)]   # row = k*n + i  (official trace convention)

    def add_U(i, j, c, val):
        U[i * m + j][c] += val
    def add_V(j, k, c, val):
        V[j * p + k][c] += val
    def add_W(i, k, c, val):
        W[k * n + i][c] += val

    # Strassen terms for one 2x2 block. Local indices: i in {0,1}, j in {0,1}, kl in {0,1}.
    # Each term: (U_contribs [(i,j,val)], V_contribs [(j,kl,val)], W_contribs [(i,kl,val)])
    # Derived from: C11=P1+P4-P5+P7, C12=P3+P5, C21=P2+P4, C22=P1-P2+P3+P6
    terms = [
        # P1 = (A00+A11)(B00+B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # P2 = (A10+A11)B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        # P3 = A00(B01-B11)
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        # P4 = A11(B10-B00)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
        # P5 = (A00+A01)B11
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        # P6 = (A10-A00)(B00+B10)
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)], [(1, 1, 1)]),
        # P7 = (A01-A11)(B01+B11)
        ([(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]

    c = 0
    for b in range(nblocks):
        for (uc, vc, wc) in terms:
            for (i, j, val) in uc:
                add_U(i, j, c, val)
            for (j, kl, val) in vc:
                add_V(j, 2 * b + kl, c, val)
            for (i, kl, val) in wc:
                add_W(i, 2 * b + kl, c, val)
            c += 1

    # ---- Exactness verification (cheap; integer arithmetic) ----
    ok = True
    try:
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W).reshape(p, n)  # Wa[k, i]
        rng = np.random.default_rng(12345)
        for _ in range(30):
            if time.time() > deadline:
                break
            A = rng.integers(-5, 6, size=(n, m))
            B = rng.integers(-5, 6, size=(m, p))
            C_target = A @ B
            Af = A.reshape(n * m)
            Bf = B.reshape(m * p)
            u = Ua.T @ Af          # (rank,)
            v = Va.T @ Bf          # (rank,)
            C_recon = Wa.T @ (u * v)  # (n, p); C_recon[i,k] = sum_c (u_c v_c) Wa[k,i]
            if not np.array_equal(C_recon, C_target):
                ok = False
                break
    except Exception:
        ok = False

    if ok and time.time() < deadline:
        return {"U": U, "V": V, "W": W}
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
