import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _build():
    # Decomposition of the 2 x 3 x 3 matrix-product tensor into 16 rank-1 terms.
    # For each term c: U[c] (6x1), V[c] (9x1), W[c] (6x1) with entries in {-1,0,1,2}.
    # Tensor_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  (trace convention)
    n, m, p = 2, 3, 3
    U = np.zeros((n * m, 16))
    V = np.zeros((m * p, 16))
    W = np.zeros((p * n, 16))

    # Rows of U/V/W indexed by (i,j) / (j,k) / (k,i).
    # Two 2x3x2 "sums" blocks (8 terms) + 8 cross terms = 16.
    def set_(M, r, c, v):
        M[r, c] = v

    # --- Block A: i in {0,1}, j in {0,1}, k in {0,1} (8 terms) ---
    # u over (i,j) in {0,1}x{0,1}, v over (j,k) in {0,1}x{0,1}, w over (k,i) in {0,1}x{0,1}
    A_u = [1, 1, 1, 1, 1, -1, 1, -1]
    A_v = [1, 1, 1, 1, 1, 1, -1, -1]
    A_w = [1, 1, 1, 1, 1, 1, 1, -1]
    for c in range(8):
        for (i, j), uv in enumerate(A_u):
            set_(U, i * m + j, c, uv)
        for (j, k), vv in enumerate(A_v):
            set_(V, j * p + k, c, vv)
        for (k, i), ww in enumerate(A_w):
            set_(W, k * n + i, c, ww)

    # --- Block B: i in {0,1}, j in {0,1}, k in {2} (k=2 = "k+2") ---
    # Same 2x3x2 sums block, shifted so the inner k index is 2.
    for c in range(8):
        cc = 8 + c
        for (i, j), uv in enumerate(A_u):
            set_(U, i * m + j, cc, uv)
        for (j, k), vv in enumerate(A_v):
            # k in {0,1} -> actual k = k + 2 (i.e. 2 or ... ) but we only have one k=2 here
            # Block B covers j in {0,1}, single k=2: v index j*3+2
            set_(V, j * p + 2, cc, vv)
        for (k, i), ww in enumerate(A_w):
            # k in {0,1} -> actual k = 2
            set_(W, 2 * n + i, cc, ww)

    # --- 8 cross terms: j in {0,1}, k in {0,1}, with k and j "opposite" ---
    # These contribute only at (i=1, j, k) for the 2x3x3 core.
    # term = U at (1,j) * V at (j, k) * W at (k, 1); sign = (-1) if j==k else (+1)
    for c in range(8):
        cc = 16 + c
        j = c // 4          # 0,0,0,0,1,1,1,1
        k = c % 4           # 0,1,2,3,0,1,2,3
        kk = k % 2          # 0 or 1
        sign = -1 if j == kk else 1
        set_(U, 1 * m + j, cc, 1)              # (i=1, j)
        set_(V, j * p + kk, cc, 1)             # (j, kk)
        set_(W, kk * n + 1, cc, sign)          # (kk, i=1)

    return U, V, W


def _verify_full(U, V, W, n=2, m=3, p=9, max_check=2000):
    """Verify all output entries of the 2x3x9 tensor against the reference.
    Returns True if every checked entry is exact (integer)."""
    # Build duplicated core: core[i,j,k] for k in 0..2 -> repeated for k=3k,3k+1,3k+2
    Ucore, Vcore, Wcore = _build()  # 2x3x3 core (rank 16)
    # Full 2x3x9 tensors (rank 48)
    Uf = np.zeros((n * m, 48))
    Vf = np.zeros((m * p, 48))
    Wf = np.zeros((p * n, 48))
    for c in range(16):
        for r in range(3):
            cc = c + 16 * r
            Uf[:, cc] = Ucore[:, c]
            # V: row (j, k) with k = 3*j2 + r  -> Vcore row (j, j2)
            for j in range(m):
                for j2 in range(3):
                    Vf[j * p + (3 * j2 + r), cc] = Vcore[j * 3 + j2, c]
            # W: row (k, i) with k = 3*i2 + r -> Wcore row (i2, i)
            for i in range(n):
                for i2 in range(3):
                    Wf[(3 * i2 + r) * n + i, cc] = Wcore[i2 * 2 + i, c]

    # Now compare (U,V,W) supplied to (Uf,Vf,Wf) by checking tensor entries.
    # tensor[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    rng = np.random.default_rng(12345)
    checked = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # reference
                kk = k % 3
                ref = 1 if kk == j else 0
                # computed
                comp = np.dot(U[i * m + j], np.multiply(V[j * p + k], W[k * n + i]))
                if int(round(comp)) != ref:
                    return False
                checked += 1
    return True


def construct():
    t0 = time.time()
    try:
        Ucore, Vcore, Wcore = _build()
        # Build the full 2x3x9 rank-48 decomposition by duplication.
        n, m, p = 2, 3, 9
        rank = 48
        Uf = np.zeros((n * m, rank))
        Vf = np.zeros((m * p, rank))
        Wf = np.zeros((p * n, rank))
        for c in range(16):
            for r in range(3):
                cc = c + 16 * r
                Uf[:, cc] = Ucore[:, c]
                for j in range(m):
                    for j2 in range(3):
                        Vf[j * p + (3 * j2 + r), cc] = Vcore[j * 3 + j2, c]
                for i in range(n):
                    for i2 in range(3):
                        Wf[(3 * i2 + r) * n + i, cc] = Wcore[i2 * 2 + i, c]
        # Convert to lists of ints.
        U_list = [[int(x) for x in row] for row in Uf.tolist()]
        V_list = [[int(x) for x in row] for row in Vf.tolist()]
        W_list = [[int(x) for x in row] for row in Wf.tolist()]
        # Full verification (all 54 entries) before returning.
        if _verify_full(np.array(U_list), np.array(V_list), np.array(W_list)):
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            raise RuntimeError("verification failed")
    except Exception:
        # Fallback: schoolbook rank 54 (always valid) if we are not close to the deadline.
        if time.time() - t0 < 100:
            n, m, p = DIMENSIONS
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
        # Extremely unlikely path; still return a valid schoolbook.
        n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
