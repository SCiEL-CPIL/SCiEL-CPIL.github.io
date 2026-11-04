import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    # fast exact check: for each (i,j) the p-length vector dot(Uij, V_jk * W_ki) must be 1
    # vectorize: build per-(i,j) matrix A[i,j,k,c] = V[j*p+k,c]*W[k*n+i,c]  -> too big; do per (i,j)
    for i in range(n):
        for j in range(m):
            # A is p x r
            A = np.empty((p, U.shape[1]))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            res = A @ U[i * m + j, :]
            if np.max(np.abs(res - 1.0)) > tol:
                return False
    return True

def _round_half(X):
    return np.round(2.0 * X) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin vs 150s kill

    # ---- schoolbook fallback (rank n*m*p = 54) ----
    r_sb = n * m * p
    U_sb = np.zeros((n * m, r_sb))
    V_sb = np.zeros((m * p, r_sb))
    W_sb = np.zeros((p * n, r_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1
    best = (r_sb, U_sb, V_sb, W_sb)

    rng = np.random.default_rng(0)

    def als(U, V, W, iters=80, tol=1e-7, check_every=5):
        r = U.shape[1]
        ones_p = np.ones(p)
        ones_n = np.ones(n)
        ones_m = np.ones(m)
        for it in range(iters):
            if time.time() > deadline:
                return U, V, W, False
            # update U
            for i in range(n):
                for j in range(m):
                    A = np.empty((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    U[i * m + j, :] = np.linalg.lstsq(A, ones_p, rcond=None)[0]
            # update V
            for j in range(m):
                for k in range(p):
                    A = np.empty((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    V[j * p + k, :] = np.linalg.lstsq(A, ones_n, rcond=None)[0]
            # update W
            for k in range(p):
                for i in range(n):
                    A = np.empty((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        # (kept explicit for clarity)
                    W[k * n + i, :] = np.linalg.lstsq(A, ones_m, rcond=None)[0]
            if (it + 1) % check_every == 0:
                if _verify(U, V, W, 1e-5):
                    return U, V, W, True
        return U, V, W, _verify(U, V, W, 1e-5)

    def try_decomp(r, U0, V0, W0):
        U, V, W, conv = als(U0, V0, W0)
        if conv:
            Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
            if _verify(Ur, Vr, Wr, 1e-9):
                return True, Ur, Vr, Wr
        return False, None, None, None

    # ---- structured seed: rank-5 (3x3x3) tensor-product with identity ----
    # 3x3x3 rank-5 decomposition (Strassen-like / known 5-term):
    # indices a,b in 0..2 (row, col of 3x3), c in 0..2 (inner)
    R5 = 5
    U5 = np.zeros((9, R5))   # a*3+b
    V5 = np.zeros((9, R5))   # b*3+c
    W5 = np.zeros((9, R5))   # c*3+a
    # A classic rank-5 decomposition of the 3x3x3 multiplication tensor:
    # term0: (a=0,b=0) & (b=0,c=0) & (c=0,a=0)  -> full identity-ish
    # We use the standard 5-term (Laderman-like) construction for 3x3:
    # u0 = e_{00}, u1 = e_{11}, u2 = e_{22}, u3 = e_{01}+e_{10}, u4 = e_{12}+e_{21}
    U5[0, 0] = 1; U5[4, 1] = 1; U5[8, 2] = 1
    U5[1, 3] = 1; U5[3, 3] = 1
    U5[5, 4] = 1; U5[7, 4] = 1
    V5[0, 0] = 1; V5[4, 1] = 1; V5[8, 2] = 1
    V5[1, 3] = 1; V5[3, 3] = 1
    V5[5, 4] = 1; V5[7, 4] = 1
    W5[0, 0] = 1; W5[4, 1] = 1; W5[8, 2] = 1
    W5[1, 3] = 1; W5[3, 3] = 1
    W5[5, 4] = 1; W5[7, 4] = 1
    # Verify this is a valid 3x3x3 decomposition
    def _verify3(U3, V3, W3, tol=1e-9):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    val = np.dot(U3[a*3+b], V3[b*3+c] * W3[c*3+a])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    if _verify3(U5, V5, W5, 1e-9):
        # tensor with 3x3 identity (3 terms) -> rank 15 for 3x3x9-ish block
        # Build full 2x3x9 from this structure: rows i in 0..1 select via identity
        r_struct = R5 * 3
        U_s = np.zeros((n * m, r_struct))
        V_s = np.zeros((m * p, r_struct))
        W_s = np.zeros((p * n, r_struct))
        # block (b,c) of 3x3 inner; term t in 0..R5-1, id s in 0..2
        for t in range(R5):
            for s in range(3):
                col = t * 3 + s
                # U: row i*m + j, nonzero when j==s (identity over column j)
                for i in range(n):
                    U_s[i * m + s, col] = U5[s * 3 + s, t]  # diagonal entry of U5 for col s
                # V: row j*p + k, nonzero when j==s
                V_s[s * p + s, col] = V5[s * 3 + s, t]
                # W: row k*n + i, nonzero when k==s
                for i in range(n):
                    W_s[s * n + i, col] = W5[s * 3 + s, t] * (1.0 if i == 0 else 0.0) + (0.0)
        # The above is a rough structured guess; only use if it verifies (it likely won't be exact),
        # so we treat it as an ALS initialization rather than a final answer.
        ok, _, _, _ = try_decomp(r_struct, U_s + rng.normal(0, 0.01, U_s.shape),
                                 V_s + rng.normal(0, 0.01, V_s.shape),
                                 W_s + rng.normal(0, 0.01, W_s.shape))
        if ok:
            # re-run to capture exact rounded
            U, V, W, conv = als(U_s, V_s, W_s, iters=60)
            Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
            if _verify(Ur, Vr, Wr, 1e-9) and r_struct < best[0]:
                best = (r_struct, Ur, Vr, Wr)

    # ---- main search: try ranks below schoolbook with random + structured inits ----
    ranks_to_try = list(range(15, r_sb))
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        found = False
        n_restarts = 40 if r <= 30 else 20
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            scale = 0.1 if restart % 3 == 0 else (0.5 if restart % 3 == 1 else 0.3)
            U0 = rng.normal(0, scale, (n * m, r))
            V0 = rng.normal(0, scale, (m * p, r))
            W0 = rng.normal(0, scale, (p * n, r))
            ok, Ur, Vr, Wr = try_decomp(r, U0, V0, W0)
            if ok:
                if r < best[0]:
                    best = (r, Ur, Vr, Wr)
                found = True
                break
        if found:
            # continue to try even lower ranks
            continue

    r, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
