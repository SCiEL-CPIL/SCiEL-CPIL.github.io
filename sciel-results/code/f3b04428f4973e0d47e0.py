"""Rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_U, dim_V, dim_W = n * m, m * p, p * n   # 6, 15, 10
    rank_school = n * m * p                       # 30

    # Build the list of (a, b, c) triples where the tensor equals 1.
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    T = np.array(triples)
    num_t = T.shape[0]
    A_idx, B_idx, C_idx = T[:, 0], T[:, 1], T[:, 2]

    # Precompute the schoolbook basis columns (rank 30).
    U0 = np.zeros((dim_U, rank_school))
    V0 = np.zeros((dim_V, rank_school))
    W0 = np.zeros((dim_W, rank_school))
    for s, (a, b, c) in enumerate(triples):
        U0[a, s] = 1.0
        V0[b, s] = 1.0
        W0[c, s] = 1.0

    def verify(U, V, W, r, tol):
        # error over the 30 triples
        vals = np.einsum('as,bs,cs->t', U[A_idx], V[B_idx], W[C_idx])
        err = float(np.sum((vals - 1.0) ** 2))
        return err

    def als(U, V, W, r, max_iter, tol):
        """Alternating least squares for the degree-3 tensor equation."""
        for _ in range(max_iter):
            # Update U: for each triple, sum_s U[a,s]*(V[b,s]*W[c,s]) = 1
            M = np.zeros((num_t, r))
            M[np.arange(num_t), np.arange(num_t)] = 0  # placeholder
            # Build matrix: rows=triples, cols=U[a,s] -> index a*r+s
            AU = np.zeros((num_t, dim_U * r))
            for s in range(r):
                AU[:, A_idx * r + s] = V[B_idx, s] * W[C_idx, s]
            sol, *_ = np.linalg.lstsq(AU, np.ones(num_t), rcond=None)
            U = sol.reshape(dim_U, r)

            AV = np.zeros((num_t, dim_V * r))
            for s in range(r):
                AV[:, B_idx * r + s] = U[A_idx, s] * W[C_idx, s]
            sol, *_ = np.linalg.lstsq(AV, np.ones(num_t), rcond=None)
            V = sol.reshape(dim_V, r)

            AW = np.zeros((num_t, dim_W * r))
            for s in range(r):
                AW[:, C_idx * r + s] = U[A_idx, s] * V[B_idx, s]
            sol, *_ = np.linalg.lstsq(AW, np.ones(num_t), rcond=None)
            W = sol.reshape(dim_W, r)

            err = verify(U, V, W, r, tol)
            if err < tol:
                break
        return U, V, W, verify(U, V, W, r, tol)

    def polish(U, V, W, r):
        """High-precision polish: more ALS iterations with tighter tol."""
        for _ in range(40):
            AU = np.zeros((num_t, dim_U * r))
            for s in range(r):
                AU[:, A_idx * r + s] = V[B_idx, s] * W[C_idx, s]
            sol, *_ = np.linalg.lstsq(AU, np.ones(num_t), rcond=None)
            U = sol.reshape(dim_U, r)
            AV = np.zeros((num_t, dim_V * r))
            for s in range(r):
                AV[:, B_idx * r + s] = U[A_idx, s] * W[C_idx, s]
            sol, *_ = np.linalg.lstsq(AV, np.ones(num_t), rcond=None)
            V = sol.reshape(dim_V, r)
            AW = np.zeros((num_t, dim_W * r))
            for s in range(r):
                AW[:, C_idx * r + s] = U[A_idx, s] * V[B_idx, s]
            sol, *_ = np.linalg.lstsq(AW, np.ones(num_t), rcond=None)
            W = sol.reshape(dim_W, r)
            if verify(U, V, W, r, 1e-14) < 1e-14:
                break
        return U, V, W

    def to_halfint(U, V, W, r):
        # Round to nearest half-integer and verify
        Uh = (np.round(U * 2.0) / 2.0)
        Vh = (np.round(V * 2.0) / 2.0)
        Wh = (np.round(W * 2.0) / 2.0)
        if verify(Uh, Vh, Wh, r, 1e-9) < 1e-9:
            return Uh, Vh, Wh
        return None

    start = time.time()
    deadline = start + 140.0   # safety margin before 150s kill

    # Fallback: schoolbook rank 30
    best = (rank_school, U0.copy(), V0.copy(), W0.copy())

    # Try ranks from small to large
    for r in range(1, rank_school):
        if time.time() > deadline - 5:
            break

        # Budget restarts for this rank
        remaining = deadline - time.time()
        if remaining < 2:
            break
        # More restarts for smaller ranks (harder), fewer as we go up
        n_restarts = max(3, int(remaining // 0.5))
        n_restarts = min(n_restarts, 200)

        found = False
        for restart in range(n_restarts):
            if time.time() > deadline - 3:
                break

            np.random.seed(restart * 7919 + r * 131)

            # Initialization: warm start from a random subset of schoolbook columns
            # plus a small random component, to give ALS a near-valid starting point.
            if r <= rank_school:
                cols = np.random.choice(rank_school, size=r, replace=False)
                U = U0[:, cols].copy()
                V = V0[:, cols].copy()
                W = W0[:, cols].copy()
                # Add mild noise to escape degenerate basins
                noise = 0.1
                U += noise * np.random.randn(dim_U, r)
                V += noise * np.random.randn(dim_V, r)
                W += noise * np.random.randn(dim_W, r)
            else:
                U = np.random.randn(dim_U, r) * 0.5
                V = np.random.randn(dim_V, r) * 0.5
                W = np.random.randn(dim_W, r) * 0.5

            # Coarse ALS
            U, V, W, err = als(U, V, W, r, max_iter=60, tol=1e-8)
            if err < 1e-8:
                # Polish to high precision
                U, V, W = polish(U, V, W, r)
                if verify(U, V, W, r, 1e-12) < 1e-12:
                    # Try half-integer rounding
                    hi = to_halfint(U, V, W, r)
                    if hi is not None:
                        Uh, Vh, Wh = hi
                    else:
                        Uh, Vh, Wh = U, V, W
                    best = (r, Uh.copy(), Vh.copy(), Wh.copy())
                    found = True
                    break

        if found:
            # Found a valid decomposition at rank r; since we go small->large,
            # this is the minimal rank we found. Continue to try even smaller?
            # No, we already tried smaller. Stop.
            break

    r_best, U_best, V_best, W_best = best

    # Final verification of the returned solution
    if verify(U_best, V_best, W_best, r_best, 1e-9) > 1e-9:
        # Fallback to schoolbook
        r_best, U_best, V_best, W_best = rank_school, U0, V0, W0

    U_list = [[float(x) for x in row] for row in U_best]
    V_list = [[float(x) for x in row] for row in V_best]
    W_list = [[float(x) for x in row] for row in W_best]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
