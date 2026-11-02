import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _make_atoms(n, m, p):
    # atoms: for each (i,j,k) the bilinear form A[i,j]*B[j,k]
    # represent each atom as a 6x12 matrix (A_flat outer B_flat)
    atoms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M = np.zeros((n*m, m*p))
                M[i*m+j, j*p+k] = 1.0
                atoms.append(M)
    return np.array(atoms)  # shape (24, 6, 12)

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    dimA = n*m   # 6
    dimB = m*p   # 12
    dimC = n*p   # 8
    atoms = _make_atoms(n, m, p)  # (24, 6, 12)

    def schoolbook():
        rank = n*m*p
        U = np.zeros((dimA, rank)); V = np.zeros((dimB, rank)); W = np.zeros((dimC, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    best = None
    best_rank = None
    U0, V0, W0 = schoolbook()
    best = (U0, V0, W0); best_rank = U0.shape[1]

    rng = np.random.default_rng(12345)

    def try_rank(r, n_trials=200):
        # We need span{s_c t_c} to contain all atoms.
        # s_c is a row vector in R^6 (u_c flattened), t_c in R^12.
        # Build S (r x 6), T (r x 12). s_c t_c = outer(s_c, t_c) as 6x12.
        # We need each atom M (6x12) = sum_c wcoeff... but w is per-output.
        # Actually C[i,k] = sum_c w_c[i,k] (s_c A)(t_c B).
        # (s_c A)(t_c B) as bilinear form = outer(s_c, t_c) (6x12 matrix, A_flat outer B_flat).
        # So C[i,k] bilinear form = sum_c W_c[i,k] * outer(s_c, t_c).
        # We need: for each (i,k), the bilinear form sum_j A[i,j] B[j,k] = sum_c W_c[i,k] outer(s_c,t_c).
        # The bilinear form sum_j A[i,j]B[j,k] = sum_j outer(e_{i,j}, e_{j,k}).
        # So we need: sum_j outer(e_{i,j}, e_{j,k}) in span{outer(s_c,t_c)} with coefficients W_c[i,k].
        # So the 8 output bilinear forms must each be in the span of the r matrices outer(s_c,t_c).
        # Build basis: Bmat = stack of outer(s_c,t_c) -> (r,6,12). Flatten to (r,72).
        # For each output (i,k), target = sum_j outer(e_{i,j},e_{j,k}) flattened (72,).
        # Solve W[:,i,k] such that Bmat_flat @ W[:,i,k] = target. (least squares / exact)
        # Feasible iff target in column space of Bmat_flat.
        out_targets = []
        for i in range(n):
            for k in range(p):
                M = np.zeros((dimA, dimB))
                for j in range(m):
                    M[i*m+j, j*p+k] = 1.0
                out_targets.append(M.ravel())
        out_targets = np.array(out_targets)  # (8,72)
        for trial in range(n_trials):
            S = rng.standard_normal((r, dimA))
            T = rng.standard_normal((r, dimB))
            Bflat = np.einsum('ca,cb->cab', S, T).reshape(r, dimA*dimB)  # (r,72)
            # check column space contains out_targets
            # solve least squares, check residual
            try:
                # W is (r,8): Bflat @ W = out_targets.T  -> (r,72)@(r,8)? no.
                # out_targets (8,72) = W.T (8,r) @ Bflat (r,72). Solve W.T = lstsq(Bflat, out_targets.T)
                Wt, res, rank_b, sv = np.linalg.lstsq(Bflat, out_targets.T, rcond=None)
                # residual
                R = out_targets.T - Bflat @ Wt
                if np.max(np.abs(R)) < 1e-6:
                    # feasible! Wt is (r,8). Build U=S.T? 
                    # u_c = S[c] flattened to (n,m). v_c = T[c] to (m,p). w_c = Wt[c] to (n,p).
                    U = S.T  # (6, r)
                    V = T.T  # (12, r)
                    W = Wt.T  # (8, r)
                    return U, V, W
            except Exception:
                pass
            if time.time() > deadline:
                return None
        return None

    # search ranks below 24
    for r in range(1, best_rank):
        if time.time() > deadline:
            break
        res = try_rank(r, n_trials=300)
        if res is not None:
            U, V, W = res
            # verify
            best = (U, V, W); best_rank = r
            break

    U, V, W = best
    # convert to lists of floats (allow half-integers)
    def to_list(M):
        return [[float(x) for x in row] for row in M]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
