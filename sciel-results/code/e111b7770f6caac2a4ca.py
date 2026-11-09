import numpy as np
import time

DIMENSIONS = [3, 3, 5]


def _build_target(n, m, p):
    """Return the 45 active (a,b,c) indices of the matmul tensor (all values 1)."""
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx.append((a, b, c))
    return np.array(idx, dtype=int)


def _schoolbook(n, m, p):
    """Sparse schoolbook factors: each row/col has exactly one 1.0."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    R = n * m * p
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    return U, V, W


def _active_matrix(U, V, W, idx):
    """Build the 45xR matrix M[e,r] = U[a_e,r]*V[b_e,r]*W[c_e,r]."""
    a = idx[:, 0]
    b = idx[:, 1]
    c = idx[:, 2]
    return U[a, :] * V[b, :] * W[c, :]


def _solve_fixed_support(M, R):
    """
    Solve for the 3R non-zero values (u,v,w) given a fixed sparsity support
    (which entries are non-zero per column). M is 45 x R with the *pattern*
    of ones at the support positions. Returns (u,v,w) arrays of length R or None.
    """
    # M already has 1.0 at support, 0 elsewhere. We need M @ (u*v*w) = 1.
    # This is nonlinear (product of three). Use a log/linearization-free least squares
    # via iterative refinement: alternate solving for one factor while others fixed.
    u = np.ones(R)
    v = np.ones(R)
    w = np.ones(R)
    target = np.ones(M.shape[0])
    for _ in range(200):
        # update u: for each row e, sum_r M[e,r]*u_r*(v_r*w_r) = 1
        vw = v * w
        # linear system in u: (M * vw) @ u = 1  ->  but M*vw is 45 x R, u is R
        A_u = M * vw
        # least squares: A_u @ u = target
        try:
            u_new, *_ = np.linalg.lstsq(A_u, target, rcond=None)
        except np.linalg.LinAlgError:
            break
        # update v
        uw = u * w
        A_v = M * uw
        try:
            v_new, *_ = np.linalg.lstsq(A_v, target, rcond=None)
        except np.linalg.LinAlgError:
            break
        # update w
        uv = u * v
        A_w = M * uv
        try:
            w_new, *_ = np.linalg.lstsq(A_w, target, rcond=None)
        except np.linalg.LinAlgError:
            break
        if (np.max(np.abs(u_new - u)) + np.max(np.abs(v_new - v)) +
                np.max(np.abs(w_new - w))) < 1e-13:
            u, v, w = u_new, v_new, w_new
            break
        u, v, w = u_new, v_new, w_new
    # verify
    recon = M @ (u * v * w)
    if np.max(np.abs(recon - target)) > 1e-8:
        return None
    return u, v, w


def _snap(x, mult):
    return np.round(x * mult) / mult


def _try_rank_sparse(n, m, p, R, idx, U_sb, V_sb, W_sb, deadline, seed_base=0):
    """
    Attempt a rank-R decomposition preserving the schoolbook sparsity pattern.
    We pick R of the 45 schoolbook columns (support) and solve for the values.
    Returns (U,V,W) or None.
    """
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    sb_R = n * m * p
    all_cols = np.arange(sb_R)

    n_restarts = 300
    for restart in range(n_restarts):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(seed_base + restart * 131)

        # Choose R columns to keep. Mix strategies.
        if restart % 5 == 0:
            # keep first R (deterministic-ish)
            keep = np.sort(all_cols[:R])
        elif restart % 5 == 1:
            # random subset
            keep = np.sort(rng.choice(all_cols, R, replace=False))
        elif restart % 5 == 2:
            # keep a structured block: first R//n*... try row-major contiguous
            keep = np.sort(all_cols[:R])
            # perturb a few
            if R < sb_R:
                for _ in range(min(5, sb_R - R)):
                    out = rng.choice(all_cols[:R])
                    new = rng.choice(all_cols[R:])
                    keep = np.sort(np.concatenate([keep[:list(keep).index(out)],
                                                   keep[list(keep).index(out)+1:],
                                                   [new]]))
        else:
            # random with bias toward low indices
            weights = 1.0 / (all_cols + 1)
            weights /= weights.sum()
            keep = np.sort(rng.choice(all_cols, R, replace=False, p=weights))

        # Build M: 45 x R with 1.0 where the support column matches the active entry
        # For each active entry e=(a,b,c) and each kept column r, the product is non-zero
        # only if the column's (a',b',c') equals (a,b,c). Since schoolbook columns are
        # unique per (i,j,k), column r corresponds to a specific (i,j,k).
        # M[e, r] = 1 if kept_col_r's (a,b,c) == idx[e], else 0.
        # Precompute each column's (a,b,c):
        col_abcs = []
        for cc in keep:
            # decode column index
            i = cc // (m * p)
            j = (cc // p) % m
            k = cc % p
            a = i * m + j
            b = j * p + k
            c = k * n + i
            col_abcs.append((a, b, c))
        col_abcs = np.array(col_abcs)
        # M[e, r] = 1 if idx[e] == col_abcs[r]
        M = np.zeros((len(idx), R))
        for e in range(len(idx)):
            match = np.where((col_abcs[:, 0] == idx[e, 0]) &
                             (col_abcs[:, 1] == idx[e, 1]) &
                             (col_abcs[:, 2] == idx[e, 2]))[0]
            if len(match) > 0:
                M[e, match] = 1.0

        sol = _solve_fixed_support(M, R)
        if sol is None:
            continue
        u, v, w = sol
        # Build sparse factors
        U = np.zeros((dim_u, R))
        V = np.zeros((dim_v, R))
        W = np.zeros((dim_w, R))
        for r in range(R):
            cc = keep[r]
            i = cc // (m * p)
            j = (cc // p) % m
            k = cc % p
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a, r] = u[r]
            V[b, r] = v[r]
            W[c, r] = w[r]
        # verify on active entries
        recon = _active_matrix(U, V, W, idx)
        if np.max(np.abs(recon - 1.0)) > 1e-8:
            continue
        # snap
        for mult in [8, 4, 2]:
            Us, Vs, Ws = _snap(U, mult), _snap(V, mult), _snap(W, mult)
            if np.max(np.abs(_active_matrix(Us, Vs, Ws, idx) - 1.0)) < 1e-9:
                return Us, Vs, Ws
        return U, V, W
    return None


def _greedy_remove(n, m, p, idx, U, V, W, deadline):
    """
    Starting from a valid rank-R decomposition, try to remove one column at a time
    (re-solve values with the remaining support) to lower the rank.
    Returns (U,V,W) with minimal rank found.
    """
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    R = U.shape[1]
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    # determine current support (non-zero column indices per factor row)
    improved = True
    while improved and R > 1:
        improved = False
        for try_col in range(R):
            if time.time() > deadline:
                break
            # try removing column try_col
            keep = [c for c in range(R) if c != try_col]
            if len(keep) < 1:
                continue
            # rebuild support from remaining columns
            # We need the (a,b,c) of each remaining column
            col_abcs = []
            for cc in keep:
                a_nz = np.nonzero(cur_U[:, cc])[0]
                b_nz = np.nonzero(cur_V[:, cc])[0]
                c_nz = np.nonzero(cur_W[:, cc])[0]
                if len(a_nz) != 1 or len(b_nz) != 1 or len(c_nz) != 1:
                    continue
                col_abcs.append((a_nz[0], b_nz[0], c_nz[0]))
            if len(col_abcs) != len(keep):
                continue
            col_abcs = np.array(col_abcs)
            M = np.zeros((len(idx), len(keep)))
            for e in range(len(idx)):
                match = np.where((col_abcs[:, 0] == idx[e, 0]) &
                                 (col_abcs[:, 1] == idx[e, 1]) &
                                 (col_abcs[:, 2] == idx[e, 2]))[0]
                if len(match) > 0:
                    M[e, match] = 1.0
            sol = _solve_fixed_support(M, len(keep))
            if sol is None:
                continue
            u, v, w = sol
            new_U = np.zeros((dim_u, len(keep)))
            new_V = np.zeros((dim_v, len(keep)))
            new_W = np.zeros((dim_w, len(keep)))
            for r, cc in enumerate(keep):
                new_U[col_abcs[r, 0], r] = u[r]
                new_V[col_abcs[r, 1], r] = v[r]
                new_W[col_abcs[r, 2], r] = w[r]
            if np.max(np.abs(_active_matrix(new_U, new_V, new_W, idx) - 1.0)) > 1e-8:
                continue
            # success
            cur_U, cur_V, cur_W = new_U, new_V, new_W
            R = len(keep)
            improved = True
            break
    return cur_U, cur_V, cur_W


def construct():
    n, m, p = DIMENSIONS
    idx = _build_target(n, m, p)
    sb_R = n * m * p

    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = sb_R

    # Phase 1: try ranks from sb_R-1 down, using sparse-support solver
    for R in range(sb_R - 1, max(15, sb_R - 20), -1):
        if time.time() > deadline - 30:
            break
        res = _try_rank_sparse(n, m, p, R, idx, U_sb, V_sb, W_sb, deadline, seed_base=R * 7)
        if res is not None:
            U_sol, V_sol, W_sol = res
            # try to improve further via greedy removal
            U_g, V_g, W_g = _greedy_remove(n, m, p, idx, U_sol, V_sol, W_sol, deadline)
            if U_g.shape[1] <= best_rank:
                best_rank = U_g.shape[1]
                best_U, best_V, best_W = U_g, V_g, W_g
            # stop early if we hit a good rank
            if best_rank <= 30:
                break

    # Final snap of best
    for mult in [16, 8, 4, 2]:
        Us, Vs, Ws = _snap(best_U, mult), _snap(best_V, mult), _snap(best_W, mult)
        if np.max(np.abs(_active_matrix(Us, Vs, Ws, idx) - 1.0)) < 1e-9:
            best_U, best_V, best_W = Us, Vs, Ws
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
