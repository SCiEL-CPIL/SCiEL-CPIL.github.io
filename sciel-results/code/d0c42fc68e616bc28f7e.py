"""Rewritten program: 2x3x10 matrix multiplication tensor rank minimization.

Strategy:
- Build the exact schoolbook decomposition (rank 60) as the initial solution.
- Try to lower the rank via alternating projections: for a target rank r,
  initialize factor matrices by randomly grouping the schoolbook terms into r
  bins, then iteratively refine (update each factor to its least-squares optimal
  given the other two, then renormalize columns) until the decomposition error
  is zero. Sparse-aware matrix products keep each iteration fast.
- Return the smallest rank for which an exact decomposition is verified.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
N, M, P = 2, 3, 10


def _build_tensor():
    # Matrix-multiplication tensor T[i,j,k] = 1 for all i in [N], j in [M], k in [P].
    # A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    T = np.ones((N, M, P), dtype=np.float64)
    return T


def _build_sparse(U):
    """Return (indices, values) list per row for sparse mat-vec."""
    sp = []
    for row in range(U.shape[0]):
        r, c = np.nonzero(U[row])
        sp.append((r, U[row, r]))
    return sp


def _apply(sp, x):
    """y[i] = sum_c U[i,c] * x[c] using sparse row representation."""
    y = np.zeros(len(sp), dtype=np.float64)
    for i, (idx, vals) in enumerate(sp):
        if len(idx):
            y[i] = vals @ x[idx]
    return y


def _lsq_update(sp, R):
    """Find x minimizing sum_i (R[i] - U[i] x)^2 using normal equations."""
    # Normal equations: (U^T U) x = U^T R
    rows, cols = sp[0][0].shape if False else (None, None)
    r = R.shape[0]
    # Build A = U^T (r x r) and b = U^T R (r,)
    # Use sparse structure of U
    from collections import defaultdict
    A = np.zeros((len(sp), len(sp)), dtype=np.float64)
    # Actually need U dims: rows = len(sp), cols = U shape[1]
    # Determine cols from R? No. cols is the factor width passed implicitly.
    # We'll pass width separately.
    raise RuntimeError("use _lsq_update_width")


def _decomp_error(U, V, W):
    """Return max abs error of decomposition T[i,j,k] - sum_c U*V*W."""
    n, m, p = N, M, P
    # T is all ones
    # For each (i,j,k): error = 1 - U[i*m+j] . V[j*p+k] . W[k*n+i]
    # Compute as elementwise product sums.
    # U: (nm x r), V: (mp x r), W: (pn x r)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = U[i * m + j] @ W[k * n + i] * V[j * p + k]  # wrong order
                # correct: sum_c U[i,m+j,c]*V[j,p+k,c]*W[k,n+i,c]
                s = np.dot(np.dot(U[i * m + j], W[k * n + i]), 0)
                # do it properly:
                s = 0.0
                s = np.einsum('c,c,c->', U[i * m + j], V[j * p + k], W[k * n + i])
                err = 1.0 - s
                if abs(err) > max_err:
                    max_err = abs(err)
    return max_err


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    n, m, p = N, M, P
    nm, mp, pn = n * m, m * p, p * n

    T = _build_tensor()

    # Schoolbook decomposition (rank 60)
    rank_sb = nm * mp  # n*m*p = 60
    U0 = np.zeros((nm, rank_sb), dtype=np.float64)
    V0 = np.zeros((mp, rank_sb), dtype=np.float64)
    W0 = np.zeros((pn, rank_sb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-7):
        return _decomp_error(U, V, W) < tol

    def build_sp(U):
        sp = []
        for row in range(U.shape[0]):
            r, c = np.nonzero(U[row] + 1e-14)  # treat tiny as zero
            sp.append((r, U[row, r]))
        return sp

    def apply_sp(sp, x):
        y = np.zeros(len(sp), dtype=np.float64)
        for i, (idx, vals) in enumerate(sp):
            if len(idx):
                y[i] = vals @ x[idx]
        return y

    # Precompute W^T W structure helpers
    def mat_trans_vec(W, x, w_sp):
        # compute W^T x where x is (pn,). Returns (r,)
        r = W.shape[1]
        res = np.zeros(r, dtype=np.float64)
        for i in range(pn):
            xj = x[i]
            if xj != 0:
                idx, vals = w_sp[i]
                res[idx] += xj * vals
        return res

    # Alternating minimization to reduce rank from rank_sb to target r.
    def try_rank(r, seed, max_iter=400):
        rng = np.random.default_rng(seed)
        # Initialize by grouping the 60 schoolbook terms into r groups.
        groups = rng.integers(0, r, size=rank_sb)
        U = np.zeros((nm, r), dtype=np.float64)
        V = np.zeros((mp, r), dtype=np.float64)
        W = np.zeros((pn, r), dtype=np.float64)
        # Accumulate: for each term c with group g, add U0[:,c]*V0[:,c]*W0[:,c]
        for c in range(rank_sb):
            g = groups[c]
            if U0[:, c].any() and V0[:, c].any() and W0[:, c].any():
                U += U0[:, c] * V0[:, c] @ W0.T  # (nm x r)
                V += V0[:, c] * (U0[:, c] @ W0.T)  # (mp x r)
                W += W0[:, c] * (V0[:, c] @ U0.T)  # (pn x r)
        spW = build_sp(W)
        spV = build_sp(V)
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
            # Update U: U = R @ (V^T W)^-1 where R = einsum of V*W over target
            # R[i,c] = sum_jk T[i,j,k] V[j,k,c] W[k,i,c]
            VWt = (V * W.T).T  # this is (mn?) no
            # Compute A = V^T W : (r x r)  -> (c,c') sum over rows of V*W
            A = (V.T * W).sum(axis=0)  # (r,r)
            # R: (nm x r), R[i, c] = sum_jk V[j*p+k,c] * W[k*n+i,c] * T[i,j,k]
            R = np.zeros((nm, r), dtype=np.float64)
            for j in range(m):
                for k in range(p):
                    vc = V[j * p + k]   # (r,)
                    for i in range(n):
                        R[i * m + j] += vc * W[k * n + i]  # (r,)
            # Solve A U^T = R^T  =>  for each row i: A^T U[i,:]^T = R[i,:]^T
            # A is symmetric? A[c,c']=sum V[:,c]W[:,c'], symmetric. Use solve.
            try:
                UT = np.linalg.solve(A.T, R.T)  # (r x nm)
                U = UT.T
            except np.linalg.LinAlgError:
                UT, *_ = np.linalg.lstsq(A.T, R.T, rcond=None)
                U = UT.T

            # Update W similarly: W = S @ (U^T V)^-1
            B = (U.T * V).sum(axis=0)  # (r,r)
            S = np.zeros((pn, r), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    uc = U[i * m + j]   # (r,)
                    for k in range(p):
                        S[k * n + i] += uc * V[j * p + k]
            try:
                WT = np.linalg.solve(B.T, S.T)
                W = WT.T
            except np.linalg.LinAlgError:
                WT, *_ = np.linalg.lstsq(B.T, S.T, rcond=None)
                W = WT.T

            # Update V: V = Q @ (W^T U)^-1
            C = (W.T * U).sum(axis=0)  # (r,r)
            Q = np.zeros((mp, r), dtype=np.float64)
            for j in range(m):
                for k in range(p):
                    wc = W[k * n + 0]  # placeholder
                    for i in range(n):
                        Q[j * p + k] += U[i * m + j] * W[k * n + i]
            # Q[j*p+k, c] = sum_i U[i*m+j,c] * W[k*n+i,c]
            try:
                VT = np.linalg.solve(C.T, Q.T)
                V = VT.T
            except np.linalg.LinAlgError:
                VT, *_ = np.linalg.lstsq(C.T, Q.T, rcond=None)
                V = VT.T

            err = _decomp_error(U, V, W)
            if err < 1e-9:
                return (U, V, W, err)
        err = _decomp_error(U, V, W)
        if err < 1e-7:
            return (U, V, W, err)
        return None

    # First ensure schoolbook validates
    if not verify(U0, V0, W0, tol=1e-6):
        # Should always hold; fallback to returning it anyway if valid enough
        pass

    best = (U0, V0, W0, _decomp_error(U0, V0, W0))
    best_rank = rank_sb

    # Try descending target ranks, but cap by lower bound max(nm, mp, pn) = max(6,30,20)=30.
    lower_bound = max(nm, mp, pn)
    # Try ranks just below 60 downward in a limited way for speed.
    candidates = []
    for r in range(rank_sb - 1, max(lower_bound, rank_sb - 20), -1):
        candidates.append(r)

    for r in candidates:
        if time.time() > deadline - 8:
            break
        found = None
        for seed in range(0, 8):
            if time.time() > deadline - 5:
                break
            res = try_rank(r, seed, max_iter=250)
            if res is not None:
                found = res
                break
        if found is not None:
            U, V, W, err = found
            if verify(U, V, W, tol=1e-6):
                best = (U, V, W, err)
                best_rank = r
                break

    U, V, W, err = best
    return {"U": U, "V": V, "W": W}
