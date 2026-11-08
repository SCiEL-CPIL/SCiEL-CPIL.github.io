"""Search for a rank-minimal decomposition of the 2x3x10 matrix-mult tensor.

Combines:
  * the current program's structural sparse-search + solve-for-W machinery, and
  * the inspiration's "tensor slicing" / linear-algebraic elimination idea:
    each rank term is a rank-1 slice; we exploit the per-(i,k) Khatri-Rao
    structure to systematically eliminate redundant columns and certify a
    lower-rank solution, plus a fast full-tensor verification and an exact
    rational polish pass.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
TARGET_DEADLINE = 140.0  # safety margin under the 150s kill

VALS = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])


def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """Fast full-tensor verification.

    Returns (ok, max_err). Checks ALL tensor entries: the n*p*m non-zero
    entries must equal 1, and every other (a,b,c) triple must equal 0.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    # Reshape to 3D tensors: Ua[i,j,r], Va[j,k,r], Wa[k,i,r]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)

    # T[i,k,j] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    target = np.ones((n, p, m))
    err_nz = float(np.max(np.abs(T - target)))

    # Zero entries: build the full (nm, mp, pn) tensor and compare to support.
    full = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    support = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                support[i * m + j, j * p + k, k * n + i] = 1.0
    err_z = float(np.max(np.abs(full - support)))

    max_err = max(err_nz, err_z)
    return (max_err < tol), max_err


def _current_rank55(n, m, p):
    """Build the known rank-55 solution (seed)."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        U[0][col] += 1; U[4][col] += 1
        V[k0][col] += 1; V[10 + k1][col] += 1
        W[k0][col] += 1; W[k1 + 2][col] += 1
        col += 1
        U[3][col] += 1; U[4][col] += 1
        V[k0][col] += 1
        W[k0 + 2][col] += 1; W[k1 + 2][col] -= 1
        col += 1
        U[0][col] += 1
        V[k1][col] += 1; V[10 + k1][col] -= 1
        W[k1][col] += 1; W[k1 + 2][col] += 1
        col += 1
        U[4][col] += 1
        V[10 + k0][col] += 1; V[k0][col] -= 1
        W[k0][col] += 1; W[k0 + 2][col] += 1
        col += 1
        U[0][col] += 1; U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0][col] -= 1; W[k1][col] += 1
        col += 1
        U[3][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[k1][col] += 1
        W[k1 + 2][col] += 1
        col += 1
        U[1][col] += 1; U[4][col] -= 1
        V[10 + k0][col] += 1; V[10 + k1][col] += 1
        W[k0][col] += 1
        col += 1
        U[2][col] += 1; V[20 + k0][col] += 1; W[k0][col] += 1
        col += 1
        U[2][col] += 1; V[20 + k1][col] += 1; W[k1][col] += 1
        col += 1
        U[5][col] += 1; V[20 + k0][col] += 1; W[k0 + 2][col] += 1
        col += 1
        U[5][col] += 1; V[20 + k1][col] += 1; W[k1 + 2][col] += 1
        col += 1
    return {"U": U, "V": V, "W": W}


def _solve_W(U, V, n, m, p, R):
    """Given U and V, solve for W so the tensor identity holds.

    For each (i,k): M[j,r] = U[i*m+j,r]*V[j*p+k,r]; need M @ w_ik = 1.
    Returns list-of-lists W or None if infeasible.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    W = np.zeros((p * n, R))
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, R))
            for j in range(m):
                M[j] = Ua[i * m + j] * Va[j * p + k]
            try:
                w, residuals, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
            except Exception:
                return None
            if residuals.size > 0 and residuals[0] > 1e-7:
                return None
            W[k * n + i] = w
    return W.tolist()


def _rand_sparse(nrows, rank, density, rng):
    A = np.zeros((nrows, rank))
    for r_ in range(nrows):
        for c_ in range(rank):
            if rng.random() < density:
                A[r_, c_] = VALS[rng.randrange(len(VALS))]
    return A.tolist()


def _try_rank(R, n, m, p, rng, max_iters=2000, deadline=TARGET_DEADLINE):
    """Random sparse U,V + solve-for-W search for a valid rank-R solution."""
    nm, mp = n * m, m * p
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        U = _rand_sparse(nm, R, 0.20, rng)
        V = _rand_sparse(mp, R, 0.20, rng)
        W = _solve_W(U, V, n, m, p, R)
        if W is None:
            continue
        ok, _ = _verify_full(U, V, W, n, m, p)
        if ok:
            return U, V, W
    return None


def _try_rank_structured(R, n, m, p, rng, max_iters=400, deadline=TARGET_DEADLINE):
    """Structured search: use block patterns inspired by the rank-55 solution.
    
    The rank-55 solution has a periodic structure with period 10 in the k-index.
    We try variations of this structure with different parameters.
    """
    nm, mp = n * m, m * p
    # Try different density patterns and value sets
    densities = [0.15, 0.25, 0.30]
    val_sets = [
        np.array([-1.0, 0.5, 1.0]),
        np.array([-2.0, -1.0, 1.0, 2.0]),
        np.array([-1.0, 1.0]),
    ]
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        density = densities[rng.randrange(len(densities))]
        vals = val_sets[rng.randrange(len(val_sets))]
        
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        for r_ in range(nm):
            for c_ in range(R):
                if rng.random() < density:
                    U[r_, c_] = vals[rng.randrange(len(vals))]
        for r_ in range(mp):
            for c_ in range(R):
                if rng.random() < density:
                    V[r_, c_] = vals[rng.randrange(len(vals))]
        
        W = _solve_W(U.tolist(), V.tolist(), n, m, p, R)
        if W is None:
            continue
        ok, _ = _verify_full(U.tolist(), V.tolist(), W, n, m, p)
        if ok:
            return U.tolist(), V.tolist(), W
    return None


def _polish_rational(U, V, W, n, m, p, tol=1e-6):
    """Snap float entries to the nearest integer/half-integer if consistent.

    Returns (U,V,W) with clean entries, or None if snapping breaks validity.
    """
    def snap(M):
        a = np.array(M, dtype=float)
        b = np.round(a * 2.0) / 2.0
        return b.tolist()

    Us, Vs, Ws = snap(U), snap(V), snap(W)
    ok, _ = _verify_full(Us, Vs, Ws, n, m, p, tol=1e-6)
    if ok:
        return Us, Vs, Ws
    return None


def _solve_U(V, W, n, m, p, R):
    """Given V and W, solve for U so the tensor identity holds.

    For each (j,k): M[i,r] = V[j*p+k,r]*W[k*n+i,r]; need M @ u_j = 1 for all i.
    Actually: T[i,k,j] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    For fixed j,k: sum_r U[i*m+j,r] * (V[j*p+k,r]*W[k*n+i,r]) = 1 for all i.
    So for each (j,k), we have n equations in R unknowns (the row U[i*m+j] for i=0..n-1).
    Wait, U[i*m+j] is a single row of length R. The equation is:
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    This is n equations (one per i) in the R entries of each U[i*m+j,r].
    But U[i*m+j,r] appears only in the equation for that specific i.
    So actually each U[i*m+j,r] appears in exactly one equation (for that i).
    Hmm, let me reconsider. For fixed j,k and fixed i:
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    This is one equation in R unknowns (U[i*m+j,0..R-1]).
    With n equations (one per i) and each involving R unknowns, but the unknowns are disjoint per i.
    So we have n independent systems of 1 equation in R unknowns each - underdetermined!
    This means we can't uniquely solve for U given V,W. We need a different approach.
    
    Let me reconsider the structure. The tensor identity is:
    T[i,k,j] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta_{ikj} (1 if valid, 0 otherwise)
    
    For fixed (i,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j.
    This gives m equations in R unknowns per (i,k) pair, but the unknowns U[i*m+j,r] are shared across k for fixed i.
    
    Actually for fixed i: for each k, we have m equations:
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1, for j=0..m-1
    This is m equations in the R*m unknowns U[i*m+0..m-1, 0..R-1].
    With p values of k, we have m*p equations in R*m unknowns.
    If m*p > R*m (i.e., p > R), we might be able to solve.
    
    For n=2, m=3, p=10: we have 3*10=30 equations in R*3 unknowns per i.
    So if R < 10, we're overdetermined and can check consistency.
    If R >= 10, we're underdetermined.
    
    Let me use least squares to find a solution that minimizes error.
    """
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    U = np.zeros((n * m, R))
    for i in range(n):
        # Build the system: for each (j,k), equation: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # This is m*p equations in R*m unknowns (U[i*m+0..m-1, :])
        # Let's vectorize: unknowns are U[i*m: (i+1)*m, :] reshaped to (m*R,)
        # Equation for (j,k): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
        # This is a dot product of U[i*m+j,:] with (V[j*p+k,:] * W[k*n+i,:])
        
        # Build A matrix: (m*p, m*R) where row (j,k) has (V[j*p+k,:] * W[k*n+i,:]) in columns j*R..(j+1)*R
        A = np.zeros((m * p, m * R))
        b = np.ones(m * p)
        for j in range(m):
            for k in range(p):
                row_idx = j * p + k
                A[row_idx, j * R: (j + 1) * R] = Va[j * p + k] * Wa[k * n + i]
        try:
            sol, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        if residuals.size > 0 and residuals[0] > 1e-7:
            return None
        U[i * m: (i + 1) * m] = sol.reshape(m, R)
    return U.tolist()


def _solve_V(U, W, n, m, p, R):
    """Given U and W, solve for V so the tensor identity holds.

    For fixed (i,j): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k.
    This is p equations in R*p unknowns V[j*p+0..p-1, :].
    With n values of i, we have n*p equations in R*p unknowns.
    For n=2: 2*10=20 equations in R*10 unknowns.
    If R < 2, overdetermined. If R >= 2, underdetermined.
    """
    Ua = np.array(U, dtype=float)
    Wa = np.array(W, dtype=float)
    V = np.zeros((m * p, R))
    for j in range(m):
        # For each i, we have p equations: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # Unknowns: V[j*p+0..p-1, :] which is p*R values
        # Build A: (n*p, p*R)
        A = np.zeros((n * p, p * R))
        b = np.ones(n * p)
        for i in range(n):
            for k in range(p):
                row_idx = i * p + k
                A[row_idx, k * R: (k + 1) * R] = Ua[i * m + j] * Wa[k * n + i]
        try:
            sol, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        if residuals.size > 0 and residuals[0] > 1e-7:
            return None
        V[j * p: (j + 1) * p] = sol.reshape(p, R)
    return V.tolist()


def _eliminate_columns(U, V, W, n, m, p, rng, deadline=TARGET_DEADLINE):
    """Tensor-slicing elimination: repeatedly try to drop one rank column.

    Try multiple strategies for each column removal:
    1. Fix U,V; solve for W
    2. Fix U,W; solve for V  
    3. Fix V,W; solve for U
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    R = len(U[0])
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        # Try each column; order randomized for diversity
        order = list(range(R))
        rng.shuffle(order)
        for r in order:
            if time.time() > deadline - 5:
                break
            # Candidate: delete column r
            U2 = [row[:r] + row[r + 1:] for row in U]
            V2 = [row[:r] + row[r + 1:] for row in V]
            W2 = [row[:r] + row[r + 1:] for row in W]
            
            # Strategy 1: Fix U2,V2; solve for W
            Ws = _solve_W(U2, V2, n, m, p, R - 1)
            if Ws is not None:
                ok, _ = _verify_full(U2, V2, Ws, n, m, p)
                if ok:
                    U, V, W = U2, V2, Ws
                    R -= 1
                    improved = True
                    break
            
            # Strategy 2: Fix U2,W2; solve for V
            Vs = _solve_V(U2, W2, n, m, p, R - 1)
            if Vs is not None:
                ok, _ = _verify_full(U2, Vs, W2, n, m, p)
                if ok:
                    U, V, W = U2, Vs, W2
                    R -= 1
                    improved = True
                    break
            
            # Strategy 3: Fix V2,W2; solve for U
            Us = _solve_U(V2, W2, n, m, p, R - 1)
            if Us is not None:
                ok, _ = _verify_full(Us, V2, W2, n, m, p)
                if ok:
                    U, V, W = Us, V2, W2
                    R -= 1
                    improved = True
                    break
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rng = random.Random(12345)

    # Start with the known rank-55 solution.
    best = _current_rank55(n, m, p)
    best_rank = 55
    ok, err = _verify_full(best["U"], best["V"], best["W"], n, m, p)
    if not ok:
        # Fallback: trivial rank-60 solution.
        U = [[0] * 60 for _ in range(n * m)]
        V = [[0] * 60 for _ in range(m * p)]
        W = [[0] * 60 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = 60

    # 1) Systematically eliminate redundant columns from the seed.
    Ue, Ve, We = _eliminate_columns(best["U"], best["V"], best["W"], n, m, p, rng)
    ok, _ = _verify_full(Ue, Ve, We, n, m, p)
    if ok and len(Ue[0]) < best_rank:
        best = {"U": Ue, "V": Ve, "W": We}
        best_rank = len(Ue[0])

    # 2) Rational polish the seed (and any improved solution).
    Up, Vp, Wp = _polish_rational(best["U"], best["V"], best["W"], n, m, p)
    if Up is not None:
        best = {"U": Up, "V": Vp, "W": Wp}

    # 3) Search for lower ranks via sparse random + solve, then eliminate.
    for target_rank in range(best_rank - 1, 40, -1):
        if time.time() > TARGET_DEADLINE - 15:
            break
        # Try both random sparse and structured patterns
        result = _try_rank(target_rank, n, m, p, rng, max_iters=800)
        if result is None:
            result = _try_rank_structured(target_rank, n, m, p, rng, max_iters=400)
        if result is not None:
            U, V, W = result
            U, V, W = _eliminate_columns(U, V, W, n, m, p, rng)
            ok, _ = _verify_full(U, V, W, n, m, p)
            if ok:
                Ur, Vr, Wr = _polish_rational(U, V, W, n, m, p)
                if Ur is not None:
                    U, V, W = Ur, Vr, Wr
                best = {"U": U, "V": V, "W": W}
                best_rank = len(U[0])
                break

    # Deadline guard.
    if time.time() > TARGET_DEADLINE - 3:
        return best
    return best


if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    ok, err = _verify_full(res["U"], res["V"], res["W"], n, m, p)
    print(f"Rank: {len(res['U'][0])}, Valid: {ok}, MaxErr: {err:.2e}")
