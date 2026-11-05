"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify that U, V, W give a valid tensor decomposition of matrix multiplication."""
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n*m, r) or Va.shape != (m*p, r) or Wa.shape != (p*n, r):
        return False
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = Ua[:, c].reshape(n, m)
        vc = Va[:, c].reshape(m, p)
        wc = Wa[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                if uc[i, j] != 0:
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _residuals(params, n, m, p, r):
    """Residual vector of the tensor equation for given parameters."""
    U = params[:n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r :].reshape(p*n, r)
    res = np.empty(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j], V[j*p+k]*W[k*n+i]) - 1.0
                idx += 1
    return res

def _jacobian(params, n, m, p, r):
    """Jacobian of the residuals (for better optimization)."""
    U = params[:n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r :].reshape(p*n, r)
    nres = n*m*p
    nparams = (n*m + m*p + p*n)*r
    J = np.zeros((nres, nparams))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    J[idx, i*m*r + j*r + c] = V[j*p+k, c] * W[k*n+i, c]
                for c in range(r):
                    J[idx, n*m*r + j*p*r + k*r + c] = U[i*m+j, c] * W[k*n+i, c]
                for c in range(r):
                    J[idx, n*m*r + m*p*r + k*n*r + i*r + c] = U[i*m+j, c] * V[j*p+k, c]
                idx += 1
    return J

def _round_to_half_int(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _verify_half_int(U, V, W, n, m, p, tol=1e-7):
    """Verify tensor equation and that entries are close to half-integers."""
    if not _verify(U, V, W, n, m, p, tol=tol):
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for arr in [Ua, Va, Wa]:
        if not np.allclose(arr * 2.0, np.round(arr * 2.0), atol=1e-6):
            return False
    return True

def _optimize_rank(n, m, p, r, deadline, n_restarts=15, seed_base=0):
    """
    Try to find a rank-r decomposition via nonlinear least squares with Jacobian.
    Returns (U, V, W) as lists or None.
    """
    from scipy.optimize import least_squares
    rng = np.random.RandomState(seed_base + r)
    best_cost = np.inf
    best_params = None

    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        if restart % 3 == 0:
            U0 = rng.randn(n*m, r)
            V0 = rng.randn(m*p, r)
            W0 = rng.randn(p*n, r)
        elif restart % 3 == 1:
            U0 = np.zeros((n*m, r))
            V0 = np.zeros((m*p, r))
            W0 = np.zeros((p*n, r))
            for c in range(r):
                ii = rng.randint(0, n*m)
                jj = rng.randint(0, m*p)
                kk = rng.randint(0, p*n)
                U0[ii, c] = rng.choice([-1, 1])
                V0[jj, c] = rng.choice([-1, 1])
                W0[kk, c] = rng.choice([-1, 1])
        else:
            U0 = rng.randn(n*m, r) / np.sqrt(n*m)
            V0 = rng.randn(m*p, r) / np.sqrt(m*p)
            W0 = rng.randn(p*n, r) / np.sqrt(p*n)
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        try:
            res = least_squares(_residuals, x0, args=(n, m, p, r),
                                jac=_jacobian, method='lm', max_nfev=3000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_params = res.x
            if best_cost < 1e-12:
                break
        except Exception:
            continue

    if best_params is None or best_cost > 1e-6:
        return None

    U = best_params[:n*m*r].reshape(n*m, r)
    V = best_params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = best_params[n*m*r + m*p*r :].reshape(p*n, r)
    
    # High-precision polish: run more iterations with tighter tolerance
    if time.time() < deadline - 15:
        try:
            res2 = least_squares(_residuals, best_params, args=(n, m, p, r),
                                 jac=_jacobian, method='lm', max_nfev=5000,
                                 xtol=1e-15, ftol=1e-15, gtol=1e-15)
            if res2.cost < best_cost:
                best_cost = res2.cost
                best_params = res2.x
                U = best_params[:n*m*r].reshape(n*m, r)
                V = best_params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = best_params[n*m*r + m*p*r :].reshape(p*n, r)
        except Exception:
            pass

    # Try rounding to half-integers
    U_rounded = _round_to_half_int(U)
    V_rounded = _round_to_half_int(V)
    W_rounded = _round_to_half_int(W)
    
    if _verify_half_int(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # If rounding fails, return the float solution (still valid for score)
    return U.tolist(), V.tolist(), W.tolist()

def _local_search_half_int(U, V, W, n, m, p, r, max_iter=500, deadline=None):
    """
    Local search on the half-integer lattice to improve residual.
    U, V, W are numpy arrays of half-integer values.
    """
    best_res = np.max(np.abs(_residuals(
        np.concatenate([U.flatten(), V.flatten(), W.flatten()]), n, m, p, r
    )))
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        if best_res < 1e-9:
            break
        improved = False
        # Try flipping each entry
        for c in range(r):
            for i in range(n*m):
                old = U_cur[i, c]
                U_cur[i, c] = old + 0.5
                res = np.max(np.abs(_residuals(
                    np.concatenate([U_cur.flatten(), V_cur.flatten(), W_cur.flatten()]), n, m, p, r
                )))
                if res < best_res:
                    best_res = res
                    improved = True
                else:
                    U_cur[i, c] = old
            
            for i in range(m*p):
                old = V_cur[i, c]
                V_cur[i, c] = old + 0.5
                res = np.max(np.abs(_residuals(
                    np.concatenate([U_cur.flatten(), V_cur.flatten(), W_cur.flatten()]), n, m, p, r
                )))
                if res < best_res:
                    best_res = res
                    improved = True
                else:
                    V_cur[i, c] = old
            
            for i in range(p*n):
                old = W_cur[i, c]
                W_cur[i, c] = old + 0.5
                res = np.max(np.abs(_residuals(
                    np.concatenate([U_cur.flatten(), V_cur.flatten(), W_cur.flatten()]), n, m, p, r
                )))
                if res < best_res:
                    best_res = res
                    improved = True
                else:
                    W_cur[i, c] = old
        
        if not improved:
            break
    
    return U_cur, V_cur, W_cur, best_res

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses schoolbook as baseline and searches for lower ranks via nonlinear optimization.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20-second safety margin before 150s

    # Schoolbook baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    if not _verify(U_sb, V_sb, W_sb, n, m, p):
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Lower bound on rank: max(n*m, m*p, p*n)
    lb = max(n*m, m*p, p*n)  # max(12, 20, 15) = 20

    # Search ranks from lower bound up to schoolbook - 1
    search_start = lb
    search_end = best_rank - 1

    n_ranks = max(1, search_end - search_start + 1)
    per_rank_time = (deadline - start_time - 20) / n_ranks  # reserve 20s for final polish

    for target_rank in range(search_start, search_end + 1):
        if time.time() > deadline - 20:
            break
        time_factor = 1.0 + (search_end - target_rank) * 0.05
        rank_deadline = min(deadline - 20, time.time() + max(3.0, per_rank_time * time_factor))
        result = _optimize_rank(n, m, p, target_rank, rank_deadline,
                                n_restarts=10, seed_base=target_rank * 101)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                # Try local search on half-integer lattice if we found a solution
                if time.time() < deadline - 25:
                    Ua = np.array(U, dtype=float)
                    Va = np.array(V, dtype=float)
                    Wa = np.array(W, dtype=float)
                    # Check if close to half-integers
                    if (np.allclose(Ua * 2, np.round(Ua * 2), atol=0.1) and
                        np.allclose(Va * 2, np.round(Va * 2), atol=0.1) and
                        np.allclose(Wa * 2, np.round(Wa * 2), atol=0.1)):
                        Ua = _round_to_half_int(Ua)
                        Va = _round_to_half_int(Va)
                        Wa = _round_to_half_int(Wa)
                        Ua, Va, Wa, res_val = _local_search_half_int(
                            Ua, Va, Wa, n, m, p, best_rank, 
                            max_iter=100, deadline=deadline - 25)
                        if res_val < 1e-8:
                            best_U, best_V, best_W = Ua.tolist(), Va.tolist(), Wa.tolist()
                continue

    # Final verification; fall back to schoolbook if anything is off
    if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-6):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
