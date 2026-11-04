"""Winograd-style search for a low-rank exact decomposition of the 2x3 * 3x7 tensor."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 7]

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin under the 150s kill limit


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _DEADLINE - time.time()


def _schoolbook():
    """Rank n*m*p = 42 exact decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_c U[:,c] outer V[:,c] outer W[:,c] == T[i,j,k]=E[i,k]."""
    # Build tensor from factors
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]  # n*m
        vc = V[:, c]  # m*p
        wc = W[:, c]  # p*n
        # reshape
        u2 = uc.reshape(n, m)
        v2 = vc.reshape(m, p)
        w2 = wc.reshape(p, n)
        # T[i,j,k] += u2[i,j] * v2[j,k] * w2[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u2[i, j] * v2[j, k] * w2[k, i]
    # target
    target = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            target[i, :, k] = 1.0
    return np.allclose(T, target, atol=tol, rtol=0.0)


def _build_from_points(points):
    """
    Winograd/Toom-Cook style construction.
    points: list of (x_i, y_k) evaluation points.
    We model the middle index j as a polynomial variable evaluated at points.
    For an n x p output, each entry (i,k) = sum_j A[i,j] B[j,k].
    We treat row i of A as a polynomial in t, column k of B as a polynomial in t,
    and use evaluation points to reduce multiplications.

    This is a heuristic: we build U, V, W from the evaluation points and verify.
    """
    n, m, p = DIMENSIONS
    # Number of evaluation points
    r = len(points)
    if r < 1:
        return None
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for c, (x, y) in enumerate(points):
        # U[i*m+j, c] = x^j  (evaluation of row polynomial at x)
        for i in range(n):
            for j in range(m):
                U[i * m + j, c] = x ** j
        # V[j*p+k, c] = y^j  (evaluation of column polynomial at y)
        for j in range(m):
            for k in range(p):
                V[j * p + k, c] = y ** j
        # W[k*n+i, c] = 1 (the output coefficient)
        for k in range(p):
            for i in range(n):
                W[k * n + i, c] = 1.0
    return {"U": U, "V": V, "W": W}


def _search_winograd():
    """
    Search over small sets of evaluation points (Winograd/Toom-Cook).
    Try candidate point sets of size r and verify.
    """
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p  # schoolbook rank

    # Candidate points: small integers and simple fractions
    candidate_x = [0, 1, 2, -1, 3, -2, 1.5, 0.5, -0.5]
    candidate_y = [0, 1, 2, -1, 3, -2, 1.5, 0.5, -0.5]

    # Try rank 1, 2, 3, ... up to a limit
    max_rank_to_try = 20  # don't exceed this in search

    for r in range(1, min(max_rank_to_try, best_rank) + 1):
        if _remaining() < 5.0:
            break
        # Try combinations of points
        # For efficiency, try a subset of combinations
        x_combos = itertools.combinations_with_replacement(candidate_x, r)
        y_combos = itertools.combinations_with_replacement(candidate_y, r)

        # Limit the number of trials
        max_trials = 5000
        trials = 0
        for x_set in x_combos:
            if trials >= max_trials or _remaining() < 5.0:
                break
            for y_set in y_combos:
                if trials >= max_trials or _remaining() < 5.0:
                    break
                trials += 1
                points = list(zip(x_set, y_set))
                result = _build_from_points(points)
                if result is None:
                    continue
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    # Found a valid decomposition of rank r
                    if r < best_rank:
                        best_rank = r
                        best = result
                    return best  # return immediately on first valid
        # If we didn't find one at this rank, continue to next
    return best


def _search_structured():
    """
    Alternative structured search: assume U and V have a Kronecker-like structure
    from evaluation points, and solve for W.
    """
    n, m, p = DIMENSIONS
    # For a given set of evaluation points (x_c, y_c), we have:
    # U[i*m+j, c] = x_c^j
    # V[j*p+k, c] = y_c^j
    # We need W[k*n+i, c] such that sum_c x_c^j y_c^j W[k*n+i, c] = 1 for all i,k
    # This is a linear system in W for each (i,k).
    # Let's try to solve it.

    candidate_x = [0, 1, 2, -1, 3, -2]
    candidate_y = [0, 1, 2, -1, 3, -2]

    best = None
    best_rank = n * m * p

    for r in range(1, min(15, best_rank) + 1):
        if _remaining() < 5.0:
            break
        # Try a few point sets
        for x_set in itertools.combinations_with_replacement(candidate_x, r):
            if _remaining() < 5.0:
                break
            for y_set in itertools.combinations_with_replacement(candidate_y, r):
                if _remaining() < 5.0:
                    break
                points = list(zip(x_set, y_set))
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                for c, (x, y) in enumerate(points):
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j, c] = x ** j
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, c] = y ** j
                # For each (i,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # This is: sum_c (sum_j x_c^j y_c^j) * W[k*n+i,c] = 1
                # Let M[i,k,c] = sum_j x_c^j y_c^j = sum_j (x_c*y_c)^j
                # Then sum_c M[i,k,c] * W[k*n+i,c] = 1
                # But M[i,k,c] doesn't depend on i or k! It's just sum_j (x_c*y_c)^j
                # So the system is: sum_c s_c * W[k*n+i,c] = 1 for all i,k
                # where s_c = sum_j (x_c*y_c)^j
                # This means W[k*n+i,c] must be the same for all i,k for each c?
                # No: sum_c s_c * W_c = 1, where W_c = W[k*n+i,c]
                # If this holds for all (i,k), then W[k*n+i,c] must be independent of (i,k)
                # OR the s_c vector must be such that the only solution is W independent of (i,k)
                # Let's just solve the linear system for W for one (i,k) and check consistency
                s = np.zeros(r)
                for c, (x, y) in enumerate(points):
                    s[c] = sum((x * y) ** j for j in range(m))
                # We need W_vec such that s @ W_vec = 1
                # This is underdetermined if r > 1. We can pick a particular solution.
                # But W must be the same for all (i,k) for this to work.
                # Let's check if s @ W = 1 has a solution and set W[k*n+i,c] = W_c for all i,k
                try:
                    # Find W_c such that sum_c s_c * W_c = 1
                    # Use least squares
                    W_c, residuals, rank_sys, sv = np.linalg.lstsq(s.reshape(-1, 1), np.ones(1), rcond=None)
                    W_c = W_c.flatten()
                    # Check residual
                    if np.abs(s @ W_c - 1.0) < 1e-9:
                        W = np.zeros((p * n, r))
                        for k in range(p):
                            for i in range(n):
                                for c in range(r):
                                    W[k * n + i, c] = W_c[c]
                        result = {"U": U, "V": V, "W": W}
                        if _verify(result["U"], result["V"], result["W"], n, m, p):
                            if r < best_rank:
                                best_rank = r
                                best = result
                            return best
                except Exception:
                    pass
    return best


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as numpy arrays.
    """
    # Always have the schoolbook as fallback
    fallback = _schoolbook()

    # Try Winograd-style search
    best = None
    try:
        if _remaining() > 10.0:
            best = _search_winograd()
    except Exception:
        pass

    # Try structured search
    try:
        if _remaining() > 10.0:
            best2 = _search_structured()
            if best2 is not None:
                if best is None or best2["U"].shape[1] < best["U"].shape[1]:
                    best = best2
    except Exception:
        pass

    # Return the best found, or fallback
    if best is not None:
        # Ensure integer/half-integer entries (round to nearest 0.5)
        for key in ["U", "V", "W"]:
            best[key] = np.round(best[key] * 2.0) / 2.0
        # Final verification
        if _verify(best["U"], best["V"], best["W"], DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2]):
            return best
    return fallback
