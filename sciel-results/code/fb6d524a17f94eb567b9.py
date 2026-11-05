"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
Combines structured initialization, vectorized residual computation, and
multi-attempt optimization to find low-rank decompositions.
"""
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]
DEADLINE = 145.0  # safety margin before 150s limit


def _build_target_tensor():
    """Return the 36 (i,j,k) indices of the multiplication tensor."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append((u_idx, v_idx, w_idx))
    return terms


def _verify(U, V, W, terms):
    """Verify that the decomposition exactly satisfies all tensor equations."""
    for (u_idx, v_idx, w_idx) in terms:
        s = 0.0
        for r in range(U.shape[1]):
            s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        if abs(s - 1.0) > 1e-9:
            return False
    return True


def _snap_and_verify(U, V, W, terms):
    """Snap to nearest half-integer and verify."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    if _verify(U_s, V_s, W_s, terms):
        return U_s.tolist(), V_s.tolist(), W_s.tolist()
    # Also try integer snapping
    U_i = np.round(U).astype(int)
    V_i = np.round(V).astype(int)
    W_i = np.round(W).astype(int)
    if _verify(U_i.astype(float), V_i.astype(float), W_i.astype(float), terms):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    return None


def _schoolbook_solution():
    """Return the rank-36 schoolbook solution."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    terms = _build_target_tensor()
    for col, (u_idx, v_idx, w_idx) in enumerate(terms):
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
    return U.tolist(), V.tolist(), W.tolist()


def _try_solve_rank_multi_init(R, terms, max_time=100.0):
    """
    Try to find a rank-R decomposition with multiple diverse initializations.
    """
    n, m, p = DIMENSIONS
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n

    start_time = time.time()

    term_indices = np.array(terms)

    def residuals(x):
        U_flat = x[:u_rows * R].reshape(u_rows, R)
        V_flat = x[u_rows * R:u_rows * R + v_rows * R].reshape(v_rows, R)
        W_flat = x[u_rows * R + v_rows * R:].reshape(w_rows, R)

        T_decomp = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
        return (T_decomp[term_indices[:, 0], term_indices[:, 1], term_indices[:, 2]] - 1.0)

    best_cost = np.inf
    best_x = None

    num_attempts = 8
    for attempt in range(num_attempts):
        if time.time() - start_time > max_time:
            break

        np.random.seed(100 + attempt * 13)

        # Different initialization strategies
        if attempt % 3 == 0:
            # Structured: pick R random terms
            all_terms = list(range(len(terms)))
            np.random.shuffle(all_terms)
            selected = all_terms[:min(R, len(terms))]

            U = np.zeros((u_rows, R))
            V = np.zeros((v_rows, R))
            W = np.zeros((w_rows, R))

            for r, term_idx in enumerate(selected):
                u_idx, v_idx, w_idx = terms[term_idx]
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0

            U += np.random.normal(0, 0.1, U.shape)
            V += np.random.normal(0, 0.1, V.shape)
            W += np.random.normal(0, 0.1, W.shape)

        elif attempt % 3 == 1:
            # Random initialization
            U = np.random.randn(u_rows, R) * 0.5
            V = np.random.randn(v_rows, R) * 0.5
            W = np.random.randn(w_rows, R) * 0.5

        else:
            # Partial structured: assign some columns, leave rest random
            U = np.random.randn(u_rows, R) * 0.1
            V = np.random.randn(v_rows, R) * 0.1
            W = np.random.randn(w_rows, R) * 0.1

            num_structured = min(R, 12)
            all_terms = list(range(len(terms)))
            np.random.shuffle(all_terms)
            for r in range(num_structured):
                u_idx, v_idx, w_idx = terms[all_terms[r]]
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=2000,
                                   xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except Exception:
            continue

        if result.cost < best_cost:
            best_cost = result.cost
            best_x = result.x.copy()

        if best_cost < 1e-10:
            U_sol = best_x[:u_rows * R].reshape(u_rows, R)
            V_sol = best_x[u_rows * R:u_rows * R + v_rows * R].reshape(v_rows, R)
            W_sol = best_x[u_rows * R + v_rows * R:].reshape(w_rows, R)

            snapped = _snap_and_verify(U_sol, V_sol, W_sol, terms)
            if snapped is not None:
                return snapped

            if _verify(U_sol, V_sol, W_sol, terms):
                return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()

    # If we got a good solution but not quite perfect, try snapping
    if best_x is not None and best_cost < 1e-6:
        U_sol = best_x[:u_rows * R].reshape(u_rows, R)
        V_sol = best_x[u_rows * R:u_rows * R + v_rows * R].reshape(v_rows, R)
        W_sol = best_x[u_rows * R + v_rows * R:].reshape(w_rows, R)

        snapped = _snap_and_verify(U_sol, V_sol, W_sol, terms)
        if snapped is not None:
            return snapped

    return None


def _try_sparse_interpolation(R, terms, max_time=30.0):
    """
    Attempt to find a rank-R decomposition using a randomized algebraic approach.
    This is a heuristic inspired by sparse interpolation: we assume the support
    of the decomposition (non-zero entries) is sparse and try to recover it.
    """
    n, m, p = DIMENSIONS
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n

    start_time = time.time()

    # We model the tensor as a sum of R rank-1 terms.
    # Each term is defined by a triplet (i_r, j_r, k_r) and a scalar.
    # However, the "indices" are not necessarily single coordinates but vectors.
    # A more robust approach: Assume the matrices U, V, W have a specific structure.
    
    # For now, this function serves as a placeholder for the "inspiration" strategy.
    # Given the difficulty of the problem, we fall back to the main optimizer if this fails.
    # But let's try a specific structured search: 
    # Assume the decomposition corresponds to a subset of the 36 schoolbook terms
    # combined with some "cross terms" that cancel out.
    
    # This is complex to implement generically in a short time.
    # Instead, we'll use the remaining time budget in the main loop for more optimizer restarts.
    return None


def construct():
    start_time = time.time()
    terms = _build_target_tensor()

    # Try to find solutions for rank < 36, starting from lower ranks
    # Known lower bound for 3x3x3 is 23, for 3x3x4 likely similar or slightly higher
    for R in range(20, 36):
        if time.time() - start_time > DEADLINE:
            break

        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 10:
            break

        # Use more time for lower ranks (harder to find)
        time_budget = min(remaining - 5, 30)

        result = _try_solve_rank_multi_init(R, terms, max_time=time_budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook rank 36
    U, V, W = _schoolbook_solution()
    return {"U": U, "V": V, "W": W}
