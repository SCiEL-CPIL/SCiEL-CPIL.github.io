"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def solve(a, b):
    n = len(a)
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s

    # Initial assignment: rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)

    # Local search: iterative improvement via best-improving swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        bp = b[p]
        # c[i,k] = sum_l a[i,l] * b[p[k],l]
        c = a @ bp.T
        # delta[i,j] for swapping facilities at positions i and j:
        # delta[i,j] = c[i,j] - c[i,i] - c[j,j] + c[j,i]
        diag = c.diagonal()
        delta_mat = c - diag[:, None] - diag[None, :] + c.T
        np.fill_diagonal(delta_mat, 0)
        # Only consider lower triangle (i > j) to avoid double counting
        delta_mat = np.tril(delta_mat, k=-1)

        min_idx = np.argmin(delta_mat)
        min_val = delta_mat.flat[min_idx]
        if min_val < 0:
            i_idx, j_idx = np.unravel_index(min_idx, delta_mat.shape)
            p[i_idx], p[j_idx] = p[j_idx], p[i_idx]
            new_cost = _qap_cost(a, b, p)
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = p.copy()
            improved = True

    return best_p.tolist()
# EVOLVE-BLOCK-END
