"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    Vectorized for speed.
    """
    pi = p[i]
    pj = p[j]
    n = len(p)

    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False

    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]

    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]

    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)

    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))

    return int(delta_k + delta_self)


def _local_search(a, b, p, deadline, max_passes=100):
    """
    First-improvement local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    cost = _cost(a, b, p)

    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break  # First improvement per i, then move to next i
        if not improved:
            break

    return p, cost


def _build_surrogate(a, b, p):
    """
    Build a surrogate cost matrix C[i, l] approximating the marginal cost
    of assigning facility i to location l, holding other assignments fixed.
    C[i, l] = sum_j (a[i, j] + a[j, i]) * b[p[j], l]
    """
    a_sym = a + a.T
    Bp = b[np.ix_(p, p)]
    # L[i, k] = sum_j a_sym[i, j] * Bp[j, k]
    L = a_sym @ Bp
    return L


def _surrogate_perturb(a, b, p, rng):
    """
    Generate a new permutation using a linear assignment problem surrogate.
    """
    n = len(p)
    L = _build_surrogate(a, b, p)
    # Solve LAP
    row_ind, col_ind = linear_sum_assignment(L)
    # row_ind is facilities, col_ind is locations in the permuted space
    # Map col_ind back to actual location indices via p
    new_p = np.empty(n, dtype=np.int64)
    for i, k in zip(row_ind, col_ind):
        new_p[i] = p[k]
    return new_p


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization: align facility ordering with location ordering
    based on eigenvectors of the flow and distance matrices.
    """
    try:
        evals_a, evecs_a = np.linalg.eigh(a)
        evals_b, evecs_b = np.linalg.eigh(b)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    idx_a = np.argmax(evals_a)
    idx_b = np.argmax(evals_b)

    vec_a = evecs_a[:, idx_a]
    vec_b = evecs_b[:, idx_b]

    facility_order = np.argsort(-vec_a, kind="stable")
    location_order = np.argsort(-vec_b, kind="stable")

    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order

    if rng.random() < 0.5:
        location_order = location_order[::-1]
        p = np.empty(n, dtype=np.int64)
        p[facility_order] = location_order

    return p


def _best_improvement_pass(a, b, p, deadline):
    """
    One pass of best-improvement: find the best swap, apply it.
    Returns (p, cost, best_delta).
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_delta = 0
    best_i = -1
    best_j = -1

    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            delta = _delta_swap(a, b, p, i, j)
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j

    if best_i >= 0:
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta

    return p, cost, best_delta


def _lap_perturbation(a_sym, b, p):
    """
    Generate a new permutation using a linear assignment problem surrogate.
    a_sym is a + a.T.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    # L[i, k] = sum_j (a[i, j] + a[j, i]) * b[p[j], k]
    # This is (a_sym @ Bp)[i, k]
    L = a_sym @ Bp
    # Solve LAP
    row_ind, col_ind = linear_sum_assignment(L)
    # row_ind is facilities, col_ind is locations in the permuted space?
    # linear_sum_assignment returns row_idx, col_idx such that cost is minimized.
    # Here rows are facilities, cols are locations (indices 0..n-1 in Bp space).
    # We need to map col_ind (which are indices into Bp columns, i.e., locations in p)
    # back to actual location indices.
    # p is facility -> location.
    # Bp columns correspond to locations p[0], p[1], ..., p[n-1].
    # So if LAP says facility i goes to column k, the actual location is p[k].
    new_p = np.empty(n, dtype=np.int64)
    for i, k in zip(row_ind, col_ind):
        new_p[i] = p[k]
    return new_p


def _quantum_perturb(a, b, p, rng, num_swaps=3):
    """
    Perturb by performing a small number of random swaps.
    """
    n = len(p)
    perturbed_p = p.copy()
    for _ in range(num_swaps):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]
    return perturbed_p


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill

    # Precompute row sums of a for heuristic
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    a_sym = a + a.T

    # --- Phase 1: Multiple heuristic initializations with local search ---
    best_p = None
    best_cost = np.inf

    rng = np.random.default_rng(12345)

    # Heuristic 1: Rank-matching (heavy facilities to central locations)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    p1, c1 = _local_search(a, b, p1.copy(), deadline)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()

    # Heuristic 2: Reverse rank-matching
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    p2, c2 = _local_search(a, b, p2.copy(), deadline)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # Heuristic 3: Spectral initialization
    p3 = _spectral_init(a, b, n, rng)
    p3, c3 = _local_search(a, b, p3, deadline)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()

    # Heuristic 4: LAP-based initialization
    # Start with a random or spectral, then do one LAP perturbation step
    p4 = _spectral_init(a, b, n, rng)
    p4_lap = _lap_perturbation(a_sym, b, p4)
    p4_lap, c4 = _local_search(a, b, p4_lap, deadline)
    if c4 < best_cost:
        best_cost = c4
        best_p = p4_lap.copy()

    # Heuristic 5: Random
    p5 = rng.permutation(n).astype(np.int64)
    p5, c5 = _local_search(a, b, p5, deadline)
    if c5 < best_cost:
        best_cost = c5
        best_p = p5.copy()

    # --- Phase 2: Iterated Local Search (ILS) with LAP perturbations ---
    current_p = best_p.copy()
    current_cost = best_cost

    ilse_deadline = deadline - 5.0

    iteration = 0
    while time.time() < ilse_deadline:
        iteration += 1

        # Perturbation: Use LAP surrogate to guide the move
        # Occasionally use random swaps to escape if LAP gets stuck
        if iteration % 5 == 0:
            # Random perturbation
            k = 2 + (iteration % 4)
            perturbed_p = _quantum_perturb(a, b, current_p, rng, num_swaps=k)
        else:
            # LAP perturbation
            perturbed_p = _lap_perturbation(a_sym, b, current_p)
            # Add a small random jitter to avoid cycles
            if rng.random() < 0.3:
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

        # Local search from perturbed solution
        perturbed_p, perturbed_cost = _local_search(a, b, perturbed_p, ilse_deadline)

        # Accept if better
        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            if rng.random() < 0.1:
                current_p = perturbed_p
                current_cost = perturbed_cost

    # --- Phase 3: Final polish with best-improvement on the best solution ---
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        for _ in range(20):
            if time.time() > polish_deadline:
                break
            polish_p, polish_cost, bd = _best_improvement_pass(a, b, polish_p, polish_deadline)
            if bd == 0:
                break
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
