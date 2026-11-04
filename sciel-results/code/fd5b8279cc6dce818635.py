"""
Symplectic dynamics + steepest-descent QAP solver for QAPLIB wil100.
Generates diverse initial permutations via a deterministic Hamiltonian-like
trajectory on the assignment simplex (relaxation), then refines with a
fast vectorized steepest-descent swap search and a final SA polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _hungarian_initial(a, b):
    """Hungarian on a row-sum outer-product linearization."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    cost = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _greedy_initial(a, b):
    """Greedy: heavy flows to central locations."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _random_initial(n, rng=None):
    if rng is None:
        return np.random.permutation(n).astype(np.int64)
    return rng.permutation(n).astype(np.int64)


def _compute_delta_matrix(a, b, p):
    """
    Full n x n matrix of swap deltas D[i,j]: cost change swapping p[i] and p[j].
    Vectorized O(n^3) via matrix products.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]

    M1 = a.T @ Bp
    M2 = a @ Bp

    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_Bp = np.diag(Bp)

    T1_raw = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    T2_raw = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T

    a_diag = np.diag(a)

    C1_part1 = (a_diag[:, None] - a) * (Bp - diag_Bp[:, None])
    C1_part2 = (a.T - a_diag[None, :]) * (diag_Bp[None, :] - Bp)
    C1 = C1_part1 + C1_part2

    C2_part1 = (a_diag[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C2_part2 = (a - a_diag[None, :]) * (diag_Bp[None, :] - Bp.T)
    C2 = C2_part1 + C2_part2

    T3 = (a - a.T) * (Bp.T - Bp)

    D = (T1_raw - C1) + (T2_raw - C2) + T3
    np.fill_diagonal(D, 0)
    return D


def _local_search_fast(a, b, p, time_limit=None, start_time=None):
    """Steepest-descent swap local search using vectorized delta matrix."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0

    current_cost = _qap_cost(a, b, p)

    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        best_delta = np.min(D)
        if best_delta >= 0:
            break
        best_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)

    return p, current_cost


def _compute_swap_delta(a, b, p, i, j):
    """Exact integer delta for swapping positions i and j."""
    pi = p[i]
    pj = p[j]

    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)

    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j

    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)

    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j

    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])

    return int(T1 + T2 + T3)


def _symplectic_trajectory_init(a, b, n, num_samples=8, steps=600, lr=0.02, seed=42):
    """
    Generate diverse initial permutations via a deterministic symplectic-like
    trajectory on a relaxed assignment matrix. The potential energy is the
    quadratic cost of the relaxed problem. We use a velocity Verlet-style
    update with momentum to trace a trajectory through the solution space,
    sampling permutations at several time points.
    """
    rng = np.random.RandomState(seed)

    # Build the relaxed cost gradient structure
    # For assignment matrix X (n x n, doubly stochastic),
    # cost = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    # Gradient w.r.t. X[i,k]: sum_{j,l} a[i,j] * b[k,l] * X[j,l] * 2 (symmetry)
    # But for the assignment problem, we can use the linearized gradient:
    # g[i,k] = sum_j a[i,j] * (sum_l b[k,l] * X[j,l]) + sum_l b[k,l] * (sum_i' a[i',k] * X[i',l])
    # Simplified: use row/col sums as in Hungarian but updated dynamically.

    # Precompute
    a_sym = (a + a.T) * 0.5
    b_sym = (b + b.T) * 0.5

    # Initialize X as a doubly stochastic matrix (uniform)
    X = np.ones((n, n)) / n
    V = np.zeros((n, n))  # velocity

    permutations = []

    for step in range(steps):
        # Compute gradient of relaxed QAP cost
        # g[i,k] = 2 * sum_j a_sym[i,j] * (b_sym[k] @ X[j]) + 2 * sum_l b_sym[k,l] * (a_sym[:, k] @ X[l])
        # = 2 * (a_sym @ (b_sym @ X.T)).T[k,i] ... let me be careful
        # cost = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
        # d/dX[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[l,j]
        #           = sum_j a[i,j] * (sum_l b[k,l] X[j,l]) + sum_l b[k,l] * (sum_j a[j,i] X[l,j])
        # Let Y = X @ b_sym.T  => Y[j,l] = sum_k X[j,k] b_sym[k,l]
        # Let Z = X @ a_sym.T   => Z[j,l] = sum_k X[j,k] a_sym[k,l]
        # Then g[i,k] = sum_j a[i,j] Y[j,k] + sum_l b[k,l] Z[l,i]
        #           = (a @ Y.T).T[i,k]... let's just compute directly with matrix ops

        Y = X @ b_sym.T  # Y[j, k] = sum_l X[j, l] b_sym[l, k]
        Z = X @ a_sym.T  # Z[j, l] = sum_k X[j, k] a_sym[k, l]

        # g[i, k] = sum_j a[i, j] * Y[j, k] + sum_l b_sym[k, l] * Z[l, i]
        g = a @ Y.T  # g1[i, k] = sum_j a[i, j] Y[j, k]
        g = g + b_sym @ Z.T  # g2[i, k] = sum_l b_sym[k, l] Z[l, i] -- wait, dimensions
        # b_sym @ Z.T: [k,l] dot [l,i] -> [k,i], but we want [i,k]
        # So g2[i,k] = sum_l b_sym[k,l] Z[l,i] = (Z @ b_sym.T)[i,k]
        g2 = Z @ b_sym.T
        g = g + g2

        # Velocity Verlet
        V = V + g * lr
        X_new = X + V
        # Project to doubly stochastic (Sinkhorn-Knopp)
        for _ in range(10):
            row_sums = X_new.sum(axis=1, keepdims=True)
            X_new = X_new / np.maximum(row_sums, 1e-10)
            col_sums = X_new.sum(axis=0, keepdims=True)
            X_new = X_new / np.maximum(col_sums, 1e-10)

        # Damp to prevent divergence
        X_new = 0.99 * X_new + 0.01 * (1.0 / n)
        X = X_new

        # Sample at certain steps
        if step % 100 == 50:
            # Convert X to permutation using Hungarian
            # We want to assign row i to col argmax X[i, :] approximately
            # Use Hungarian on -X to maximize
            row_ind, col_ind = linear_sum_assignment(-X)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            permutations.append(p.copy())

    if not permutations:
        row_ind, col_ind = linear_sum_assignment(-X)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        permutations.append(p)

    return permutations


def _final_polish(a, b, p, current_cost, time_limit, start_time):
    """SA polish with tight cooling."""
    n = len(p)
    best_p = p.copy()
    best_cost = current_cost

    T_start = max(1.0, current_cost * 0.003)
    T_end = 0.005
    elapsed_total = max(time_limit, 1.0)
    rng = np.random.RandomState(123)

    while time.time() - start_time < time_limit:
        i = rng.randint(n)
        j = rng.randint(n)
        if i == j:
            continue

        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = current_cost + delta

        elapsed = time.time() - start_time
        frac = min(elapsed / elapsed_total, 1.0)
        T = T_start * (T_end / T_start) ** frac

        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-10)):
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()

    return best_p, best_cost


def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()

    best_p = None
    best_cost = float('inf')

    # Collect initial permutations from multiple strategies
    initial_perms = []

    # 1. Hungarian
    p_hung = _hungarian_initial(a, b)
    initial_perms.append(p_hung)

    # 2. Greedy
    p_greedy = _greedy_initial(a, b)
    initial_perms.append(p_greedy)

    # 3. Symplectic trajectory samples (deterministic, diverse)
    remaining_for_init = total_deadline - (time.time() - start_time)
    if remaining_for_init > 5:
        try:
            symplectic_perms = _symplectic_trajectory_init(
                a, b, n, num_samples=8, steps=400, lr=0.015, seed=7
            )
            initial_perms.extend(symplectic_perms[:4])
        except Exception:
            pass

    # 4. A couple random starts
    rng = np.random.RandomState(99)
    initial_perms.append(_random_initial(n, rng))
    initial_perms.append(_random_initial(n, rng))

    # Deduplicate (by tuple)
    seen = set()
    unique_perms = []
    for perm in initial_perms:
        key = tuple(perm.tolist())
        if key not in seen:
            seen.add(key)
            unique_perms.append(perm)

    # Local search on each initial permutation
    for idx, p_init in enumerate(unique_perms):
        remaining = total_deadline - (time.time() - start_time)
        if remaining < 5:
            break

        # Allocate time proportionally: earlier starts get more time
        # but cap each to allow others to run
        time_per_start = min(remaining * 0.6, 30.0)
        if idx == len(unique_perms) - 1:
            time_per_start = remaining - 5  # last one gets the rest

        p_work = p_init.copy()
        current_cost = _qap_cost(a, b, p_work)

        p_work, current_cost = _local_search_fast(
            a, b, p_work,
            time_limit=time_per_start,
            start_time=time.time()
        )

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p_work.copy()

    # Final polish
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 3 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        p, current_cost = _final_polish(a, b, p, current_cost, remaining_time - 2, start_time)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()

    if best_p is None:
        best_p = _greedy_initial(a, b)

    return best_p.tolist()
# EVOLVE-BLOCK-END
