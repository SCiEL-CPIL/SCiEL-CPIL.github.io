"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
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


def _local_search(a, b, p, deadline, max_passes=200):
    """
    First-improvement local search with vectorized 2-opt swap deltas.
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
            pi = p[i]
            Bp = b[np.ix_(p, p)]
            b_pi_row = b[pi, p]
            diff_B = Bp - b_pi_row[np.newaxis, :]

            a_diff_B = a * diff_B
            sum_a_diff = a_diff_B.sum(axis=1)
            a_i_diff = a[i, :] @ diff_B
            term1 = a_i_diff - sum_a_diff

            b_p_pi = b[p, pi]
            diff_B2 = Bp.T - b_p_pi[np.newaxis, :]
            a_T_diff_B2 = a.T * diff_B2
            sum_aT_diff = a_T_diff_B2.sum(axis=1)
            a_i_col_diff = a[:, i] @ diff_B2
            term2 = a_i_col_diff - sum_aT_diff

            b_pj_pi = b[p, pi]
            b_pi_pj = b[pi, p]
            b_pj_pj = b[p, p]
            b_pi_pi = b[pi, pi]

            self1 = a[i, p] * (b_pj_pi - b_pi_pj)
            self2 = a[p, i] * (b_pi_pj - b_pj_pi)
            self3 = a[i, i] * (b_pj_pj - b_pi_pi)
            self4 = a[p, p] * (b_pi_pi - b_pj_pj)

            delta_all = term1 + term2 + self1 + self2 + self3 + self4
            delta_all[i] = 0

            mask_j = np.zeros(n, dtype=bool)
            mask_j[i+1:] = True
            valid_deltas = delta_all[mask_j]
            if valid_deltas.size > 0 and valid_deltas.min() < 0:
                best_idx_in_mask = valid_deltas.argmin()
                j = np.where(mask_j)[0][best_idx_in_mask]
                p[i], p[j] = p[j], p[i]
                cost += int(delta_all[j])
                improved = True

        if not improved:
            break

    return p, cost


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization: align facility ordering with location ordering
    based on eigenvectors of the flow and distance matrices.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    k = min(15, n)
    idx_a = np.argsort(-np.abs(evals_a))[:k]
    fac_emb = evecs_a[:, idx_a]

    idx_b = np.argsort(-np.abs(evals_b))[:k]
    loc_emb = evecs_b[:, idx_b]

    fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
    fac_norm[fac_norm == 0] = 1
    fac_emb = fac_emb / fac_norm

    loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
    loc_norm[loc_norm == 0] = 1
    loc_emb = loc_emb / loc_norm

    cost_matrix = -fac_emb @ loc_emb.T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    return p


def _riemannian_cqap(a, b, n, rng, deadline, max_iters=500):
    """
    Riemannian optimization on the Stiefel manifold for CQAP.
    The permutation matrix X is orthonormal: X^T X = I.
    Cost: f(X) = sum_{i,j} a[i,j] * sum_{k,l} X[i,k] * b[k,l] * X[j,l]
         = trace(B^T X^T A X)
    Gradient: 2 * (A + A^T) X B^T  (for the non-symmetric case, more carefully:
              d/dX trace(B^T X^T A X) = (A + A^T) X B^T + (A^T + A) X B
              For symmetric A,B: 2 * A X B)
    
    We use the Riemannian gradient and geodesic steps (expmap) on the Stiefel manifold.
    """
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    # Initialize X as a random orthonormal matrix (via QR of random Gaussian)
    Z = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(Z)
    # Ensure Q is properly oriented
    d = np.diag(R)
    ph = np.sign(d)
    ph[ph == 0] = 1
    X = Q * ph[np.newaxis, :]

    # Riemannian gradient of f(X) = trace(B^T X^T A X)
    # grad = 2 * A X B^T (for symmetric A, B)
    # More precisely: grad = A X B^T + A^T X B
    # For symmetric: 2 * A X B

    best_X = X.copy()
    best_val = np.inf

    step_size = 1.0 / (2.0 * np.linalg.norm(a_sym) * np.linalg.norm(b_sym))
    step_size = max(step_size, 1e-6)
    step_size = min(step_size, 1.0)

    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Euclidean gradient
        grad_euc = 2.0 * (a_sym @ X @ b_sym)

        # Riemannian gradient: project to tangent space
        # Tangent space at X: {X * Omega + X_perp * M} where Omega is skew-symmetric
        # Projection: P_X(grad) = grad - X * (X^T @ grad + (X^T @ grad)^T) / 2
        XTG = X.T @ grad_euc
        grad_riem = grad_euc - X * ((XTG + XTG.T) / 2.0)

        # Geodesic step (expmap): X_new = X * exp(Omega) where Omega is skew-symmetric
        # For small steps, use first-order: X_new = X + step_size * grad_riem
        # Then re-orthonormalize via QR
        X_new = X + step_size * grad_riem

        # Re-orthonormalize
        Q_new, R_new = np.linalg.qr(X_new)
        d_new = np.diag(R_new)
        ph_new = np.sign(d_new)
        ph_new[ph_new == 0] = 1
        X_new = Q_new * ph_new[np.newaxis, :]

        # Evaluate cost
        val = np.sum(b_sym * (X_new.T @ a_sym @ X_new))

        if val < best_val:
            best_val = val
            best_X = X_new.copy()

        # Adaptive step size
        if val > best_val * 1.01:
            step_size *= 0.7
        elif val < best_val * 0.99:
            step_size = min(step_size * 1.1, 1.0)

        X = X_new

    # Extract permutation from best_X using Hungarian
    row_ind, col_ind = linear_sum_assignment(-best_X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    return p


def _quantum_perturb(p, rng, n, temperature=0.5):
    """
    Quantum-inspired perturbation: probabilistic multi-swap moves.
    """
    perturbed_p = p.copy()

    base_swaps = 2
    max_swaps = 8
    n_swaps = base_swaps + int(temperature * (max_swaps - base_swaps))
    n_swaps = max(1, min(n_swaps, n // 2))

    for _ in range(n_swaps):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

    if rng.random() < 0.1 * temperature:
        block_len = rng.integers(3, min(10, n // 3))
        start = rng.integers(0, n - block_len + 1)
        perturbed_p[start:start + block_len] = perturbed_p[start:start + block_len][::-1]

    return perturbed_p


def _surrogate_perturb(a, b, p, current_cost, rng, n):
    """
    Surrogate-guided perturbation: identify facilities/locations with high
    contribution to cost and perturb those positions.
    """
    perturbed_p = p.copy()

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    contrib = a_row_sums * b_row_sums[p]

    k = 3 + int(rng.integers(0, 4))
    top_idx = np.argsort(-contrib)[:k]

    for idx in top_idx:
        if rng.random() < 0.5:
            j = rng.integers(0, n)
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]
        else:
            low_idx = np.argsort(contrib)[:k]
            j = low_idx[rng.integers(0, k)]
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]

    return perturbed_p


def _best_improvement_pass(a, b, p, deadline):
    """
    One pass of best-improvement using vectorized delta computation.
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_delta = 0
    best_i = -1
    best_j = -1

    Bp = b[np.ix_(p, p)]

    for i in range(n):
        if time.time() > deadline:
            break
        pi = p[i]
        b_pi_p = b[pi, p]
        diff_B = Bp - b_pi_p[np.newaxis, :]

        a_diff_B = a * diff_B
        sum_a_diff = a_diff_B.sum(axis=1)
        a_i_diff = a[i, :] @ diff_B
        term1 = a_i_diff - sum_a_diff

        b_p_pi = b[p, pi]
        diff_B2 = Bp.T - b_p_pi[np.newaxis, :]
        a_T_diff_B2 = a.T * diff_B2
        sum_aT_diff = a_T_diff_B2.sum(axis=1)
        a_i_col_diff = a[:, i] @ diff_B2
        term2 = a_i_col_diff - sum_aT_diff

        b_pj_pi = b[p, pi]
        b_pi_pj = b[pi, p]
        b_pj_pj = b[p, p]
        b_pi_pi = b[pi, pi]

        self1 = a[i, p] * (b_pj_pi - b_pi_pj)
        self2 = a[p, i] * (b_pi_pj - b_pj_pi)
        self3 = a[i, i] * (b_pj_pj - b_pi_pi)
        self4 = a[p, p] * (b_pi_pi - b_pj_pj)

        delta_all = term1 + term2 + self1 + self2 + self3 + self4
        delta_all[i] = 0

        mask_j = np.zeros(n, dtype=bool)
        mask_j[i+1:] = True
        valid_deltas = delta_all[mask_j]
        if valid_deltas.size > 0:
            min_d = valid_deltas.min()
            if min_d < best_delta:
                best_delta = min_d
                best_i = i
                best_j = np.where(mask_j)[0][valid_deltas.argmin()]

    if best_i >= 0:
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += int(best_delta)

    return p, cost, best_delta


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin before 150s kill

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    best_p = None
    best_cost = np.inf

    rng = np.random.default_rng(42)

    # Heuristic 1: Rank-matching
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

    # Heuristic 4: Riemannian CQAP
    if time.time() < deadline - 30.0:
        p4 = _riemannian_cqap(a, b, n, rng, deadline - 25.0, max_iters=300)
        p4, c4 = _local_search(a, b, p4, deadline)
        if c4 < best_cost:
            best_cost = c4
            best_p = p4.copy()

    # Heuristic 5: Random
    if time.time() < deadline - 10.0:
        p5 = rng.permutation(n).astype(np.int64)
        p5, c5 = _local_search(a, b, p5, deadline)
        if c5 < best_cost:
            best_cost = c5
            best_p = p5.copy()

    # --- Phase 2: Benders-style surrogate-guided master problem ---
    if time.time() < deadline - 15.0:
        surrogate = (a @ b[:, best_p].T + a.T @ b[best_p, :])
        row_ind, col_ind = linear_sum_assignment(surrogate)
        p_surr = np.empty(n, dtype=np.int64)
        p_surr[row_ind] = col_ind
        p_surr, c_surr = _local_search(a, b, p_surr, deadline - 10.0)
        if c_surr < best_cost:
            best_cost = c_surr
            best_p = p_surr.copy()

    # --- Phase 3: Iterated Local Search with mixed perturbations ---
    current_p = best_p.copy()
    current_cost = best_cost

    ilse_deadline = deadline - 10.0

    iteration = 0
    total_iterations_expected = 40
    while time.time() < ilse_deadline:
        iteration += 1
        progress = min(1.0, iteration / total_iterations_expected)
        temperature = 1.0 - 0.7 * progress

        r = rng.random()
        if r < 0.15 and time.time() < ilse_deadline - 8.0:
            # Riemannian CQAP perturbation
            perturbed_p = _riemannian_cqap(a, b, n, rng, ilse_deadline - 5.0, max_iters=100)
        elif r < 0.35:
            perturbed_p = _quantum_perturb(current_p, rng, n, temperature)
        elif r < 0.65:
            perturbed_p = _surrogate_perturb(a, b, current_p, current_cost, rng, n)
        else:
            k = 2 + int(temperature * 4)
            perturbed_p = current_p.copy()
            for _ in range(k):
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

        perturbed_p, perturbed_cost = _local_search(a, b, perturbed_p, ilse_deadline)

        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost < current_cost:
            delta = perturbed_cost - current_cost
            accept_prob = np.exp(-delta / (temperature * 1000.0))
            if rng.random() < accept_prob:
                current_p = perturbed_p
                current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            if rng.random() < 0.1:
                current_p = perturbed_p
                current_cost = perturbed_cost

    # --- Phase 4: Aggressive final polish with best-improvement ---
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        for _ in range(50):
            if time.time() > polish_deadline:
                break
            polish_p, polish_cost, bd = _best_improvement_pass(a, b, polish_p, polish_deadline)
            if bd == 0:
                break
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()

    # --- Phase 5: Final first-improvement polish ---
    if time.time() < deadline - 0.5:
        final_p = best_p.copy()
        final_p, final_cost = _local_search(a, b, final_p, deadline - 0.1, max_passes=5)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
