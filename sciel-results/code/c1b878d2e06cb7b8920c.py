# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the arithmetic Kakeya entropy ratio D/M on a 10x10 grid,
    where D = H(x - y) and M = max(H(x), H(x + y), H(x + 2y)).
    Uses direct high-dimensional optimization with multiple restarts.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    # Precompute projection indices
    # P is indexed as P[y, x], so cell (x, y) has index y*n + x
    ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    # Flatten: index = y*n + x
    idx = ys * n + xs  # shape (n, n), value = y*n + x

    # Projection values for each slope
    proj_neg1 = xs - ys          # x - y  (D)
    proj_0    = xs               # x      (M component)
    proj_1    = xs + ys          # x + y  (M component)
    proj_2    = xs + 2*ys        # x + 2y (M component)

    # For each projection, build a mapping from value -> list of cell indices
    # This lets us compute projected distributions efficiently
    def build_value_map(proj):
        """Map each projection value to the list of flat indices."""
        vals = proj.flatten()
        unique_vals = np.unique(vals)
        vmap = {}
        for v in unique_vals:
            mask = (vals == v)
            vmap[v] = np.where(mask)[0]
        return unique_vals, vmap

    vmap_neg1 = build_value_map(proj_neg1)
    vmap_0    = build_value_map(proj_0)
    vmap_1    = build_value_map(proj_1)
    vmap_2    = build_value_map(proj_2)

    def compute_entropy_from_probs(probs):
        """Shannon entropy in bits, ignoring zero probabilities."""
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def compute_ratio(P_flat):
        """
        Compute D/M ratio from a flat probability vector.
        P_flat has shape (n*n,), indexed as P_flat[y*n + x].
        """
        # Projected distributions
        # D: distribution of x - y
        D_vals = vmap_neg1[0]
        D_probs = np.array([P_flat[i].sum() for i in vmap_neg1[1].values()])
        H_D = compute_entropy_from_probs(D_probs)

        # M: max of H(x), H(x+y), H(x+2y)
        M_probs_0 = np.array([P_flat[i].sum() for i in vmap_0[1].values()])
        M_probs_1 = np.array([P_flat[i].sum() for i in vmap_1[1].values()])
        M_probs_2 = np.array([P_flat[i].sum() for i in vmap_2[1].values()])

        H_0 = compute_entropy_from_probs(M_probs_0)
        H_1 = compute_entropy_from_probs(M_probs_1)
        H_2 = compute_entropy_from_probs(M_probs_2)

        M = max(H_0, H_1, H_2)

        if M < 1e-9:
            return -1.0  # invalid

        return H_D / M

    def objective(log_theta):
        """
        Objective function for minimization (we minimize -D/M).
        log_theta is the log of the unnormalized weights.
        We use a softmax-like normalization: P = exp(log_theta) / sum(exp(log_theta))
        To avoid overflow, we subtract the max.
        """
        if time.time() > deadline:
            return 0.0  # timeout guard, return neutral value
        theta_max = np.max(log_theta)
        log_weights = log_theta - theta_max
        weights = np.exp(log_weights)
        P_flat = weights / np.sum(weights)
        ratio = compute_ratio(P_flat)
        if ratio < 0:
            return 1000.0  # large penalty for invalid
        return -ratio

    def objective_grad(log_theta):
        """
        Objective with numerical gradient for L-BFGS-B.
        We use finite differences since the entropy has kinks.
        Actually, let's just use the objective without gradient and let L-BFGS-B use finite differences.
        """
        return objective(log_theta)

    # Multi-start optimization
    best_P_flat = None
    best_score = -np.inf

    # Start 1: uniform distribution
    uniform_log = np.zeros(n * n)
    res = minimize(objective, uniform_log, method='L-BFGS-B',
                   options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    if res.success or res.fun < 1000:
        theta_max = np.max(res.x)
        P_flat = np.exp(res.x - theta_max)
        P_flat /= np.sum(P_flat)
        score = compute_ratio(P_flat)
        if score > best_score:
            best_score = score
            best_P_flat = P_flat.copy()

    # Random restarts
    rng = np.random.default_rng(42)
    n_restarts = 30

    for i in range(n_restarts):
        if time.time() > deadline:
            break

        # Random initial point in log-space
        # Mix of different strategies:
        # - sparse (most mass on few cells)
        # - structured (diagonals, rows, columns)
        # - random smooth

        strategy = i % 4

        if strategy == 0:
            # Sparse: pick k random cells, give them high weight
            k = rng.integers(3, 12)
            log_theta = np.full(n * n, -10.0)
            chosen = rng.choice(n * n, size=k, replace=False)
            log_theta[chosen] = rng.uniform(0, 5, size=k)
        elif strategy == 1:
            # Diagonal emphasis: cells where x - y is constant get high weight
            # This directly boosts D
            diag_val = rng.integers(-5, 6)
            log_theta = np.full(n * n, -5.0)
            for y in range(n):
                for x in range(n):
                    if x - y == diag_val:
                        log_theta[y * n + x] = rng.uniform(2, 8)
        elif strategy == 2:
            # Row or column emphasis: concentrate on a few rows or columns
            # This can reduce M (H(x) or H(y) projections)
            if rng.random() < 0.5:
                # Concentrate on few columns (x values)
                log_theta = np.full(n * n, -5.0)
                n_cols = rng.integers(2, 5)
                chosen_cols = rng.choice(n, size=n_cols, replace=False)
                for c in chosen_cols:
                    for y in range(n):
                        log_theta[y * n + c] = rng.uniform(3, 10)
            else:
                # Concentrate on few rows (y values)
                log_theta = np.full(n * n, -5.0)
                n_rows = rng.integers(2, 5)
                chosen_rows = rng.choice(n, size=n_rows, replace=False)
                for r in chosen_rows:
                    for x in range(n):
                        log_theta[r * n + x] = rng.uniform(3, 10)
        else:
            # Smooth random: low-rank structure
            A = rng.normal(0, 1, size=(n, 4))
            B = rng.normal(0, 1, size=(4, n))
            M_mat = A @ B
            log_theta = M_mat.flatten() - np.max(M_mat)
            log_theta = np.clip(log_theta, -10, 5)

        res = minimize(objective, log_theta, method='L-BFGS-B',
                       options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < 1000:
            theta_max = np.max(res.x)
            P_flat = np.exp(res.x - theta_max)
            P_flat /= np.sum(P_flat)
            score = compute_ratio(P_flat)
            if score > best_score:
                best_score = score
                best_P_flat = P_flat.copy()

    # Fallback: uniform if nothing worked
    if best_P_flat is None:
        best_P_flat = np.ones(n * n) / (n * n)

    # Reshape to (n, n) with P[y, x]
    P = best_P_flat.reshape(n, n)

    # Final validation and clipping
    P = np.clip(P, 0, None)
    total = P.sum()
    if total < 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P /= total

    return P


# EVOLVE-BLOCK-END
