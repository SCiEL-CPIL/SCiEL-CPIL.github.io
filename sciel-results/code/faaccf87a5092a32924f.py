"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 10.0


def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0

    # Min distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))

    # Min pair distance via vectorized computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary)


def _hex_lattice_init(spacing, angle_deg=0):
    """Generate hexagonal lattice points within the unit disk."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rows.append((xr, yr))

    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]

    if len(points) < N:
        return None

    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def _radial_ring_init():
    """Place circles in concentric rings."""
    best_points = None
    best_score = -np.inf

    for n_rings in range(5, 16):
        weights = []
        for i in range(n_rings):
            r_i = (i + 1) / n_rings * 0.95
            weights.append(2 * np.pi * r_i)

        total_w = sum(weights)
        counts = []
        remaining = N
        for i in range(n_rings - 1):
            c_i = max(1, int(round(N * weights[i] / total_w)))
            c_i = min(c_i, remaining - (n_rings - 1 - i))
            counts.append(c_i)
            remaining -= c_i
        counts.append(remaining)

        points = []
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            r_i = (i + 1) / n_rings * 0.95
            offset = (i * np.pi) / max(1, count)
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for a in angles:
                points.append((r_i * np.cos(a), r_i * np.sin(a)))

        if len(points) != N:
            continue

        points = np.array(points)
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    return best_points


def _random_init():
    """Random initialization within unit disk."""
    points = np.random.randn(N, 2)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    points *= 0.95
    return points


def _soft_min_obj_and_grad(x, beta, iu):
    """
    Compute soft-min objective and gradient for L-BFGS-B.
    x: flattened (N*2,) array
    beta: temperature parameter
    iu: upper triangle indices
    Returns (value, gradient)
    """
    n = N
    p = x.reshape(n, 2)

    i_idx, j_idx = iu
    vecs = p[i_idx] - p[j_idx]
    pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0

    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms

    # Use log-sum-exp trick for numerical stability
    z_pairs = -beta * pair_dists
    z_bounds = -beta * bound_dists

    max_z_pair = np.max(z_pairs)
    max_z_bound = np.max(z_bounds)
    max_z = max(max_z_pair, max_z_bound)

    exp_pairs = np.exp(z_pairs - max_z)
    exp_bounds = np.exp(z_bounds - max_z)

    sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
    log_sum_exp = max_z + np.log(sum_exp)

    val = log_sum_exp / beta

    # Gradient computation
    w_pairs = exp_pairs / sum_exp
    w_bounds = exp_bounds / sum_exp

    grad = np.zeros_like(x)
    p_grad = grad.reshape(n, 2)

    # Pairwise gradient contribution
    # d(pair_dist)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|) for pair_dist = |p_i-p_j|/2
    # So d(-pair_dist)/d(p_i) = -(p_i - p_j) / (2 * |p_i - p_j|)
    # But we're minimizing val = -soft_min, so we need d(val)/dx
    # val = (1/beta) * log(sum(exp(-beta * dist)))
    # d(val)/dx = (1/beta) * (1/sum) * sum(exp(-beta*dist)) * (-beta * d(dist)/dx)
    #           = -sum(w_k * d(dist_k)/dx)

    # For pair dist k between i and j: dist_k = |p_i - p_j| / 2
    # d(dist_k)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|)
    # d(dist_k)/d(p_j) = (p_j - p_i) / (2 * |p_i - p_j|)

    norms_pair = np.sqrt(np.sum(vecs**2, axis=1))
    norms_pair[norms_pair < 1e-12] = 1e-12
    unit_vecs = vecs / (2.0 * norms_pair[:, np.newaxis])

    # Accumulate gradients for pairs
    np.add.at(p_grad, i_idx, -w_pairs[:, np.newaxis] * unit_vecs)
    np.add.at(p_grad, j_idx, w_pairs[:, np.newaxis] * unit_vecs)

    # Boundary gradient
    # dist_b = 1 - |p_i|
    # d(dist_b)/d(p_i) = -p_i / |p_i|
    norms_safe = norms.copy()
    norms_safe[norms_safe < 1e-12] = 1e-12
    unit_pos = p / norms_safe[:, np.newaxis]

    p_grad += w_bounds[:, np.newaxis] * (-unit_pos)

    return val, grad


def _local_optimize_lbfgs(points, deadline=None):
    """
    Local optimization using L-BFGS-B with soft-min objective.
    Uses adaptive beta schedule to refine the packing.
    """
    n = N
    iu = np.triu_indices(n, k=1)

    x0 = points.flatten()
    best_x = x0.copy()
    best_score = _compute_score(points)

    # Adaptive beta schedule: start moderate, increase over time
    beta_schedule = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]

    start_time = time.time()

    for beta in beta_schedule:
        if deadline is not None and time.time() > deadline:
            break

        def obj_and_grad(x):
            return _soft_min_obj_and_grad(x, beta, iu)

        # L-BFGS-B optimization
        try:
            result = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20
                }
            )

            if result.success or result.fun < obj_and_grad(x0)[0]:
                x0 = result.x
                # Clip to unit circle
                p = x0.reshape(n, 2)
                norms = np.linalg.norm(p, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    p[mask] = p[mask] / norms[mask, np.newaxis]
                x0 = p.flatten()

                score = _compute_score(p)
                if score > best_score:
                    best_score = score
                    best_x = x0.copy()

        except Exception:
            pass

    return best_x.reshape(n, 2)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Generate multiple initial configurations
    configs = []

    # Hexagonal lattice with different angles
    for angle in [0, 15, 30, 45, 60]:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        for _ in range(20):
            pts = _hex_lattice_init(spacing, angle)
            if pts is not None:
                # Add small perturbation to break symmetry
                pts = pts + np.random.normal(0, 1e-4, pts.shape)
                configs.append(pts)
                break
            spacing *= 0.98

    # Radial ring initialization
    ring_pts = _radial_ring_init()
    if ring_pts is not None:
        ring_pts = ring_pts + np.random.normal(0, 1e-4, ring_pts.shape)
        configs.append(ring_pts)

    # Random initializations
    for _ in range(3):
        configs.append(_random_init())

    # Optimize each configuration and keep the best
    best_points = None
    best_score = -np.inf

    for cfg in configs:
        if time.time() > deadline:
            break

        # Local optimization with L-BFGS-B
        optimized = _local_optimize_lbfgs(cfg, deadline=deadline)
        score = _compute_score(optimized)

        if score > best_score:
            best_score = score
            best_points = optimized.copy()

    # If no configuration was optimized, use the first one as fallback
    if best_points is None:
        best_points = configs[0] if configs else np.zeros((N, 2))

    return best_points
# EVOLVE-BLOCK-END
