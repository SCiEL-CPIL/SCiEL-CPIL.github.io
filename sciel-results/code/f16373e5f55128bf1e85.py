"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(pts):
    """Compute the exact packing radius for given centers."""
    if len(pts) != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    dists = np.linalg.norm(pts, axis=1)
    if np.any(dists > 1.0):
        return -1.0
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = pts[i_idx] - pts[j_idx]
    pair_dists = np.linalg.norm(diffs, axis=1)
    min_pair_dist = np.min(pair_dists)
    min_radial_dist = 1.0 - np.max(dists)
    r = min(min_pair_dist / 2.0, min_radial_dist)
    return r

def _hex_lattice_init():
    """Generate a hexagonal lattice clipped to the unit circle."""
    best_pts = None
    best_score = -1.0

    for scale in [0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]:
        spacing = 2.0 / (np.sqrt(N) * scale)

        nx = int(2.0 / spacing) + 3
        ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3

        x_coords = (np.arange(-nx // 2, nx // 2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny // 2, ny // 2 + 1)) * spacing * np.sqrt(3.0) / 2.0

        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                    pts_list.append([px, py])

        if len(pts_list) < N:
            continue

        pts = np.array(pts_list)
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:N]]

        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts

    if best_pts is None:
        rng = np.random.RandomState(42)
        best_pts = rng.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
        best_score = _compute_score(best_pts)

    return best_pts, best_score

def _ring_init():
    """Generate ring-based initialization."""
    best_pts = None
    best_score = -1.0

    for n_rings in range(6, 22):
        ring_radii = np.linspace(0.05, 0.95, n_rings)
        ring_points = []

        for r_ring in ring_radii:
            est_r = 0.053
            n_circ = max(1, int(2 * np.pi * r_ring / (2 * est_r)))
            angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
            for a in angles:
                x = r_ring * np.cos(a)
                y = r_ring * np.sin(a)
                ring_points.append([x, y])

        ring_points.append([0.0, 0.0])
        ring_points = np.array(ring_points)

        if len(ring_points) > N:
            dists = np.linalg.norm(ring_points, axis=1)
            idx = np.argsort(dists)[:N]
            ring_points = ring_points[idx]
        elif len(ring_points) < N:
            continue

        score = _compute_score(ring_points)
        if score > best_score:
            best_score = score
            best_pts = ring_points.copy()

    return best_pts, best_score

def _soft_min_obj_and_grad(params, beta):
    """
    Smooth objective for L-BFGS-B with analytic gradient.
    Returns (negative soft-min radius, gradient).
    """
    pts = params.reshape(-1, 2)

    if not np.all(np.isfinite(pts)):
        return 1e10, np.zeros(2 * N)

    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = pts[i_idx] - pts[j_idx]
    pair_dists = np.sqrt(np.sum(diffs ** 2, axis=1) + 1e-15)

    dists = np.sqrt(np.sum(pts ** 2, axis=1) + 1e-15)
    radial_margins = 1.0 - dists

    pair_vals = pair_dists / 2.0
    radial_vals = radial_margins

    all_vals = np.concatenate([pair_vals, radial_vals])
    n_pairs = len(pair_vals)
    n_radial = len(radial_vals)

    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)

    soft_min_val = -1.0 / beta * (np.log(sum_exp) - beta * min_val)
    obj = -soft_min_val

    weights = exp_shifted / sum_exp

    grad_pair = -weights[:n_pairs] / beta
    grad_radial = -weights[n_pairs:] / beta

    grad_pts = np.zeros((N, 2))

    # Vectorized pairwise gradient accumulation
    for k in range(n_pairs):
        i, j = i_idx[k], j_idx[k]
        d = diffs[k]
        dist = pair_dists[k]
        if dist > 1e-10:
            unit_vec = d / dist
            factor = grad_pair[k] * 0.5
            grad_pts[i] += factor * unit_vec
            grad_pts[j] -= factor * unit_vec

    for i in range(N):
        dist = dists[i]
        if dist > 1e-10:
            unit_vec = pts[i] / dist
            grad_pts[i] -= grad_radial[i] * unit_vec

    grad_params = -grad_pts.flatten()

    return obj, grad_params

def _optimize_lbfgs(pts, beta, maxiter=50, maxfun=500):
    """Run L-BFGS-B optimization with given beta."""
    x0 = pts.flatten()
    bounds = [(-0.9999, 0.9999)] * (2 * N)
    try:
        result = minimize(
            _soft_min_obj_and_grad, x0, args=(beta,),
            method='L-BFGS-B', bounds=bounds,
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxfun,
                     'ftol': 1e-14, 'gtol': 1e-12,
                     'maxls': 50}
        )
        if result.fun < 1e9:
            new_points = result.x.reshape(-1, 2)
            return new_points
    except Exception:
        pass
    return pts

def _gd_polish(pts, lr=0.0005, n_steps=200, beta=5000.0):
    """Gradient descent polish with adaptive learning rate."""
    current = pts.copy()
    best = current.copy()
    best_score = _compute_score(best)

    for step in range(n_steps):
        obj, grad = _soft_min_obj_and_grad(current.flatten(), beta)
        if not np.isfinite(obj):
            break

        # Adaptive learning rate
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-10:
            break

        # Scale learning rate based on gradient magnitude
        effective_lr = lr * min(1.0, 0.01 / max(g_norm, 1e-10))

        new_pts = (current - effective_lr * grad).reshape(-1, 2)
        # Clamp to bounds
        new_pts = np.clip(new_pts, -0.9999, 0.9999)

        new_score = _compute_score(new_pts)
        if new_score > best_score:
            best = new_pts.copy()
            best_score = new_score
            current = new_pts
        else:
            # Reduce learning rate
            lr *= 0.7
            if lr < 1e-8:
                break

    return best, best_score

def _anneal_beta(pts, betas, deadline):
    """Anneal through beta values, optimizing at each stage."""
    current = pts.copy()
    best = pts.copy()
    best_score = _compute_score(best)

    for beta in betas:
        if time.time() > deadline - 10:
            break

        new_pts = _optimize_lbfgs(current, beta, maxiter=40, maxfun=400)
        new_score = _compute_score(new_pts)

        if new_score > best_score:
            best = new_pts.copy()
            best_score = new_score
            current = new_pts.copy()

    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Generate initial points from multiple methods
    hex_pts, hex_score = _hex_lattice_init()
    ring_pts, ring_score = _ring_init()

    if ring_score > hex_score:
        best_pts = ring_pts.copy()
        best_score = ring_score
    else:
        best_pts = hex_pts.copy()
        best_score = hex_score

    # 2. Multi-stage beta annealing with L-BFGS-B
    betas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]

    best_pts, best_score = _anneal_beta(best_pts, betas, deadline)

    # 3. Also optimize the other initialization if time permits
    other_pts = hex_pts if ring_score > hex_score else ring_pts
    if time.time() < deadline - 30:
        other_best, other_score = _anneal_beta(other_pts, betas[2:], deadline)
        if other_score > best_score:
            best_pts = other_best
            best_score = other_score

    # 4. Gradient descent polish
    if time.time() < deadline - 15:
        polished, polished_score = _gd_polish(best_pts, lr=0.0005, n_steps=300, beta=10000.0)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score

    # 5. Final high-precision L-BFGS-B polish
    if time.time() < deadline - 8:
        final_pts = _optimize_lbfgs(best_pts, 50000.0, maxiter=80, maxfun=800)
        final_score = _compute_score(final_pts)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score

    # 6. Another GD polish at very high beta
    if time.time() < deadline - 5:
        polished2, polished2_score = _gd_polish(best_pts, lr=0.0002, n_steps=200, beta=50000.0)
        if polished2_score > best_score:
            best_pts = polished2
            best_score = polished2_score

    # 7. Final validation
    final_score = _compute_score(best_pts)
    if final_score <= 0:
        pts, _ = _hex_lattice_init()
        return pts

    return best_pts
# EVOLVE-BLOCK-END
