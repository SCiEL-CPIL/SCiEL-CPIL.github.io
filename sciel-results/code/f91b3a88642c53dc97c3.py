"""Pack 50 equal circles in a unit circle.

Uses a physics-inspired approach:
1. Start from a good hexagonal initial configuration.
2. Use gradient-based optimization (L-BFGS-B from scipy) to minimize a
   smooth objective that penalizes overlaps and boundary violations.
3. Iteratively increase the target radius and re-optimize, using the
   previous solution as the initial guess for the next radius.
4. Finish with a short simulated-annealing polish to squeeze out extra
   radius.
"""

import numpy as np
from scipy.optimize import minimize

N = 50


def _hex_init(r):
    """Return a hexagonal lattice of up to N points inside the unit circle
    with spacing 2*r, then top up with random points if needed."""
    spacing = 2.0 * r
    pts = []
    k = 8
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2.0
            if x * x + y * y <= (1.0 - r) ** 2:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < N:
        # Add random points
        rng = np.random.default_rng(42)
        while len(pts) < N:
            x, y = rng.uniform(-1, 1, size=2)
            if x * x + y * y <= (1.0 - r) ** 2:
                pts = np.vstack([pts, [x, y]])
    pts = pts[:N]
    # Shuffle for diversity
    rng2 = np.random.default_rng(123)
    idx = rng2.permutation(N)
    return pts[idx]


def _objective_flat(x, r):
    """Objective on flat vector x of length 2N."""
    c = x.reshape(N, 2)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu, ju = np.triu_indices(N, k=1)
    d = dists[iu, ju]
    # Smooth overlap penalty: for each pair, max(0, 2r - d)^2
    ov = np.maximum(0.0, 2.0 * r - d)
    e_pairs = np.sum(ov ** 2)
    # Boundary
    radii = np.sqrt(np.sum(c ** 2, axis=1))
    bv = np.maximum(0.0, radii + r - 1.0)
    e_bound = np.sum(bv ** 2)
    return e_pairs + 10.0 * e_bound


def _objective_grad_flat(x, r):
    """Objective and gradient on flat vector."""
    c = x.reshape(N, 2)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    iu, ju = np.triu_indices(N, k=1)
    d = dists[iu, ju]
    ov = np.maximum(0.0, 2.0 * r - d)
    e_pairs = np.sum(ov ** 2)
    # Gradient of pairs
    grad = np.zeros_like(c)
    for i, j in zip(iu, ju):
        if ov[i] > 0:
            direction = (c[i] - c[j]) / dists_safe[i, j]
            # d/ dx_i of (2r - d)^2 = -2(2r-d) * d/d x_i (d) = -2(2r-d) * (x_i - x_j)/d
            # So gradient contribution to x_i is -2*ov*(c[i]-c[j])/d
            grad[i] += -2.0 * ov[i] * direction
            grad[j] += 2.0 * ov[i] * direction
    # Boundary
    radii = np.sqrt(np.sum(c ** 2, axis=1))
    radii_safe = np.where(radii < 1e-12, 1e-12, radii)
    bv = np.maximum(0.0, radii + r - 1.0)
    e_bound = np.sum(bv ** 2)
    for i in range(N):
        if bv[i] > 0:
            direction = c[i] / radii_safe[i]
            # d/ dx_i of (|c_i| + r - 1)^2 = 2*(|c_i|+r-1) * (c_i/|c_i|)
            grad[i] += 2.0 * 10.0 * bv[i] * direction
    return e_pairs + 10.0 * e_bound, grad.flatten()


def _sa_polish(centers, r, max_iter=8000, seed=7):
    """Short simulated annealing polish."""
    rng = np.random.default_rng(seed)
    c = centers.copy()
    best_c = c.copy()
    best_e = _objective_flat(c, r)
    T = 0.005
    T_min = 1e-7
    cooling = 0.998
    for it in range(max_iter):
        idx = rng.integers(N)
        old = c[idx].copy()
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, 0.05)
        c[idx] += dist * np.array([np.cos(angle), np.sin(angle)])
        e = _objective_flat(c, r)
        if e < best_e:
            best_e = e
            best_c = c.copy()
        if e <= best_e or rng.random() < np.exp(-(e - best_e) / T):
            pass  # accept
        else:
            c[idx] = old
        T *= cooling
        if T < T_min:
            break
    return best_c, best_e


def construct():
    # Start near known best
    r_target = 0.1258254895304
    centers = _hex_init(r_target)

    # First, optimize at the known best radius to get a clean starting point
    x0 = centers.flatten()
    res = minimize(_objective_flat, x0, args=(r_target,), method='L-BFGS-B',
                   options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
    centers = res.x.reshape(N, 2)
    e0 = _objective_flat(centers, r_target)

    # Now try to increase radius in small steps
    current_r = r_target
    current_centers = centers
    step = 0.00002
    for attempt in range(600):
        new_r = current_r + step
        x0 = current_centers.flatten()
        res = minimize(_objective_flat, x0, args=(new_r,), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        test_centers = res.x.reshape(N, 2)
        test_e = _objective_flat(test_centers, new_r)
        if test_e < 1e-6:
            current_r = new_r
            current_centers = test_centers
            # Keep step small for fine-tuning
            step = 0.00001
        else:
            # Try a smaller step once
            step *= 0.5
            if step < 1e-7:
                break

    # SA polish at the final radius
    current_centers, _ = _sa_polish(current_centers, current_r, max_iter=5000)

    # Verify validity: compute the actual achievable radius
    diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu, ju = np.triu_indices(N, k=1)
    min_pair = np.min(dists[iu, ju]) / 2.0
    radii = np.sqrt(np.sum(current_centers ** 2, axis=1))
    max_r = np.max(radii)
    boundary_r = 1.0 - max_r
    achievable = min(min_pair, boundary_r)

    # If the packing is slightly invalid, rescale to make it valid
    if achievable < current_r:
        scale = achievable / current_r
        current_centers = current_centers * scale

    return current_centers
