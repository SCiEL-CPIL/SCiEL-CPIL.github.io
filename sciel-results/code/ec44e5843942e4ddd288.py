"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_init(n, rng):
    """Generate a hexagonal lattice of points inside a disk, return n points."""
    # Estimate spacing so that ~n points fit in unit disk
    # Area of unit disk = pi, each hex cell area ~ (sqrt(3)/2)*s^2
    s = np.sqrt(2.0 * np.pi / (n * 1.0)) * 1.05  # slightly denser
    # Build lattice
    pts = []
    k = int(1.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            pts.append((x, y))
    pts = np.array(pts)
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[r <= 1.0]
    if len(pts) < n:
        # Add random points to fill
        while len(pts) < n:
            angle = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0, 1.0)
            pts = np.vstack([pts, [[rad * np.cos(angle), rad * np.sin(angle)]]])
    # Shuffle and take n
    idx = rng.permutation(len(pts))[:n]
    return pts[idx]


def _compute_radius(pts):
    """Compute the largest common radius for a set of points."""
    n = len(pts)
    # Min distance between pairs
    min_pair = np.inf
    for i in range(n):
        dx = pts[i, 0] - pts[i + 1:, 0]
        dy = pts[i, 1] - pts[i + 1:, 1]
        d = np.hypot(dx, dy)
        if d.min() < min_pair:
            min_pair = d.min()
    # Min distance to boundary
    r_bound = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair / 2.0, r_bound)


def _relax(pts, n_iter, rng, deadline):
    """Force-based relaxation. Returns best points and best radius."""
    pts = pts.copy()
    n = len(pts)
    best_radius = _compute_radius(pts)
    best_pts = pts.copy()

    # Parameters
    repel_strength = 1.0
    contain_strength = 2.0
    damping = 0.8
    step_scale = 0.001

    for it in range(n_iter):
        if time.time() > deadline:
            break
        # Compute forces
        forces = np.zeros_like(pts)
        # Pairwise repulsion
        for i in range(n):
            dx = pts[i, 0] - pts[i + 1:, 0]
            dy = pts[i, 1] - pts[i + 1:, 1]
            d = np.hypot(dx, dy)
            # Avoid division by zero
            d = np.where(d < 1e-10, 1e-10, d)
            # Force magnitude: 1/d^2 (strong when close)
            f_mag = repel_strength / (d * d)
            fx = f_mag * dx / d
            fy = f_mag * dy / d
            forces[i, 0] += fx.sum()
            forces[i, 1] += fy.sum()
            # Apply equal and opposite to others
            forces[i + 1:, 0] -= fx
            forces[i + 1:, 1] -= fy

        # Containment: push inward if too close to boundary
        r = np.hypot(pts[:, 0], pts[:, 1])
        # Target: keep r < 1 - margin
        margin = 0.05
        over = np.maximum(0, r - (1.0 - margin))
        # Force direction: toward center
        inv_r = np.where(r < 1e-10, 0.0, 1.0 / r)
        forces[:, 0] -= contain_strength * over * inv_r * pts[:, 0]
        forces[:, 1] -= contain_strength * over * inv_r * pts[:, 1]

        # Update
        step = step_scale * (1.0 + 0.5 * np.cos(it * 0.01))  # oscillating step
        pts += damping * step * forces

        # Clamp to unit disk
        r = np.hypot(pts[:, 0], pts[:, 1])
        over = r > 1.0
        if over.any():
            inv_r = np.where(r[over] < 1e-10, 1.0, 1.0 / r[over])
            pts[over, 0] *= (1.0 - 1e-6) * inv_r[over]
            pts[over, 1] *= (1.0 - 1e-6) * inv_r[over]

        # Evaluate
        rad = _compute_radius(pts)
        if rad > best_radius:
            best_radius = rad
            best_pts = pts.copy()

    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    rng = np.random.default_rng(42)

    best_radius = -1.0
    best_pts = None

    # Multiple restarts
    n_restarts = 5
    iters_per_restart = 5000

    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        # Initialize
        if restart == 0:
            pts = _hex_init(N, rng)
        else:
            # Perturb the best found so far
            if best_pts is not None:
                pts = best_pts + rng.normal(0, 0.01, size=(N, 2))
            else:
                pts = _hex_init(N, rng)

        # Relax
        pts, rad = _relax(pts, iters_per_restart, rng, deadline)
        if rad > best_radius:
            best_radius = rad
            best_pts = pts

    # Final check
    if best_pts is None:
        best_pts = _hex_init(N, rng)
        best_radius = _compute_radius(best_pts)

    return best_pts
# EVOLVE-BLOCK-END
