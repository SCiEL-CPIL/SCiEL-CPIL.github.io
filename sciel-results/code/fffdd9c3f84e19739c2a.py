"""Pack 200 equal circles in a unit square to maximize common radius.

Combines:
  - Hexagonal-lattice multi-start initialisation (structural idea from inspiration 1)
  - Force-directed repulsion with KDTree acceleration (structural idea from current)
  - L-BFGS-B optimisation of a smooth soft-min objective (refinement machinery from inspiration 1)
  - Aggressive high-beta polishing passes (from knowledge: 1.068 -> 1.83)

The soft-min objective smoothly approximates the min of all wall and pairwise
constraints, allowing gradient-based optimisation to navigate the non-smooth
landscape of the minimum-distance constraint.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 135  # seconds, leave margin before 150 s kill


# ---------------------------------------------------------------------------
# Core objective: soft-min of all constraints
# ---------------------------------------------------------------------------

def _soft_min(vals, beta):
    """Compute the soft minimum of a set of values (numerically stable)."""
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta


def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < N:
        return 0.0
    if np.any(~np.isfinite(pts)):
        return 0.0
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0

    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0

    r = min(wall_dist, min_pair)
    return r if r > 0 else 0.0


def _objective_and_grad(x, beta):
    """
    Objective function and gradient for L-BFGS-B.
    We want to MAXIMIZE the soft-min of constraints, so MINIMIZE -soft_min.
    """
    pts = x.reshape((N, 2))

    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)

    C = np.empty(4 * N + num_pairs)

    # Wall values
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]

    # Pair values
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    C[4*N:] = dists / 2.0

    # Soft min computation
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)

    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val

    # Weights
    weights = exp_shifted / sum_exp

    # Gradient computation
    grad_pts = np.zeros((N, 2))

    # Wall gradients
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]

    # Pair gradients
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)

    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)

    pair_weights = weights[4*N:]

    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)

    grad_pts = -grad_pts / beta

    return obj_val, grad_pts.flatten()


# ---------------------------------------------------------------------------
# Force-directed layout (fast coarse spreading)
# ---------------------------------------------------------------------------

def _force_directed_layout(pts, n_iters=2000, dt=0.005, repulsion_strength=0.5,
                           r_target=0.035):
    """
    Use repulsive forces to spread out points, with KDTree for efficiency.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)

    for it in range(n_iters):
        forces = np.zeros_like(pts)

        # Wall forces
        for i in range(n):
            if pts[i, 0] < -0.5 + r_target:
                forces[i, 0] += 5.0 * (r_target - (pts[i, 0] + 0.5))
            if pts[i, 0] > 0.5 - r_target:
                forces[i, 0] -= 5.0 * (pts[i, 0] - (0.5 - r_target))
            if pts[i, 1] < -0.5 + r_target:
                forces[i, 1] += 5.0 * (r_target - (pts[i, 1] + 0.5))
            if pts[i, 1] > 0.5 - r_target:
                forces[i, 1] -= 5.0 * (pts[i, 1] - (0.5 - r_target))

        # Repulsive forces using KDTree
        tree = KDTree(pts)
        cutoff = max(2 * r_target, 0.05)

        for i in range(n):
            neighbors = tree.query_ball_point(pts[i], cutoff)
            for j in neighbors:
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    forces[i] += np.random.randn(2) * 0.01
                    continue
                if dist < 2 * r_target:
                    overlap = 2 * r_target - dist
                    force_mag = 10.0 * overlap / dist
                    forces[i] += force_mag * diff

        pts += dt * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)

    return pts


# ---------------------------------------------------------------------------
# Initialisation strategies
# ---------------------------------------------------------------------------

def _generate_hex_initial(spacing=0.078, seed=42):
    """Generate a centered hexagonal lattice packing, select N best points."""
    rng = np.random.RandomState(seed)
    points = []
    row_h = spacing * np.sqrt(3) / 2

    y = -1.0
    row_idx = 0
    while y < 1.0:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -1.0 + shift
        while x < 1.0:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1

    pts = np.array(points)

    # Filter points inside the square
    mask = (pts[:, 0] >= -0.5) & (pts[:, 0] <= 0.5) & \
           (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
    pts = pts[mask]

    if len(pts) < N:
        return rng.uniform(-0.45, 0.45, (N, 2))

    # Add small jitter to break symmetry
    pts = pts + rng.randn(len(pts), 2) * 0.002
    pts = np.clip(pts, -0.5, 0.5)

    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]

    return pts


def _generate_grid_initial(rows=14, cols=15, seed=42):
    """Generate a regular grid and select N points."""
    rng = np.random.RandomState(seed)
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])

    if len(grid) >= N:
        # Jitter and select
        grid = grid + rng.randn(len(grid), 2) * 0.003
        grid = np.clip(grid, -0.5, 0.5)
        dists_to_center = np.sum(grid**2, axis=1)
        indices = np.argsort(dists_to_center)
        return grid[indices[:N]]
    else:
        return rng.uniform(-0.45, 0.45, (N, 2))


def _generate_random_initial(seed=42):
    """Random initialisation."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 2))


# ---------------------------------------------------------------------------
# Iterated Local Search (ILS) with vectorized repulsion
# ---------------------------------------------------------------------------

def _local_repulse(pts, n_iters=100, dt=0.005):
    """
    Vectorized local repulsion to spread points.
    """
    pts = pts.copy().astype(np.float64)
    for _ in range(n_iters):
        # Calculate differences and distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Avoid division by zero and huge forces
        dists_clipped = np.clip(dists, 0.01, 1.0)
        dists_inv_sq = 1.0 / (dists_clipped ** 2)
        
        # Force matrix: F[i,j] = (pts[i] - pts[j]) / dist^2
        F = diffs * dists_inv_sq[:, :, np.newaxis]
        
        # Sum forces for each point
        forces = np.sum(F, axis=1)
        
        # Wall forces: Push away from walls
        margin = 0.05
        wall_force = np.zeros_like(pts)
        # x-axis
        wall_force[:, 0] += np.where(pts[:, 0] < -0.5 + margin, (-0.5 + margin - pts[:, 0]) / margin, 0)
        wall_force[:, 0] += np.where(pts[:, 0] > 0.5 - margin, (0.5 - margin - pts[:, 0]) / margin, 0)
        # y-axis
        wall_force[:, 1] += np.where(pts[:, 1] < -0.5 + margin, (-0.5 + margin - pts[:, 1]) / margin, 0)
        wall_force[:, 1] += np.where(pts[:, 1] > 0.5 - margin, (0.5 - margin - pts[:, 1]) / margin, 0)
        
        forces += wall_force * 10.0
        
        pts += dt * forces
        pts = np.clip(pts, -0.5, 0.5)
        
    return pts

def _run_ils(pts, time_limit):
    """
    Iterated Local Search using _local_repulse as the local search procedure.
    """
    best_pts = pts.copy()
    best_r = _get_radius(best_pts)
    t_start = time.time()
    rng = np.random.RandomState(42)
    
    noise_scale = 0.02
    min_noise = 0.001
    
    while time.time() - t_start < time_limit:
        # Perturb
        perturbed = best_pts + rng.randn(N, 2) * noise_scale
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Local Search
        # 50 iterations of repulsion is fast
        improved = _local_repulse(perturbed, n_iters=50, dt=0.005)
        
        r_improved = _get_radius(improved)
        
        if r_improved > best_r:
            best_r = r_improved
            best_pts = improved
        else:
            # Decrease noise to refine
            noise_scale *= 0.9
            if noise_scale < min_noise:
                noise_scale = min_noise
                
    return best_pts, best_r


# ---------------------------------------------------------------------------
# Push-apart local search (fast, cheap refinement)
# ---------------------------------------------------------------------------

def _push_apart(pts, n_iters=300, strength=0.003):
    """Iteratively push apart the closest pairs of points."""
    pts = pts.copy().astype(np.float64)
    for it in range(n_iters):
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_idx
        min_dist = D[i, j]
        if min_dist < 1e-12:
            pts[j] += np.random.randn(2) * 0.01
            continue
        direction = pts[j] - pts[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = np.random.randn(2)
            norm = 1.0
        direction /= norm
        move = strength * direction
        new_i = pts[i] - move
        new_j = pts[j] + move
        if -0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5 and \
           -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5:
            pts[i] = new_i
            pts[j] = new_j
        else:
            for frac in [0.5, 0.25, 0.1]:
                move = strength * frac * direction
                new_i = pts[i] - move
                new_j = pts[j] + move
                if -0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5 and \
                   -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5:
                    pts[i] = new_i
                    pts[j] = new_j
                    break
    return pts


# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None

    # --- Phase 1: Multi-start initialisation + force-directed spreading ---
    # Try several diverse initialisations
    initialisations = [
        ("hex_0.078", _generate_hex_initial(0.078, seed=42)),
        ("hex_0.080", _generate_hex_initial(0.080, seed=123)),
        ("hex_0.076", _generate_hex_initial(0.076, seed=777)),
        ("grid_14x15", _generate_grid_initial(14, 15, seed=42)),
        ("grid_20x10", _generate_grid_initial(20, 10, seed=99)),
        ("random", _generate_random_initial(seed=42)),
    ]

    for name, init_pts in initialisations:
        if time.time() - t0 > _TIME_LIMIT - 30:
            break
        try:
            # Force-directed spreading
            pts_fd = _force_directed_layout(init_pts, n_iters=1500, dt=0.005,
                                            repulsion_strength=0.5, r_target=0.035)
            r_fd = _get_radius(pts_fd)
            if r_fd > best_r:
                best_r = r_fd
                best_pts = pts_fd
        except Exception:
            pass

    # Fallback if nothing worked
    if best_pts is None:
        best_pts = _generate_grid_initial(20, 10, seed=42)
        best_r = _get_radius(best_pts)

    # --- Phase 2: Iterated Local Search (ILS) on best candidate ---
    if time.time() - t0 < _TIME_LIMIT - 15:
        try:
            # Allocate remaining time minus safety margin for polishing
            remaining = _TIME_LIMIT - 15 - (time.time() - t0)
            if remaining > 10:
                pts_ils, r_ils = _run_ils(
                    best_pts, time_limit=remaining
                )
                if r_ils > best_r:
                    best_r = r_ils
                    best_pts = pts_ils
        except Exception:
            pass

    # --- Phase 3: Push-apart local refinement ---
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 5:
        try:
            pts_pushed = _push_apart(best_pts, n_iters=300, strength=0.003)
            r_pushed = _get_radius(pts_pushed)
            if r_pushed > best_r:
                best_r = r_pushed
                best_pts = pts_pushed
        except Exception:
            pass

    # --- Phase 4: Final high-beta polish if time permits ---
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 2:
        try:
            remaining = _TIME_LIMIT - 2 - (time.time() - t0)
            if remaining > 5:
                pts_polish, r_polish = _run_gradient_optimization(
                    best_pts, time_limit=remaining,
                    betas=[200, 500]
                )
                if r_polish > best_r:
                    best_r = r_polish
                    best_pts = pts_polish
        except Exception:
            pass

    if best_r <= 0:
        best_pts = _generate_grid_initial(20, 10, seed=42)

    return best_pts
