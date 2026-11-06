"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

DEADLINE = 140.0  # Safety margin within 150s limit


def _compute_radius(centres):
    """Compute the maximum common radius for a given set of centres."""
    if len(centres) < 2:
        return 0.0
    # Minimum distance to walls
    wall_min = min(0.5 - np.min(centres[:, 0]), 0.5 - np.max(centres[:, 0]),
                   0.5 - np.min(centres[:, 1]), 0.5 - np.max(centres[:, 1]))
    # Minimum pairwise distance / 2
    dists = pdist(centres)
    min_pair_dist = np.min(dists) if len(dists) > 0 else 1.0
    return min(wall_min, min_pair_dist / 2.0)


def _smooth_energy(centres, r_target, beta=50.0):
    """
    Smooth surrogate energy using log-sum-exp for soft-min.
    Minimizes violations of: pairwise dist >= 2*r_target and wall dist >= r_target.
    """
    x = centres[:, 0]
    y = centres[:, 1]
    
    # Pairwise: we want min dist >= 2*r_target
    # Penalty: sum over pairs of exp(-beta * (dist - 2*r_target)) for dist < 2*r_target
    dists = pdist(centres)
    violations = np.maximum(0, 2.0 * r_target - dists)
    # Use soft penalty: sum of violations^2 for close pairs + log-sum-exp style
    if np.any(violations > 0):
        close_dists = dists[dists < 2.0 * r_target]
        close_dists = np.maximum(close_dists, 1e-12)
        # Quadratic penalty for close pairs
        pair_penalty = np.sum((2.0 * r_target - close_dists) ** 2)
    else:
        pair_penalty = 0.0
    
    # Wall distances
    dx_left = x - (-0.5)
    dx_right = 0.5 - x
    dy_bottom = y - (-0.5)
    dy_top = 0.5 - y
    
    wall_dists = np.minimum(
        np.minimum(dx_left, dx_right),
        np.minimum(dy_bottom, dy_top)
    )
    wall_violations = np.maximum(0, r_target - wall_dists)
    wall_penalty = np.sum(wall_violations ** 2)
    
    return pair_penalty + wall_penalty


def _hex_grid_init():
    """Generate a hexagonal grid initialization for ~200 points."""
    dx = np.sqrt(2.0 / (N * np.sqrt(3)))
    dy = dx * np.sqrt(3) / 2.0
    
    pts = []
    x_start = -0.5 + dx / 2.0
    y_start = -0.5 + dy / 2.0
    
    row = 0
    y = y_start
    while y <= 0.5 - dy / 2.0 + 1e-6:
        offset = 0.0 if row % 2 == 0 else dx / 2.0
        x = x_start + offset
        while x <= 0.5 - dx / 2.0 + 1e-6:
            pts.append([x, y])
            x += dx
        row += 1
        y += dy
    
    pts = np.array(pts)
    
    if len(pts) > N:
        centers = np.mean(pts, axis=0)
        dists_to_center = np.linalg.norm(pts - centers, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        pts = pts[idx]
    elif len(pts) < N:
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-0.4, 0.4, size=2)])
    
    return pts


def _initial_grid():
    """Create initial square grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_initial():
    """Create random initial placement inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.9


def _optimize_for_radius(centres, r_target, deadline, max_iter=500):
    """
    Optimize centres to achieve radius >= r_target.
    Uses L-BFGS-B on smooth energy with bounds.
    """
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (2 * N)
    
    def objective(x):
        if time.time() > deadline:
            return np.inf
        c = x.reshape(-1, 2)
        return _smooth_energy(c, r_target)
    
    try:
        result = minimize(
            objective,
            centres.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 10, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.fun < np.inf:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return centres


def _polish_centres(centres, deadline):
    """
    High-precision polish pass: iteratively fix the bottleneck constraint.
    Uses Nelder-Mead on the subset of points involved in the minimum distance.
    """
    current = centres.copy()
    current_radius = _compute_radius(current)
    
    for iteration in range(100):
        if time.time() > deadline - 1:
            break
        
        # Compute all pairwise distances and wall distances
        dists = pdist(current)
        dist_matrix = squareform(dists)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Wall distances
        wall_dists = np.minimum(
            np.minimum(0.5 - np.abs(current[:, 0]), 0.5 - np.abs(current[:, 1])),
            0.5
        )
        
        # Find the bottleneck
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall is bottleneck
            idx = np.argmin(wall_dists)
            x, y = current[idx]
            d_left = x - (-0.5)
            d_right = 0.5 - x
            d_bottom = y - (-0.5)
            d_top = 0.5 - y
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            step = 0.0005
            if min_d == d_left:
                current[idx, 0] += step
            elif min_d == d_right:
                current[idx, 0] -= step
            elif min_d == d_bottom:
                current[idx, 1] += step
            else:
                current[idx, 1] -= step
            
            # Clamp
            current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_radius = _compute_radius(current)
            if new_radius > current_radius:
                current_radius = new_radius
            else:
                # Revert
                if min_d == d_left:
                    current[idx, 0] -= step
                elif min_d == d_right:
                    current[idx, 0] += step
                elif min_d == d_bottom:
                    current[idx, 1] -= step
                else:
                    current[idx, 1] += step
                current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
                current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        else:
            # Pair is bottleneck
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            d = current[i] - current[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-12
                dist = np.linalg.norm(d)
            d = d / dist
            
            step = 0.0005
            current[i] += step * d
            current[j] -= step * d
            
            current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_radius = _compute_radius(current)
            if new_radius > current_radius:
                current_radius = new_radius
            else:
                current[i] -= step * d
                current[j] += step * d
                current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
                current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check convergence
        if current_radius >= 0.036612798904 - 1e-8:
            break
    
    return current


def construct():
    """Construct 200 circle centres in the unit square to maximize common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_centres = _hex_grid_init()
    best_radius = _compute_radius(best_centres)
    
    np.random.seed(42)
    
    # Phase 1: Multiple restarts with different initializations
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            centres = _hex_grid_init()
        elif restart == 1:
            centres = _initial_grid()
        elif restart == 2:
            centres = _random_initial()
        elif restart == 3:
            centres = _hex_grid_init() + np.random.randn(N, 2) * 0.005
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        elif restart == 4:
            centres = best_centres + np.random.randn(N, 2) * 0.008
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        elif restart == 5:
            centres = _hex_grid_init() + np.random.randn(N, 2) * 0.01
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        elif restart == 6:
            centres = best_centres + np.random.randn(N, 2) * 0.003
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        else:
            centres = _random_initial()
        
        # Adaptive target: try to push toward known best
        r_target = min(0.036612798904, max(0.03, best_radius * 1.02))
        
        if time.time() < deadline - 15:
            centres = _optimize_for_radius(centres, r_target, deadline - 15, max_iter=400)
        
        current_radius = _compute_radius(centres)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = centres.copy()
    
    # Phase 2: Final polish on the best solution
    if time.time() < deadline - 5:
        polished = _polish_centres(best_centres, deadline - 2)
        polished_radius = _compute_radius(polished)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centres = polished
    
    # Final verification: ensure all centres are strictly inside
    best_centres[:, 0] = np.clip(best_centres[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
    best_centres[:, 1] = np.clip(best_centres[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centres
# EVOLVE-BLOCK-END
