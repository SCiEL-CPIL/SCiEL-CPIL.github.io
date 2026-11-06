"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

DEADLINE = 140.0  # Safety margin within 150s limit
KNOWN_BEST = 0.036612798904


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


def _smooth_energy_and_grad(centres, r_target):
    """
    Compute smooth energy and its gradient.
    Energy: sum of squared violations of pairwise dist >= 2*r_target and wall dist >= r_target.
    Uses a smooth approximation for the min over walls.
    """
    n = len(centres)
    x = centres[:, 0]
    y = centres[:, 1]
    
    energy = 0.0
    grad = np.zeros_like(centres)
    
    # Pairwise: want min dist >= 2*r_target
    # Penalty: sum over pairs of (2*r_target - dist)^2 for dist < 2*r_target
    for i in range(n):
        for j in range(i + 1, n):
            dx = centres[i, 0] - centres[j, 0]
            dy = centres[i, 1] - centres[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
            if dist < 2.0 * r_target:
                violation = 2.0 * r_target - dist
                energy += violation * violation
                inv_dist = 1.0 / dist
                grad[i, 0] += 2.0 * violation * (-dx * inv_dist)
                grad[i, 1] += 2.0 * violation * (-dy * inv_dist)
                grad[j, 0] += 2.0 * violation * (dx * inv_dist)
                grad[j, 1] += 2.0 * violation * (dy * inv_dist)
    
    # Wall distances - use smooth minimum via log-sum-exp
    # For each point, the wall distance is min of 4 distances
    # We use a soft-min: -log(sum(exp(-d_i/beta)))*beta approx min(d_i) for small beta
    # But for gradient computation, let's just use the hard min with subgradient
    # Actually, let's use a smooth penalty: sum of (r_target - d_i)^2 for each wall
    # This is equivalent to pushing away from all walls, which is fine
    
    for i in range(n):
        dx_left = x[i] + 0.5
        dx_right = 0.5 - x[i]
        dy_bottom = y[i] + 0.5
        dy_top = 0.5 - y[i]
        
        # Smooth penalty: sum over all 4 walls of (r_target - d)^2 if d < r_target
        for d in [dx_left, dx_right, dy_bottom, dy_top]:
            if d < r_target:
                violation = r_target - d
                energy += 0.25 * violation * violation  # Scale down to avoid over-penalizing
        
        # Gradient for each wall
        if dx_left < r_target:
            grad[i, 0] += 0.25 * 2.0 * (r_target - dx_left) * (-1.0)
        if dx_right < r_target:
            grad[i, 0] += 0.25 * 2.0 * (r_target - dx_right) * (1.0)
        if dy_bottom < r_target:
            grad[i, 1] += 0.25 * 2.0 * (r_target - dy_bottom) * (-1.0)
        if dy_top < r_target:
            grad[i, 1] += 0.25 * 2.0 * (r_target - dy_top) * (1.0)
    
    return energy, grad


def _hex_grid_init():
    """Generate a hexagonal grid initialization for ~200 points."""
    # For N=200, use a hexagonal packing
    # dx = 2r, dy = 2r*sqrt(3)/2 = r*sqrt(3)
    # To fit 200 points: roughly sqrt(200) ~ 14 per side
    # Let's try different spacings
    
    best_pts = None
    best_r = 0
    
    for spacing_mult in np.linspace(0.9, 1.3, 20):
        r_est = 0.5 / (np.sqrt(N) * spacing_mult)
        dx = 2 * r_est
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
            # Keep the N closest to center
            centers = np.mean(pts, axis=0)
            dists_to_center = np.linalg.norm(pts - centers, axis=1)
            idx = np.argsort(dists_to_center)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            # Add random points
            while len(pts) < N:
                pts = np.vstack([pts, np.random.uniform(-0.4, 0.4, size=2)])
        
        r_test = _compute_radius(pts)
        if r_test > best_r:
            best_r = r_test
            best_pts = pts
    
    return best_pts


def _initial_grid():
    """Create initial square grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_initial():
    """Create random initial placement inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.9


def _optimize_for_radius(centres, r_target, deadline, max_iter=300):
    """
    Optimize centres to achieve radius >= r_target.
    Uses L-BFGS-B with explicit gradients on smooth energy with bounds.
    """
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (2 * N)
    
    def objective(x):
        if time.time() > deadline:
            return np.inf
        c = x.reshape(-1, 2)
        e, _ = _smooth_energy_and_grad(c, r_target)
        return e
    
    def jac(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        c = x.reshape(-1, 2)
        _, g = _smooth_energy_and_grad(c, r_target)
        return g.flatten()
    
    try:
        result = minimize(
            objective,
            centres.flatten(),
            jac=jac,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 5, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result.fun < np.inf:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return centres


def _polish_centres(centres, deadline):
    """
    High-precision polish pass: iteratively fix the bottleneck constraint.
    Uses smaller steps and more iterations for precision.
    """
    current = centres.copy()
    current_radius = _compute_radius(current)
    
    for iteration in range(500):
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
            step = 0.0001
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
            
            step = 0.0001
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
        if current_radius >= KNOWN_BEST - 1e-8:
            break
    
    return current


def _iterative_radius_increase(centres, deadline):
    """
    Iteratively increase target radius using binary search.
    Start from current radius and try to push higher.
    """
    current = centres.copy()
    current_radius = _compute_radius(current)
    best_centres = current.copy()
    best_radius = current_radius
    
    # Binary search for maximum achievable radius
    r_low = current_radius
    r_high = min(KNOWN_BEST * 1.05, current_radius * 1.3)
    
    max_iterations = 15
    for _ in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to optimize for r_mid
        optimized = _optimize_for_radius(current, r_mid, deadline - 5, max_iter=250)
        optimized_radius = _compute_radius(optimized)
        
        if optimized_radius >= r_mid - 1e-6:
            # Success: can achieve r_mid, try higher
            r_low = r_mid
            if optimized_radius > best_radius:
                best_radius = optimized_radius
                best_centres = optimized.copy()
            current = optimized.copy()
        else:
            # Failure: can't achieve r_mid, try lower
            r_high = r_mid
    
    # Final polish
    if time.time() < deadline - 2:
        polished = _polish_centres(best_centres, deadline - 2)
        polished_radius = _compute_radius(polished)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centres = polished
    
    return best_centres, best_radius


def construct():
    """Construct 200 circle centres in the unit square to maximize common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    # Phase 1: Multiple restarts with different initializations
    best_centres = _hex_grid_init()
    best_radius = _compute_radius(best_centres)
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        
        if restart == 0:
            centres = _hex_grid_init()
        elif restart == 1:
            centres = _initial_grid()
        elif restart == 2:
            centres = _random_initial()
        elif restart == 3:
            centres = _hex_grid_init() + np.random.randn(N, 2) * 0.003
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        elif restart == 4:
            centres = best_centres + np.random.randn(N, 2) * 0.005
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        elif restart == 5:
            centres = best_centres + np.random.randn(N, 2) * 0.002
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        elif restart == 6:
            centres = _hex_grid_init() + np.random.randn(N, 2) * 0.01
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        else:
            centres = best_centres + np.random.randn(N, 2) * 0.001
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        
        # Adaptive target: try to push toward known best
        r_target = min(KNOWN_BEST, max(0.032, best_radius * 1.03))
        
        if time.time() < deadline - 35:
            centres = _optimize_for_radius(centres, r_target, deadline - 35, max_iter=300)
        
        current_radius = _compute_radius(centres)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = centres.copy()
    
    # Phase 2: Iterative radius increase on the best solution
    if time.time() < deadline - 15:
        best_centres, best_radius = _iterative_radius_increase(best_centres, deadline - 5)
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 2:
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
