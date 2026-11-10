"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: return best result if we get too close to the wall-clock limit
_START_TIME = time.time()
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _compute_radius(centers):
    """Compute the max common radius for a set of centers in [-0.5, 0.5]^3."""
    n = len(centers)
    if n < 2:
        return 0.0
    
    # Minimum distance to walls
    wall_dists = np.min(np.min(0.5 - np.abs(centers), axis=1), axis=0)
    r_wall = float(wall_dists)
    
    # Minimum pairwise distance / 2
    # Use a more efficient approach for larger N
    min_pair_dist = float('inf')
    for i in range(n):
        # Vectorized distance from center i to all others
        diffs = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        if len(dists) > 0:
            d = float(np.min(dists))
            if d < min_pair_dist:
                min_pair_dist = d
    
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)


def _pairwise_min_dist(centers):
    """Compute minimum pairwise distance."""
    n = len(centers)
    if n < 2:
        return float('inf')
    min_d = float('inf')
    for i in range(n):
        diffs = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        if len(dists) > 0:
            d = float(np.min(dists))
            if d < min_d:
                min_d = d
    return min_d


def _force_directed_relaxation(centers, max_iters=500, lr=0.01):
    """
    Force-directed relaxation to push spheres apart while keeping them in the cube.
    Uses a smooth surrogate: repulsion ~ 1/d^2 when close, attraction to center when far.
    """
    centers = centers.astype(np.float64).copy()
    n = len(centers)
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        forces = np.zeros_like(centers)
        
        # Repulsion between pairs (smoothed)
        for i in range(n):
            diffs = centers[i+1:] - centers[i]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1)) + 1e-12
            # Smooth repulsion: stronger when close
            rep = 1.0 / (dists + 0.05)
            force_i = np.sum(diffs * rep[:, np.newaxis], axis=0)
            forces[i] += force_i
            for j in range(i+1, n):
                forces[j] -= diffs[j - i - 1] * rep[j - i - 1]
        
        # Wall forces: push inward when close to walls
        for axis in range(3):
            # Left wall (x < 0)
            close_left = centers[:, axis] < -0.45
            forces[close_left, axis] += (1.0 + 10.0 * (centers[close_left, axis] + 0.5)) ** 2
            # Right wall (x > 0)
            close_right = centers[:, axis] > 0.45
            forces[close_right, axis] -= (1.0 + 10.0 * (0.5 - centers[close_right, axis])) ** 2
        
        # Centering force (weak, to keep things balanced)
        forces -= 0.01 * centers
        
        # Update positions
        centers += lr * forces
        # Clip to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Check improvement
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Adaptive learning rate
        if it % 50 == 0:
            lr *= 0.95
    
    return best_centers, best_r


def _local_optimization(centers, max_iters=200):
    """
    Try to improve using scipy.optimize.minimize on a subset or the full set.
    Objective: minimize negative of the minimum radius (i.e., maximize r).
    """
    if time.time() > DEADLINE - SAFETY_MARGIN:
        return centers, _compute_radius(centers)
    
    n = len(centers)
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    # Flatten for optimization
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(-1, 3)
        # Penalty for going outside
        outside_penalty = 0.0
        for axis in range(3):
            outside_penalty += np.sum(np.maximum(0, c[:, axis] - 0.4999)) ** 2
            outside_penalty += np.sum(np.maximum(0, -0.4999 - c[:, axis])) ** 2
        if outside_penalty > 0.01:
            return -0.01 + outside_penalty
        r = _compute_radius(c)
        return -r
    
    # Try a few random perturbations and optimize
    rng = np.random.default_rng(42)
    for trial in range(3):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        perturbation = rng.normal(0, 0.01, size=x0.shape)
        x_trial = x0 + perturbation
        # Clip
        for axis in range(3):
            x_trial[axis::3] = np.clip(x_trial[axis::3], -0.4999, 0.4999)
        
        try:
            result = minimize(objective, x_trial, method='L-BFGS-B', 
                              options={'maxiter': max_iters, 'ftol': 1e-8})
            c_trial = result.x.reshape(-1, 3)
            c_trial = np.clip(c_trial, -0.5, 0.5)
            r_trial = _compute_radius(c_trial)
            if r_trial > best_r:
                best_r = r_trial
                best_centers = c_trial.copy()
        except Exception:
            pass
    
    return best_centers, best_r


def _generate_grid_start():
    """Generate a good initial grid-based arrangement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # m = 5 for N=100 (5^3 = 125 >= 100)
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _generate_body_centered_start():
    """
    Generate a body-centered cubic-like arrangement.
    For 100 spheres in a unit cube, try a 5x5x5 FCC-like pattern.
    """
    # Start with grid
    m = 5
    s = 1.0 / m
    centers = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * s - 0.5
                y = (j + 0.5) * s - 0.5
                z = (k + 0.5) * s - 0.5
                centers.append([x, y, z])
                if len(centers) >= N:
                    break
                if len(centers) >= N:
                    break
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    
    centers = np.array(centers[:N], dtype=np.float64)
    return centers


def _fcco_start():
    """
    Face-centered cubic inspired arrangement.
    In FCC, spheres are at corners and face centers of the cube.
    Scale to fit 100 spheres.
    """
    # Generate FCC lattice points in a larger region, then select 100 closest to center
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC: points at (i,j,k) where i+j+k is even, or shifted
                x = i * 0.5
                y = j * 0.5
                z = k * 0.5
                # FCC condition: all coordinates integers or all half-integers with sum even
                if (i + j + k) % 2 == 0:
                    points.append([x, y, z])
    
    points = np.array(points)
    # Sort by distance from center
    dists = np.sum(points * points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    # Scale to fit in [-0.5, 0.5]^3
    max_coord = np.max(np.abs(points[:N]))
    if max_coord > 0:
        scale = 0.48 / max_coord
        points = points * scale
    
    return points[:N].astype(np.float64)


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Multiple initial strategies, pick the best
    strategies = [
        ("grid", _generate_grid_start),
        ("bcc", _generate_body_centered_start),
        ("fcc", _fcco_start),
    ]
    
    best_centers = None
    best_r = -1.0
    
    for name, gen_func in strategies:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        try:
            centers = gen_func()
        except Exception:
            continue
        
        # Force-directed relaxation
        centers_fd, r_fd = _force_directed_relaxation(centers, max_iters=200)
        
        # Local optimization
        centers_opt, r_opt = _local_optimization(centers_fd, max_iters=50)
        
        if r_opt > best_r:
            best_r = r_opt
            best_centers = centers_opt.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        best_centers = _generate_grid_start()
        best_r = _compute_radius(best_centers)
    
    # Final polish with more iterations if time permits
    if time.time() < DEADLINE - SAFETY_MARGIN:
        centers_final, r_final = _force_directed_relaxation(best_centers, max_iters=300, lr=0.005)
        if r_final > best_r:
            best_centers = centers_final
            best_r = r_final
        
        if time.time() < DEADLINE - SAFETY_MARGIN:
            centers_final2, r_final2 = _local_optimization(best_centers, max_iters=100)
            if r_final2 > best_r:
                best_centers = centers_final2
                best_r = r_final2
    
    # Ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
