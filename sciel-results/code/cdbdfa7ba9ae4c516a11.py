"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # safety margin before 150s kill


def _compute_radius(c):
    """Compute the achievable common radius for centres c (N,3)."""
    n = c.shape[0]
    # Wall distances
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    min_dist = np.inf
    for i in range(n - 1):
        d = np.linalg.norm(c[i+1:] - c[i], axis=1)
        m = np.min(d)
        if m < min_dist:
            min_dist = m
    return min(min_dist / 2.0, min_wall)


def _make_grid(nx, ny, nz):
    """Create a grid of points in [-0.5, 0.5]^3."""
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N]


def _relax_spheres(c, r_target, max_iter=500, lr_init=0.02, lr_min=0.001,
                   alpha_coarse=30, alpha_fine=200, momentum=0.9):
    """
    Relax spheres using repulsive forces to achieve radius r_target.
    Uses a two-phase approach: coarse then fine alpha (repulsion strength).
    """
    c = c.copy()
    n = c.shape[0]
    limit = 0.5 - r_target
    
    best_c = c.copy()
    best_obj = float('inf')
    velocity = np.zeros_like(c)
    
    # Phase 1: coarse (large alpha, larger steps)
    # Phase 2: fine (smaller alpha, smaller steps)
    
    for phase in range(2):
        if phase == 0:
            alpha = alpha_coarse
            lr = lr_init
            phase_iters = max_iter // 2
        else:
            alpha = alpha_fine
            lr = lr_init * 0.3
            phase_iters = max_iter - max_iter // 2
        
        for it in range(phase_iters):
            force = np.zeros_like(c)
            
            # Wall repulsion: push spheres away from walls
            for k in range(3):
                # Left wall
                mask_l = c[:, k] < -limit
                if np.any(mask_l):
                    force[mask_l, k] += alpha * (limit + c[mask_l, k])  # push right
                # Right wall
                mask_r = c[:, k] > limit
                if np.any(mask_r):
                    force[mask_r, k] -= alpha * (c[mask_r, k] - limit)  # push left
            
            # Pairwise repulsion
            for i in range(n - 1):
                diff = c[i] - c[i+1:]
                d = np.linalg.norm(diff, axis=1)
                # Avoid division by zero
                mask = d < 2.0 * r_target
                if np.any(mask):
                    # Repulsive force proportional to overlap
                    overlap = 2.0 * r_target - d[mask]
                    # Normalize direction
                    diff_norm = diff[mask] / d[mask, np.newaxis]
                    force[i] += alpha * np.sum(overlap[:, np.newaxis] * diff_norm, axis=0)
                    force[i+1:] -= alpha * overlap[:, np.newaxis] * diff_norm
            
            # Update with momentum
            velocity = momentum * velocity + force
            c = c + lr * velocity
            c = np.clip(c, -limit, limit)
            
            # Track best
            obj = _compute_radius(c)
            if obj > best_obj:
                best_obj = obj
                best_c = c.copy()
            
            if best_obj >= r_target - 1e-9:
                return best_c, best_obj
    
    return best_c, best_obj


def _local_search(c, r_current, time_budget=30.0):
    """
    Local search: try to increase radius by perturbing the configuration.
    Uses a penalty-based optimization approach.
    """
    t_start = time.time()
    best_c = c.copy()
    best_r = r_current
    
    # Try small random perturbations and relax
    for attempt in range(200):
        if time.time() - t_start > time_budget:
            break
        
        # Random perturbation
        c_perturbed = c + np.random.randn(*c.shape) * 0.01
        limit = 0.5 - best_r
        c_perturbed = np.clip(c_perturbed, -limit, limit)
        
        # Try to relax to a slightly larger radius
        r_target = best_r * 1.005
        c_relaxed, r_achieved = _relax_spheres(c_perturbed, r_target, max_iter=100)
        
        if r_achieved > best_r + 1e-9:
            best_c = c_relaxed.copy()
            best_r = r_achieved
    
    return best_c, best_r


def construct():
    t0 = time.time()
    
    # Initial grid: 5 x 5 x 4 = 100
    c0 = _make_grid(5, 5, 4)
    r_current = _compute_radius(c0)
    best_c = c0.copy()
    best_r = r_current
    
    # Phase 1: Gradually increase radius
    r_target = r_current * 1.01
    max_r_target = 0.115
    
    while time.time() - t0 < _DEADLINE - 60:  # Leave 60s for local search
        if r_target > max_r_target:
            break
        
        c_relaxed, r_achieved = _relax_spheres(best_c, r_target, max_iter=300)
        
        if r_achieved >= r_target - 1e-9:
            best_c = c_relaxed.copy()
            best_r = r_achieved
            r_target *= 1.01
        else:
            # If we can't achieve r_target, try a smaller increment
            r_target = best_r * 1.002
    
    # Phase 2: Local search to squeeze out more radius
    if time.time() - t0 < _DEADLINE - 10:
        best_c, best_r = _local_search(best_c, best_r, time_budget=_DEADLINE - time.time() - 10)
    
    # Final check: ensure we return a valid configuration
    final_r = _compute_radius(best_c)
    if final_r > best_r:
        best_r = final_r
    
    return best_c
# EVOLVE-BLOCK-END
