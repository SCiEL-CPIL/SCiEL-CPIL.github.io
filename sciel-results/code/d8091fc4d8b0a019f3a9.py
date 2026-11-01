import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
import random

N = 50
KNOWN_BEST_R = 0.071377103865

def _compute_radius(pos):
    """Compute the maximum valid radius for a given configuration."""
    pos = np.asarray(pos).reshape(N, 2)
    # Pairwise distances
    dists = pdist(pos)
    min_dist = np.min(dists) if len(dists) > 0 else np.inf
    
    # Wall distances
    # Distance from each point to the nearest wall
    # x in [-0.5, 0.5], so distance to wall is min(x + 0.5, 0.5 - x)
    wall_dists = np.minimum(np.minimum(pos[:, 0] + 0.5, 0.5 - pos[:, 0]),
                            np.minimum(pos[:, 1] + 0.5, 0.5 - pos[:, 1]))
    min_wall = np.min(wall_dists)
    
    return min(min_dist / 2.0, min_wall)

def _violation(pos, r):
    """
    Compute the total violation for a configuration with target radius r.
    Violation = sum of (2r - d_ij)^2 for all overlapping pairs + wall violations.
    Returns 0 if feasible.
    """
    pos = np.asarray(pos).reshape(N, 2)
    # Pairwise
    dists = pdist(pos)
    # We only care about pairs where dist < 2r
    # To be efficient, we can just compute all, but for gradient we need pairs.
    # For simple violation check:
    # overlap = 2*r - dists
    # viol = np.sum(np.maximum(0, overlap)**2)
    
    # Wall violations
    L = -0.5 + r
    R = 0.5 - r
    # x violations
    x_viol = np.maximum(0, L - pos[:, 0]) + np.maximum(0, pos[:, 0] - R)
    y_viol = np.maximum(0, L - pos[:, 1]) + np.maximum(0, pos[:, 1] - R)
    wall_viol = np.sum(x_viol**2) + np.sum(y_viol**2)
    
    # Pair violations
    # For large N, pdist is fast.
    # If dists < 2r, there is a violation.
    # Violation magnitude: (2r - dist)^2
    # We can vectorize this.
    # However, pdist returns condensed vector. We need to map back or just sum squares of positive overlaps.
    overlap = 2 * r - dists
    pair_viol = np.sum(np.maximum(0, overlap)**2)
    
    return pair_viol + wall_viol

def _grad_violation(pos, r):
    """
    Compute the gradient of the violation function.
    This is used for projected gradient descent.
    """
    pos = np.asarray(pos).reshape(N, 2)
    grad = np.zeros_like(pos)
    
    # Pairwise gradient
    # d/dx_i sum_{j>i} (2r - ||x_i - x_j||)^2
    # = sum_{j>i} 2(2r - d_ij) * (x_i - x_j) / d_ij
    # For j < i, the term is (2r - d_ij) * (x_i - x_j) / d_ij * 2?
    # Let V = sum_{i<j} (2r - d_ij)^2 + wall terms.
    # dV/dx_i = sum_{j != i} 2(2r - d_ij) * (x_i - x_j) / d_ij
    
    # Efficiently compute using cdist or loops?
    # N=50, so 50x50 matrix is fine.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # (N, N, 2)
    dists = np.linalg.norm(diff, axis=2) # (N, N)
    np.fill_diagonal(dists, 1.0) # Avoid division by zero
    
    # overlap_matrix[i, j] = 2r - dists[i, j]
    # Gradient contribution from pair (i, j) to x_i:
    # 2 * (2r - d_ij) * (x_i - x_j) / d_ij
    # Note: The sum in V is over i<j.
    # dV/dx_i = sum_{j<i} 2(2r-d_ij)(x_i-x_j)/d_ij + sum_{j>i} 2(2r-d_ij)(x_i-x_j)/d_ij
    # Actually, the derivative of (2r - ||x_i - x_j||)^2 w.r.t x_i is
    # 2(2r - ||x_i - x_j||) * (-(x_i - x_j)/||x_i - x_j||) * (-1)?
    # Let f(x) = (2r - ||x_i - x_j||)^2.
    # df/dx_i = 2(2r - d) * d/dx_i (2r - d) = 2(2r - d) * (-d/dx_i d)
    # d/dx_i ||x_i - x_j|| = (x_i - x_j) / ||x_i - x_j||
    # So df/dx_i = 2(2r - d) * (-(x_i - x_j)/d) = -2(2r - d)(x_i - x_j)/d
    
    # Wait, if I define V = sum (2r - d)^2, and d decreases as x_i moves towards x_j.
    # If x_i moves towards x_j, d decreases, (2r-d) increases, V increases.
    # So gradient should point AWAY from x_j if overlapping?
    # Let's check: x_i = 0, x_j = 1, 2r = 1.2. d=1. 2r-d=0.2.
    # Move x_i to 0.1 (towards x_j). d=0.9. 2r-d=0.3. V increases.
    # So gradient at x_i=0 should be in direction of x_j? No, gradient points in direction of steepest increase.
    # We want to DECREASE V, so we move AGAINST gradient.
    
    # Let's re-evaluate:
    # term = (2r - d)^2
    # dterm/dx_i = 2(2r - d) * (-1) * d(d)/dx_i
    # d(d)/dx_i = (x_i - x_j)/d
    # dterm/dx_i = -2(2r - d) (x_i - x_j)/d
    
    # So if 2r > d (overlap), (2r-d) is positive.
    # (x_i - x_j) points from j to i.
    # So gradient is in direction -(x_i - x_j), i.e., towards x_j.
    # So to decrease V, we move in direction (x_i - x_j), i.e., away from x_j. Correct.
    
    # Gradient matrix
    # G[i, j] = -2 * (2r - dists[i, j]) * diff[i, j] / dists[i, j]
    # Only non-zero if 2r > dists[i, j]
    
    factor = -2.0 * (2.0 * r - dists)
    factor[dists >= 2.0 * r] = 0.0 # No overlap, no force
    # Also handle dist=0 case (coincident)
    factor[dists < 1e-12] = 0.0 # Avoid numerical issues, handled separately or by random jitter
    
    grad_pairs = np.sum(factor[:, :, np.newaxis] * diff / dists[:, :, np.newaxis], axis=1)
    
    # Wall gradient
    # V_wall = sum (max(0, L - x)^2 + max(0, x - R)^2)
    # dV/dx = 2(L - x) if x < L, 2(x - R) if x > R, 0 otherwise
    
    L = -0.5 + r
    R = 0.5 - r
    
    # x walls
    mask_left_x = pos[:, 0] < L
    mask_right_x = pos[:, 0] > R
    grad[:, 0] += 2.0 * (L - pos[:, 0]) * mask_left_x
    grad[:, 0] += 2.0 * (pos[:, 0] - R) * mask_right_x
    
    # y walls
    mask_left_y = pos[:, 1] < L
    mask_right_y = pos[:, 1] > R
    grad[:, 1] += 2.0 * (L - pos[:, 1]) * mask_left_y
    grad[:, 1] += 2.0 * (pos[:, 1] - R) * mask_right_y
    
    return grad

def _project(pos, r):
    """Project positions onto the feasible region defined by walls for radius r."""
    pos = np.asarray(pos).copy()
    L = -0.5 + r
    R = 0.5 - r
    pos[:, 0] = np.clip(pos[:, 0], L, R)
    pos[:, 1] = np.clip(pos[:, 1], L, R)
    return pos

def _relax(pos, r, max_iters=1000, tol=1e-8):
    """
    Relax the configuration using gradient descent on violation.
    Returns (pos, final_violation)
    """
    pos = _project(pos, r)
    lr = 0.1
    best_pos = pos.copy()
    best_viol = _violation(pos, r)
    
    for i in range(max_iters):
        grad = _grad_violation(pos, r)
        # Simple gradient descent with projection
        pos_new = pos - lr * grad
        pos_new = _project(pos_new, r)
        
        viol_new = _violation(pos_new, r)
        
        if viol_new < best_viol:
            best_viol = viol_new
            best_pos = pos_new.copy()
            
        if abs(viol_new - best_viol) < tol and best_viol < 1e-6:
            break
            
        pos = pos_new
        
        # Adaptive learning rate
        if i % 50 == 0:
            lr *= 0.95
            if lr < 1e-5:
                break
                
    return best_pos, best_viol

def _sa_search(start_pos, target_r, time_limit, initial_temp=0.01, cooling=0.95):
    """
    Simulated Annealing to find a valid configuration for target_r.
    """
    pos = _project(start_pos, target_r)
    current_viol = _violation(pos, target_r)
    best_pos = pos.copy()
    best_viol = current_viol
    
    temp = initial_temp
    step_size = 0.01
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Generate a random move
        # Pick a random circle and move it slightly
        idx = random.randint(0, N - 1)
        move = np.random.normal(0, step_size, size=2)
        
        pos_new = pos.copy()
        pos_new[idx] += move
        pos_new = _project(pos_new, target_r)
        
        new_viol = _violation(pos_new, target_r)
        
        delta = new_viol - current_viol
        if delta < 0 or random.random() < np.exp(-delta / temp):
            pos = pos_new
            current_viol = new_viol
            
            if current_viol < best_viol:
                best_viol = current_viol
                best_pos = pos.copy()
                
        # Cooling
        temp *= cooling
        if temp < 1e-5:
            break
            
    return best_pos, best_viol

def _gradient_polish(pos, r, max_iters=10000):
    """
    Final polish using L-BFGS-B on the violation function.
    """
    from scipy.optimize import minimize
    
    def obj(x):
        p = x.reshape(N, 2)
        return _violation(p, r)
    
    def grad(x):
        p = x.reshape(N, 2)
        g = _grad_violation(p, r)
        return g.flatten()
    
    bounds = [(-0.5 + r, 0.5 - r)] * (2 * N)
    
    try:
        res = minimize(obj, pos.flatten(), jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < _violation(pos, r):
            return res.x.reshape(N, 2), res.fun
    except Exception:
        pass
        
    return pos, _violation(pos, r)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial configurations
    configs = []
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    configs.append(grid[:N].copy())
    
    # 2. Hexagonal
    # Hexagonal packing is denser.
    # For N=50, we can try to arrange in hexagonal rows.
    # Row lengths: 5, 6, 5, 6, 5, 6, 5, 6, 5, 5? Sum=50?
    # 5+6+5+6+5+6+5+6+5+5 = 54. Too many.
    # 5+6+5+6+5+6+5+6+5 = 49. Add 1 to first row? 6+6+5+6+5+6+5+6+5 = 50.
    rows = [6, 6, 5, 6, 5, 6, 5, 6, 5]
    hex_pos = []
    dx = 1.0 # spacing
    dy = np.sqrt(3) / 2 * dx
    for i, length in enumerate(rows):
        y = (i - len(rows) / 2) * dy
        x_offset = (i % 2) * dx / 2
        for j in range(length):
            x = (j - (length - 1) / 2) * dx + x_offset
            hex_pos.append([x, y])
    hex_pos = np.array(hex_pos[:N])
    # Scale to fit in [-0.5, 0.5]
    # Current extent is approx 6 * 1.0 = 6 in x, 8 * 0.866 = 6.9 in y.
    # We need to scale down.
    scale = 0.9 / max(np.max(np.abs(hex_pos)))
    hex_pos *= scale
    configs.append(hex_pos)
    
    # 3. Random
    random.seed(42)
    rand_pos = np.random.uniform(-0.5, 0.5, size=(N, 2))
    configs.append(rand_pos)
    
    best_r = 0.0
    best_pos = configs[0]
    
    # Binary search on r
    # Lower bound: current best radius from initial configs
    for cfg in configs:
        r_cfg = _compute_radius(cfg)
        if r_cfg > best_r:
            best_r = r_cfg
            best_pos = cfg
    
    r_low = best_r
    r_high = KNOWN_BEST_R * 1.01 # Slightly above known best to allow discovery
    
    # We want to find the maximum r such that a valid configuration exists.
    # We will iterate, trying to increase r.
    
    # Time management
    # We have ~145 seconds.
    # We'll spend time on a few promising r values.
    
    # Strategy:
    # 1. Try to improve current best_r by small increments.
    # 2. If stuck, try a larger jump with SA.
    
    current_r = r_low
    delta_r = 0.0005
    
    while time.time() < deadline - 10: # Leave 10s for final polish
        r_try = current_r + delta_r
        if r_try > r_high:
            r_try = r_high
            
        # Check if r_try is feasible
        # Use relaxation from best_pos
        pos_relaxed, viol = _relax(best_pos, r_try, max_iters=500)
        
        if viol < 1e-6:
            # Feasible
            current_r = r_try
            best_r = current_r
            best_pos = pos_relaxed
            delta_r = min(delta_r * 1.1, 0.002)
        else:
            # Not feasible with simple relaxation.
            # Try SA for a short time
            time_for_sa = 2.0
            pos_sa, viol_sa = _sa_search(best_pos, r_try, time_for_sa)
            
            if viol_sa < 1e-6:
                current_r = r_try
                best_r = current_r
                best_pos = pos_sa
                delta_r = min(delta_r * 1.05, 0.002)
            else:
                # Failed to increase r.
                # Maybe reduce delta_r or try a different start?
                delta_r *= 0.8
                if delta_r < 1e-5:
                    break
                # Also, if we've been stuck for a while, try to optimize the current best_r
                # to get a better starting point?
                # Or just break if we're close to known best.
                if current_r > KNOWN_BEST_R * 0.99:
                     # We are close to the known best. Try to squeeze out more with SA on current_r?
                     # Or just stop.
                     break

    # Final polish on the best found configuration
    if time.time() < deadline - 5:
        # Use gradient polish on the best_pos with radius best_r
        # Actually, we want to maximize radius.
        # So we should polish the position for the best_r we found.
        # Then check if we can go higher.
        
        # Let's do a final binary search step with more time.
        # If we have time, try to push r a bit more.
        
        # Polish the current best_pos at best_r
        pos_polished, _ = _gradient_polish(best_pos, best_r, max_iters=5000)
        # Check if valid
        if _violation(pos_polished, best_r) < 1e-6:
            best_pos = pos_polished
            
        # Try to increase r one last time with the polished position
        r_try = best_r + 0.001
        if r_try < r_high and time.time() < deadline - 2:
            pos_relaxed, viol = _relax(best_pos, r_try, max_iters=2000)
            if viol < 1e-6:
                best_r = r_try
                best_pos = pos_relaxed
                # Polish again
                pos_polished, _ = _gradient_polish(best_pos, best_r, max_iters=2000)
                if _violation(pos_polished, best_r) < 1e-6:
                    best_pos = pos_polished

    return best_pos

# Test run
if __name__ == "__main__":
    pos = construct()
    r = _compute_radius(pos)
    print(f"Best radius: {r}")
    print(f"Known best: {KNOWN_BEST_R}")
    print(f"Improvement: {r / KNOWN_BEST_R}")
