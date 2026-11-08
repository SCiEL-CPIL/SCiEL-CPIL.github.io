import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 145.0  # Safety margin before 150s limit
SOFTMIN_TAU = 0.02  # Temperature for soft-min smoothing

def _softmin_radii(x, y, W, H, tau=SOFTMIN_TAU):
    """
    Compute smoothed radii using soft-min for differentiability.
    x, y: arrays of shape (N,)
    W, H: rectangle dimensions
    tau: soft-min temperature
    Returns: array of shape (N,) with soft-min radii
    """
    n = len(x)
    
    # Pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Distances to other circles (half distance)
    # Set diagonal to infinity so it doesn't affect the min
    r_other = dist / 2.0
    np.fill_diagonal(r_other, np.inf)
    
    # Wall distances
    # Shape (N, 4)
    wall_dists = np.column_stack([
        x,          # left
        W - x,      # right
        y,          # bottom
        H - y       # top
    ])
    
    # Combine all constraints for each circle
    # Shape (N, N-1 + 4)
    constraints = np.hstack([r_other, wall_dists])
    
    # Soft-min calculation
    # softmin = -tau * log(sum(exp(-constraints/tau)))
    # To avoid overflow/underflow, we can use the log-sum-exp trick
    # log(sum(exp(z))) = max(z) + log(sum(exp(z - max(z))))
    # Here z = -constraints/tau
    z = -constraints / tau
    z_max = np.max(z, axis=1, keepdims=True)
    # Handle case where all are inf (shouldn't happen for valid boxes)
    z_safe = np.where(np.isfinite(z_max), z - z_max, 0)
    sum_exp = np.sum(np.exp(z_safe), axis=1)
    
    # If sum_exp is 0, it means all constraints were -inf? No, z is negative or -inf.
    # If constraints are large, z is close to 0.
    # If constraints are 0, z is 0.
    # The term is -tau * (z_max + log(sum_exp))
    soft_r = -tau * (np.squeeze(z_max) + np.log(sum_exp + 1e-15))
    
    # Clamp to 0 just in case of numerical issues
    soft_r = np.maximum(soft_r, 0.0)
    
    return soft_r

def _hard_radii(x, y, W, H):
    """
    Compute true max radii (hard min) for validation and final score.
    """
    n = len(x)
    r = np.minimum(x, W - x)
    r = np.minimum(r, y)
    r = np.minimum(r, H - y)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    half_dist = dist / 2.0
    np.fill_diagonal(half_dist, np.inf)
    
    r_other = np.min(half_dist, axis=1)
    r = np.minimum(r, r_other)
    
    return r

def _objective(v, W, H, tau):
    """
    Objective function for minimization (maximize sum of soft radii).
    """
    n = N
    x = v[:n]
    y = v[n:2*n]
    r = _softmin_radii(x, y, W, H, tau)
    return -np.sum(r)

def _objective_grad(v, W, H, tau):
    """
    Numerical gradient for L-BFGS-B.
    Analytical gradient is complex due to soft-min.
    Numerical gradient is robust enough for this scale.
    """
    eps = 1e-7
    n = N
    grad = np.zeros_like(v)
    for i in range(len(v)):
        v_plus = v.copy()
        v_minus = v.copy()
        v_plus[i] += eps
        v_minus[i] -= eps
        grad[i] = (_objective(v_plus, W, H, tau) - _objective(v_minus, W, H, tau)) / (2 * eps)
    return grad

def _optimize_aspect(W, H, start_x, start_y, tau, max_iter=1000):
    """
    Optimize positions for a given aspect ratio.
    """
    n = N
    x0 = np.concatenate([start_x, start_y])
    bounds = [(0.0, W)] * n + [(0.0, H)] * n
    
    try:
        res = minimize(
            _objective, x0, args=(W, H, tau),
            jac=_objective_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
        )
    except Exception:
        return None
    
    x_opt = res.x[:n]
    y_opt = res.x[n:2*n]
    
    # Compute hard radii for the optimized positions
    r_hard = _hard_radii(x_opt, y_opt, W, H)
    
    # Validation
    if np.any(r_hard < 1e-6):
        return None
        
    # Check overlaps with hard radii
    dx = x_opt[:, np.newaxis] - x_opt[np.newaxis, :]
    dy = y_opt[:, np.newaxis] - y_opt[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r_hard[:, np.newaxis] + r_hard[np.newaxis, :]
    # Diagonal should be 0 distance, 2*r sum. We check off-diagonal.
    mask = ~np.eye(n, dtype=bool)
    if np.any(dist[mask] < sum_r[mask] - 1e-5):
        return None
        
    # Check boundaries
    if np.any(x_opt < r_hard - 1e-5) or np.any(x_opt > W - r_hard + 1e-5):
        return None
    if np.any(y_opt < r_hard - 1e-5) or np.any(y_opt > H - r_hard + 1e-5):
        return None

    return x_opt, y_opt, r_hard, np.sum(r_hard)

def _initial_grid(W, H, cols, rows, seed=None):
    """Generate grid initialization."""
    n = N
    xs = np.zeros(n)
    ys = np.zeros(n)
    k = 0
    rng = np.random.RandomState(seed) if seed is not None else np.random
    for i in range(cols):
        for j in range(rows):
            if k >= n:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            # Add small jitter to break symmetry
            cx += (rng.rand() - 0.5) * 0.1 * W / cols
            cy += (rng.rand() - 0.5) * 0.1 * H / rows
            xs[k] = np.clip(cx, 0.01, W - 0.01)
            ys[k] = np.clip(cy, 0.01, H - 0.01)
            k += 1
    return xs, ys

def _initial_random(W, H, seed=None):
    """Random initialization."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    xs = rng.uniform(0.1, W - 0.1, N)
    ys = rng.uniform(0.1, H - 0.1, N)
    return xs, ys

def construct():
    t0 = time.time()
    best_val = -1
    best_result = None
    
    # Aspect ratios W such that W + H = 2.
    # By symmetry, W=1 is the square. We explore W in [0.7, 1.3].
    aspect_ratios = np.linspace(0.7, 1.3, 13)
    
    # Grid configurations for initialization
    grid_configs = [
        (4, 6), (5, 5), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4)
    ]
    
    restarts_per_aspect = 3
    tau = 0.02
    
    for ai, W in enumerate(aspect_ratios):
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        for ri in range(restarts_per_aspect):
            # Check deadline
            elapsed = time.time() - t0
            if elapsed > DEADLINE - 5.0:
                break
            
            # Choose initialization
            if ri == 0:
                cols, rows = grid_configs[ai % len(grid_configs)]
                start_x, start_y = _initial_grid(W, H, cols, rows, seed=ai)
            elif ri == 1:
                cols, rows = grid_configs[(ai + 1) % len(grid_configs)]
                start_x, start_y = _initial_grid(W, H, cols, rows, seed=ai + 100)
            else:
                start_x, start_y = _initial_random(W, H, seed=ai * 10 + ri)
            
            result = _optimize_aspect(W, H, start_x, start_y, tau, max_iter=500)
            
            if result is not None:
                x, y, r, val = result
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack((x, y, r))
        
        # Check deadline after each aspect ratio
        if time.time() - t0 > DEADLINE - 5.0:
            break
    
    # If we have time, refine the best solution found with a smaller tau (sharper min)
    if best_result is not None and time.time() - t0 < DEADLINE - 10.0:
        # Try to find which aspect ratio was best? We didn't store it.
        # Just re-optimize the best positions found? 
        # Actually, we can just run a few more restarts on the most promising aspect ratios.
        # Since we don't know which was best, let's just do random restarts on W=1.0 and W=1.1
        for W in [1.0, 1.1, 0.9, 1.2]:
            H = 2.0 - W
            for seed in range(5):
                if time.time() - t0 > DEADLINE - 5.0:
                    break
                start_x, start_y = _initial_random(W, H, seed=seed + 500)
                result = _optimize_aspect(W, H, start_x, start_y, tau, max_iter=300)
                if result is not None:
                    x, y, r, val = result
                    if val > best_val:
                        best_val = val
                        best_result = np.column_stack((x, y, r))

    if best_result is not None:
        return best_result
    
    # Fallback: simple grid in 1x1 square
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])

if __name__ == "__main__":
    res = construct()
    print(res.shape)
