"""Improved packing: gradient-based optimization with force-directed initialization."""
import numpy as np
import time
import math

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def _compute_radius(centres):
    """Compute the radius for a set of centres (must be inside [-0.5,0.5]^2)."""
    if len(centres) == 0:
        return 0.0
    # Boundary constraint
    dist_to_boundary = np.min(0.5 - np.abs(centres), axis=1)
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(dist_to_boundary))


def _soft_min(values, temperature=0.01):
    """Soft minimum using log-sum-exp trick."""
    # Use a large negative multiplier to approximate min
    if len(values) == 0:
        return 0.0
    m = np.min(values)
    if m <= 0:
        return m
    # Soft min approximation
    log_sum = np.log(np.sum(np.exp(-values / temperature)))
    return m - temperature * log_sum


def _objective_and_grad(centres, beta=20.0):
    """
    Objective: maximize the minimum of:
      - min_{i<j} |c_i - c_j| / 2
      - min_i min_k (0.5 - |c_ik|)
    
    We minimize the negative of this, using a smooth approximation.
    
    Returns (objective, gradient)
    """
    n = len(centres)
    
    # Compute all pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Boundary distances
    bound_dists = np.min(0.5 - np.abs(centres), axis=1)
    
    # Collect all "gap" values: pairwise distances / 2 and boundary distances
    # We want to maximize the minimum of these
    # Use a soft-min approximation: -log(sum(exp(-x/beta))) * beta approximates min(x)
    
    # Pairwise gaps
    pair_gaps = dists / 2.0
    # Boundary gaps
    bound_gaps = bound_dists
    
    # Soft min of all gaps
    all_gaps = np.concatenate([pair_gaps.flatten(), bound_gaps])
    
    # Soft min: m - beta * log(sum(exp(-(x-m)/beta)))
    m = np.min(all_gaps)
    if m <= 0:
        # If any gap is non-positive, we're infeasible
        obj = -m * 100  # large penalty
        # Gradient: push points apart / away from boundary
        grad = np.zeros_like(centres)
        
        # Boundary gradient
        for i in range(n):
            for k in range(2):
                if centres[i, k] > 0:
                    grad[i, k] -= 100
                else:
                    grad[i, k] += 100
        
        # Pairwise: push apart
        for i in range(n):
            for j in range(i + 1, n):
                d = centres[i] - centres[j]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    grad[i] += d / dist * 100
                    grad[j] -= d / dist * 100
        return obj, grad
    
    # Soft min computation
    exp_terms = np.exp(-(all_gaps - m) / beta)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - beta * log_sum
    
    obj = -soft_min_val
    
    # Gradient of soft min
    # d(soft_min)/dx_i = -beta * sum_j (exp(-(x_j-m)/beta) / sum_k exp(-(x_k-m)/beta)) * dx_j/dx_i
    # For pairwise distance |c_i - c_j|/2:
    #   d/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    #   d/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    # For boundary dist 0.5 - |c_ik|:
    #   d/d(c_ik) = -sign(c_ik)
    
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(centres)
    
    # Pairwise contributions
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[idx]
            d = centres[i] - centres[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                grad[i] -= w * d / (2 * dist) * beta  # derivative of -soft_min
                grad[j] += w * d / (2 * dist) * beta
            idx += 1
    
    # Boundary contributions
    for i in range(n):
        for k in range(2):
            w = weights[idx]
            if centres[i, k] > 0:
                grad[i, k] += w * beta  # d(0.5 - |x|)/dx = -1, so -soft_min gradient = +w*beta
            elif centres[i, k] < 0:
                grad[i, k] -= w * beta
            idx += 1
    
    return obj, grad


def _generate_hex_initial(n=N):
    """Generate initial configuration on a hexagonal grid, then optimize."""
    # Start with a rough hexagonal packing
    # For 200 circles in unit square, radius ~0.0366
    # Hex packing: rows offset by half spacing
    
    # Estimate spacing from area: 200 * pi * r^2 <= 1
    # r ~ 0.0398 max, but hex packing is denser
    # Let's start with a grid and refine
    
    # Use a grid of about 14x15 = 210 points, then trim
    n_rows = 14
    n_cols = 15
    x = np.linspace(-0.48, 0.48, n_cols)
    y = np.linspace(-0.48, 0.48, n_rows)
    
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.flatten(), Y.flatten()])
    
    # Apply hexagonal offset
    for i in range(len(points)):
        row = i // n_cols
        if row % 2 == 1:
            points[i, 0] += (x[1] - x[0]) / 2
    
    # Clip to square
    points = np.clip(points, -0.48, 0.48)
    
    # Select n points: use farthest point sampling
    selected = []
    remaining = points.copy()
    
    # Start with center
    center = np.array([0.0, 0.0])
    idx = np.argmin(np.sum((remaining - center) ** 2, axis=1))
    selected.append(remaining[idx])
    remaining = np.delete(remaining, idx, axis=0)
    
    for _ in range(n - 1):
        if len(remaining) == 0:
            break
        # For each remaining point, compute min distance to selected
        min_dists = []
        for pt in remaining:
            d = np.min(np.linalg.norm(pt - selected, axis=1))
            # Also consider boundary
            bd = np.min(0.5 - np.abs(pt))
            min_dists.append(min(d, bd))
        
        idx = np.argmax(min_dists)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    
    return np.array(selected[:n])


def _generate_grid_initial(n=N):
    """Generate initial configuration on a regular grid."""
    # Use a square grid
    n_side = int(np.ceil(np.sqrt(n)))
    # We need n points, so use n_side x n_side and trim
    x = np.linspace(-0.48, 0.48, n_side)
    y = np.linspace(-0.48, 0.48, n_side)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.flatten(), Y.flatten()])
    
    # Randomly shuffle and take first n
    np.random.shuffle(points)
    return points[:n]


def _run_gradient_optimization(centres, max_iter=2000, beta_start=5.0, beta_end=1.0):
    """Run gradient-based optimization."""
    best = centres.copy()
    best_r = _compute_radius(best)
    
    n = len(centres)
    beta = beta_start
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Decrease beta over time
        beta = beta_start * (beta_end / beta_start) ** (it / max_iter)
        
        obj, grad = _objective_and_grad(centres, beta=beta)
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-8:
            break
        
        # Simple gradient descent with adaptive step size
        step = 0.01
        for _ in range(10):
            new_centres = centres - step * grad
            # Clip to square
            new_centres = np.clip(new_centres, -0.49, 0.49)
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r:
                best = new_centres.copy()
                best_r = new_r
                centres = new_centres
                break
            else:
                step *= 0.5
        
        # Occasionally do a local refinement
        if it % 50 == 0 and it > 0:
            # Try small random perturbations
            for _ in range(20):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                i = np.random.randint(n)
                orig = centres[i].copy()
                centres[i] = orig + np.random.randn(2) * 0.005
                centres[i] = np.clip(centres[i], -0.49, 0.49)
                r = _compute_radius(centres)
                if r > best_r:
                    best = centres.copy()
                    best_r = r
                else:
                    centres[i] = orig
    
    return best, best_r


def _polish(centres, n_iter=500):
    """Local polish: coordinate ascent on the minimum gap."""
    best = centres.copy()
    best_r = _compute_radius(best)
    n = len(centres)
    
    for it in range(n_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Find the bottleneck: closest pair or boundary violation
        # Compute all pairwise distances
        diff = best[:, np.newaxis, :] - best[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pair
        min_dist = np.min(dists)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Boundary distance
        bound_dists = np.min(0.5 - np.abs(best), axis=1)
        min_bound = np.min(bound_dists)
        
        # If boundary is the bottleneck, move that point inward
        if min_bound < min_dist / 2:
            idx = np.argmin(bound_dists)
            # Move point toward center
            direction = -np.sign(best[idx])
            best[idx] += direction * 0.001
            best[idx] = np.clip(best[idx], -0.49, 0.49)
        else:
            # Move the closest pair apart
            d = best[i] - best[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                direction = d / dist
                best[i] += direction * 0.001
                best[j] -= direction * 0.001
                best[i] = np.clip(best[i], -0.49, 0.49)
                best[j] = np.clip(best[j], -0.49, 0.49)
        
        new_r = _compute_radius(best)
        if new_r > best_r:
            best_r = new_r
        else:
            # Revert
            if min_bound < min_dist / 2:
                direction = -np.sign(best[idx])
                best[idx] -= direction * 0.001
                best[idx] = np.clip(best[idx], -0.49, 0.49)
            else:
                d = best[i] - best[j]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    direction = d / dist
                    best[i] -= direction * 0.001
                    best[j] += direction * 0.001
                    best[i] = np.clip(best[i], -0.49, 0.49)
                    best[j] = np.clip(best[j], -0.49, 0.49)
    
    return best, best_r


def construct():
    """Construct 200 circle centres in [-0.5, 0.5]^2 maximizing minimum radius."""
    t0 = time.time()
    
    # Try multiple initializations and keep the best
    best_centres = None
    best_r = 0.0
    
    # Initialization 1: Hexagonal grid
    if time.time() - START_TIME < TIME_LIMIT - 30:
        centres = _generate_hex_initial(N)
        centres, r = _run_gradient_optimization(centres, max_iter=1000, beta_start=5.0, beta_end=1.0)
        if r > best_r:
            best_r = r
            best_centres = centres
    
    # Initialization 2: Grid
    if time.time() - START_TIME < TIME_LIMIT - 30:
        centres = _generate_grid_initial(N)
        centres, r = _run_gradient_optimization(centres, max_iter=1000, beta_start=5.0, beta_end=1.0)
        if r > best_r:
            best_r = r
            best_centres = centres
    
    # Final polish
    if best_centres is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        best_centres, best_r = _polish(best_centres, n_iter=1000)
    
    if best_centres is None:
        # Fallback: simple grid
        n_side = int(np.ceil(np.sqrt(N)))
        x = np.linspace(-0.48, 0.48, n_side)
        y = np.linspace(-0.48, 0.48, n_side)
        X, Y = np.meshgrid(x, y)
        best_centres = np.column_stack([X.flatten(), Y.flatten()])[:N]
    
    # Ensure valid
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
