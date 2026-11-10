import numpy as np
import time
import sys
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
TIME_LIMIT = 145.0  # Seconds, leaving margin before 150s kill
START_TIME = time.time()

def _check_time():
    return time.time() - START_TIME

def _compute_radius(centres):
    """Compute the packing radius for a set of centres.
    Returns r if valid, 0.0 if invalid.
    """
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    
    # Distance to walls
    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    wall_dists = np.min(np.stack([0.5 - dx, 0.5 - dy], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0

    # Minimum pairwise distance
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)

    return min(min_dist / 2.0, min_wall)

def _generate_initial_config():
    """Generate a high-quality hexagonal initial configuration."""
    w, h = 20, 10
    total = w * h
    
    # Hexagonal spacing: dx = 1/(w-1), dy = dx * sqrt(3)/2
    dx = 1.0 / (w - 1)
    dy = dx * np.sqrt(3) / 2.0
    
    # Total height in hex packing
    total_h = (h - 1) * dy
    
    # Center vertically
    y_start = -total_h / 2.0
    
    grid = np.zeros((total, 2))
    idx = 0
    for row in range(h):
        y = y_start + row * dy
        for col in range(w):
            x = -0.5 + col * dx
            if row % 2 == 1:
                x += dx / 2.0
            grid[idx, 0] = x
            grid[idx, 1] = y
            idx += 1
    
    # Check bounds and scale if needed
    max_x = np.max(np.abs(grid[:, 0]))
    max_y = np.max(np.abs(grid[:, 1]))
    if max_x > 0.499 or max_y > 0.499:
        scale_down = 0.499 / max(max_x, max_y)
        grid *= scale_down
    
    return grid

def _energy_and_grad(centres, t, beta=50.0):
    """
    Compute the soft-min energy and gradient.
    We want to maximize r = min(min_pair_dist/2, min_wall_dist).
    Using epigraph: maximize t subject to all constraints >= t.
    Soft formulation: minimize -t + (1/beta) * sum(log(sum(exp(-beta*(constraint_j - t))))
    where constraint_j are all pairwise distances/2 and wall distances.
    """
    centres = centres.reshape(N, 2)
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centres[:, 0]), 0.5 - np.abs(centres[:, 1]))
    
    # Pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # All constraints: wall_dists (N values) + pair_dists (N*(N-1)/2 values)
    # For efficiency, we'll compute the soft-min over all constraints
    all_constraints = np.concatenate([wall_dists, pair_dists.ravel()])
    
    # Soft-min: -logsumexp(-beta * (constraints - t)) / beta
    # We want to minimize: -t - (1/beta) * log(sum(exp(-beta*(c_j - t))))
    # = -t - (1/beta) * log(sum(exp(-beta*c_j) * exp(beta*t)))
    # = -t - (1/beta) * (beta*t + log(sum(exp(-beta*c_j))))
    # = -t - t - (1/beta) * log(sum(exp(-beta*c_j)))
    # = -2t - (1/beta) * logsumexp(-beta * all_constraints)
    
    # Actually, let me be more careful.
    # We want: f(t, x) = -t + (1/beta) * log(sum_j exp(-beta*(c_j(x) - t)))
    #          = -t + (1/beta) * log(sum_j exp(-beta*c_j(x)) * exp(beta*t))
    #          = -t + (1/beta) * (beta*t + log(sum_j exp(-beta*c_j(x))))
    #          = -t + t + (1/beta) * log(sum_j exp(-beta*c_j(x)))
    #          = (1/beta) * logsumexp(-beta * c_j(x))
    
    # So the objective is just (1/beta) * logsumexp(-beta * all_constraints)
    # And we minimize this over x, with t being implicitly determined.
    # But we also need to track t. Actually, at optimum, t = min_j c_j(x).
    # The soft objective approximates -min_j c_j(x) when beta is large.
    # Wait, let me reconsider.
    
    # We want to MAXIMIZE r = min_j c_j(x).
    # Equivalent to MINIMIZING -r = -min_j c_j(x).
    # Soft approximation: -min_j c_j(x) ≈ -(1/beta) * log(sum_j exp(-beta*c_j(x)))
    # = (1/beta) * log(sum_j exp(-beta*c_j(x))) ... no that's wrong sign.
    
    # Let's use: -min_j c_j ≈ -(1/beta)*logsumexp(-beta*c_j) + (1/beta)*log(n)
    # Actually: min_j c_j = -(1/beta) * log(sum_j exp(-beta*c_j)) + (1/beta)*log(n) approximately
    # So -min_j c_j = (1/beta) * log(sum_j exp(-beta*c_j)) - (1/beta)*log(n)
    
    # To maximize min_j c_j, we minimize: (1/beta) * logsumexp(-beta * c_j)
    # (the -log(n)/beta constant doesn't matter)
    
    # So objective = (1/beta) * logsumexp(-beta * all_constraints)
    # Gradient w.r.t. x: (1/beta) * sum_j exp(-beta*c_j) * (-beta * grad c_j) / sum_j exp(-beta*c_j)
    #                  = -sum_j softmax_j * grad c_j
    # where softmax_j = exp(-beta*c_j) / sum_k exp(-beta*c_k)
    
    # This is a weighted sum of gradients, with weights proportional to how "tight" each constraint is.
    
    # Compute logsumexp safely
    neg_beta_c = -beta * all_constraints
    max_val = np.max(neg_beta_c)
    logsumexp_val = max_val + np.log(np.sum(np.exp(neg_beta_c - max_val)))
    objective = logsumexp_val / beta
    
    # Softmax weights
    weights = np.exp(neg_beta_c - max_val)
    weights /= np.sum(weights)
    
    # Now compute gradient
    grad = np.zeros_like(centres)
    
    # Wall gradient contribution
    # c_wall_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # d(c_wall_i)/dx_i = -sign(x_i) if 0.5-|x_i| < 0.5-|y_i|, else 0
    # d(c_wall_i)/dy_i = -sign(y_i) if 0.5-|y_i| < 0.5-|x_i|, else 0
    
    wall_weights = weights[:N]
    for i in range(N):
        wx = 0.5 - abs(centres[i, 0])
        wy = 0.5 - abs(centres[i, 1])
        if wx < wy:
            grad[i, 0] -= wall_weights[i] * np.sign(centres[i, 0])
        else:
            grad[i, 1] -= wall_weights[i] * np.sign(centres[i, 1])
    
    # Pairwise gradient contribution
    # c_pair_ij = |x_i - x_j| / 2
    # d(c_pair_ij)/dx_i = (x_i - x_j) / |x_i - x_j| * (1/2) ... wait
    # Actually c_pair_ij = dist(i,j) / 2
    # d(dist)/dx_i = (x_i - x_j) / dist
    # So d(c_pair_ij)/dx_i = (x_i - x_j) / (2 * dist)
    
    pair_weights = weights[N:]
    # Reshape to get pair indices
    idx_i, idx_j = np.triu_indices(N, k=1)
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        w = pair_weights[k]
        if w < 1e-12:
            continue
        dx = centres[i, 0] - centres[j, 0]
        dy = centres[i, 1] - centres[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 1e-12:
            continue
        grad_factor = w / (2 * dist)
        grad[i, 0] -= grad_factor * dx
        grad[i, 1] -= grad_factor * dy
        grad[j, 0] += grad_factor * dx
        grad[j, 1] += grad_factor * dy
    
    return objective, grad

def _local_optimize(centres, time_limit):
    """
    Optimize using L-BFGS-B with soft-min energy, followed by a polish pass.
    """
    current_r = _compute_radius(centres)
    if current_r == 0.0:
        return centres, 0.0
        
    best_centres = centres.copy()
    best_r = current_r
    
    deadline = START_TIME + time_limit
    
    # Try multiple beta values, starting moderate and increasing
    betas = [30.0, 80.0, 150.0, 300.0]
    
    for beta in betas:
        if time.time() >= deadline - 10:
            break
            
        def obj_and_grad(x):
            if time.time() >= deadline - 3:
                return 1e10, np.zeros_like(x)
            c = x.reshape(N, 2)
            c = np.clip(c, -0.4999, 0.4999)
            val, grad = _energy_and_grad(c, 0, beta)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad,
                best_centres.ravel(),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'maxfun': 800,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 50
                }
            )
            
            new_centres = result.x.reshape(N, 2)
            new_centres = np.clip(new_centres, -0.5, 0.5)
            new_r = _compute_radius(new_centres)
            
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres
        except Exception:
            pass
    
    # Final polish: direct coordinate descent on the exact min function
    if time.time() < deadline - 2:
        best_centres = _polish(best_centres, deadline)
        final_r = _compute_radius(best_centres)
        if final_r > best_r:
            best_r = final_r
    
    return best_centres, best_r

def _polish(centres, deadline):
    """
    Direct coordinate descent polish on the exact radius function.
    For each point, try small perturbations to increase the minimum constraint.
    """
    centres = centres.copy()
    best_r = _compute_radius(centres)
    if best_r == 0.0:
        return centres
    
    step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        if time.time() >= deadline - 1:
            break
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                if time.time() >= deadline - 1:
                    break
                for dx in [-step, step]:
                    for dy in [-step, step]:
                        new_c = centres.copy()
                        new_c[i, 0] += dx
                        new_c[i, 1] += dy
                        if -0.5 <= new_c[i, 0] <= 0.5 and -0.5 <= new_c[i, 1] <= 0.5:
                            r = _compute_radius(new_c)
                            if r > best_r + 1e-15:
                                centres = new_c
                                best_r = r
                                improved = True
                                break
                    if improved:
                        break
    
    return centres

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - (start_time - START_TIME)
    
    # 1. Generate Initial Configuration
    centres = _generate_initial_config()
    current_r = _compute_radius(centres)
    
    if current_r == 0.0:
        # Emergency fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # 2. Local Optimization
    if time_budget > 10:
        centres, current_r = _local_optimize(centres, time_budget - 5)
        
    # 3. Final Validation
    final_r = _compute_radius(centres)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return centres
