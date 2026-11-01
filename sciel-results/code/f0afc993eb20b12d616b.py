"""
Strategy: Binary Search on Radius with Proximal Gradient Descent on Smooth Violation.

This approach fundamentally changes the optimization paradigm from the current program:
1. Current: SLSQP maximizes r subject to hard inequality constraints. This is slow 
   (matrix factorizations per step) and prone to getting stuck in local minima where 
   the constraint Jacobian is ill-conditioned.
2. New: Binary search on target radius R. For each R, we minimize a smooth, 
   differentiable violation potential V(cents) = sum of squared violations.
   - Pairwise: max(0, 2R - ||c_i - c_j||)^2
   - Wall:     max(0, R - dist_to_wall)^2
   We use proximal gradient descent (L-BFGS-B style but manual for speed) or simple 
   projected gradient descent with vectorized NumPy operations.
   
   The smoothness of the squared violation allows the gradient to guide circles 
   apart smoothly, avoiding the "sticking" behavior of hard constraints.
   
   Vectorization allows ~1000-5000 steps in 120s, vastly exceeding SLSQP's ~100 steps.
"""

import numpy as np
import time

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _score(cents):
    """Compute the packing radius."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _violation_and_grad(cents, R):
    """
    Compute smooth violation potential and its gradient.
    V = sum_{i<j} [max(0, 2R - d_ij)]^2 + sum_i sum_w [max(0, R - w_dist_iw)]^2
    
    Gradient:
    For pair (i,j) with d < 2R:
      grad_i += 2 * (2R - d) * (c_i - c_j) / d
      grad_j += 2 * (2R - d) * (c_j - c_i) / d
    For wall w of circle i with dist < R:
      grad_i += 2 * (R - dist) * (unit vector towards wall)
    """
    n = len(cents)
    grad = np.zeros_like(cents)
    V = 0.0
    
    # Pairwise
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider pairs where dist < 2R
    mask = dists < 2 * R
    # Avoid division by zero
    safe_dists = np.where(mask, dists, 1.0)
    
    # Contribution: 2 * (2R - d) * (c_i - c_j) / d
    # We need to sum over j for each i
    # term[i, j] = 2 * (2R - d_ij) / d_ij * (c_i - c_j)
    # Only non-zero where mask is true
    
    # Efficient calculation:
    # factor[i, j] = 2 * (2R - d_ij) / d_ij if d_ij < 2R else 0
    factor = np.zeros_like(dists)
    factor[mask] = 2.0 * (2.0 * R - dists[mask]) / safe_dists[mask]
    
    # grad_i = sum_j factor[i, j] * (c_i - c_j)
    # = sum_j factor[i, j] * c_i - sum_j factor[i, j] * c_j
    # = c_i * sum_j factor[i, j] - sum_j (factor[i, j] * c_j)
    
    sum_factors = np.sum(factor, axis=1)  # shape (n,)
    weighted_c = factor @ cents  # shape (n, 2), where (factor @ cents)[i] = sum_j factor[i,j]*c_j
    
    grad_pair = cents * sum_factors[:, np.newaxis] - weighted_c
    grad += grad_pair
    
    # Violation value
    viol_vals = np.where(mask, (2.0 * R - dists) ** 2, 0.0)
    V += np.sum(viol_vals)
    
    # Walls
    # Left: dist = 0.5 + x. Violation if 0.5 + x < R => x < R - 0.5
    # Grad x += 2(R - (0.5+x)) * (-1) ? No, dist_to_wall = 0.5 + x. 
    # V = (R - (0.5+x))^2. dV/dx = 2(R - 0.5 - x) * (-1) = 2(0.5 + x - R)
    # Wait, if dist < R, we want to push away from wall. 
    # Wall is at -0.5. Center x. dist = x - (-0.5) = x + 0.5.
    # If x + 0.5 < R, violation. Push x to increase.
    # dV/dx = 2(R - (x+0.5)) * d(R-x-0.5)/dx = 2(R-x-0.5)*(-1) ... 
    # Let's re-derive carefully.
    # V_left = (R - (x+0.5))^2 if x+0.5 < R else 0.
    # dV/dx = 2(R - x - 0.5) * (-1) = 2(x + 0.5 - R).
    # Since x+0.5 < R, this is negative. So grad points left? 
    # No, if we minimize V, we follow -grad. -grad points right (positive). Correct.
    
    # Let's just compute the vector directly.
    # Left wall (x = -0.5): dist = x + 0.5
    dist_l = cents[:, 0] + 0.5
    mask_l = dist_l < R
    if np.any(mask_l):
        # V += (R - dist_l)^2
        V += np.sum((R - dist_l[mask_l]) ** 2)
        # grad_x += 2 * (R - dist_l) * (-1)  -> because d(dist_l)/dx = 1
        # Wait. V = (R - dist)^2. dV/ddist = 2(R-dist). ddist/dx = 1.
        # So dV/dx = 2(R - dist).
        # If dist < R, R-dist > 0. So grad is positive. Push x to the right. Correct.
        grad[mask_l, 0] += 2.0 * (R - dist_l[mask_l])
        
    # Right wall (x = 0.5): dist = 0.5 - x
    dist_r = 0.5 - cents[:, 0]
    mask_r = dist_r < R
    if np.any(mask_r):
        V += np.sum((R - dist_r[mask_r]) ** 2)
        # dV/dx = 2(R - dist_r) * d(0.5-x)/dx = 2(R-dist_r)*(-1)
        grad[mask_r, 0] += -2.0 * (R - dist_r[mask_r])
        
    # Bottom wall (y = -0.5): dist = y + 0.5
    dist_b = cents[:, 1] + 0.5
    mask_b = dist_b < R
    if np.any(mask_b):
        V += np.sum((R - dist_b[mask_b]) ** 2)
        grad[mask_b, 1] += 2.0 * (R - dist_b[mask_b])
        
    # Top wall (y = 0.5): dist = 0.5 - y
    dist_t = 0.5 - cents[:, 1]
    mask_t = dist_t < R
    if np.any(mask_t):
        V += np.sum((R - dist_t[mask_t]) ** 2)
        grad[mask_t, 1] += -2.0 * (R - dist_t[mask_t])
        
    return V, grad

def _project(cents):
    """Project centers into the valid square [-0.5, 0.5]^2."""
    cents[:, 0] = np.clip(cents[:, 0], -0.5, 0.5)
    cents[:, 1] = np.clip(cents[:, 1], -0.5, 0.5)
    return cents

def _try_radius(R, init_cents, time_budget):
    """
    Try to find a feasible configuration for radius R.
    Returns (success, best_cents).
    """
    start_time = time.time()
    cents = init_cents.copy()
    
    # Adaptive step size for gradient descent
    # Use a simple backtracking line search or fixed step with decay
    lr = 0.01
    lr_decay = 0.999
    momentum = 0.9
    vel = np.zeros_like(cents)
    
    best_cents = cents.copy()
    best_V = np.inf
    
    steps = 0
    while time.time() - start_time < time_budget:
        V, grad = _violation_and_grad(cents, R)
        
        if V < best_V:
            best_V = V
            best_cents = cents.copy()
            
        if V < 1e-10:
            return True, cents
            
        # Momentum update
        vel = momentum * vel - lr * grad
        cents = cents + vel
        
        # Project to square
        _project(cents)
        
        # Decay learning rate slightly
        lr *= lr_decay
        steps += 1
        
    return False, best_cents

def _construct_hexagonal():
    """Create a good initial hexagonal packing."""
    best_cents = None
    best_r = 0
    
    for n_rows in range(5, 9):
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        
        if r <= best_r:
            continue
            
        centers = []
        total_y = (n_rows - 1) * np.sqrt(3.0) * r
        y_start = -0.5 + r + total_y / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * np.sqrt(3.0) * r
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = np.sqrt(3.0) * r * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
                
        centers = np.array(centers[:N])
        r_actual = _score(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
            
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)
        
    return best_cents, best_r

def construct():
    start_time = time.time()
    
    # 1. Initial Hexagonal Packing
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # 2. Binary Search on Radius
    # Lower bound: current best r
    # Upper bound: theoretical max ~ 0.08 (for 50 circles, roughly 1/sqrt(50) ~ 0.14, but packing density limits it)
    # Boll et al. found 0.07137...
    # Let's search from r0 to r0 * 1.5
    
    low = best_r
    high = 0.085  # Safe upper bound
    
    # We will perform a binary search. 
    # For each mid-point, we try to find a feasible configuration.
    # If successful, we update low and keep the configuration.
    # If failed, we update high.
    
    # To allow more search time for "hard" radii, we allocate time based on remaining budget.
    
    # Initial refinement of the starting configuration
    # Try to improve r0 slightly
    init_budget = 10.0
    success, improved_cents = _try_radius(best_r * 1.001, best_cents, init_budget)
    if success:
        best_cents = improved_cents
        best_r = _score(best_cents)
        low = best_r

    # Binary search loop
    # We want to find the maximum feasible R.
    # We'll do about 15-20 iterations of binary search.
    
    rng = np.random.default_rng(42)
    
    for i in range(20):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5.0:
            break
            
        mid = (low + high) / 2.0
        
        # Allocate time for this trial.
        # Early trials (coarse) need less time. Later trials (fine) might need more if close to boundary.
        # Let's use a fixed budget per trial, decreasing as we approach deadline.
        trial_budget = min(5.0, remaining * 0.8)
        
        # Start from the best known feasible configuration (which was feasible for 'low', so likely feasible for 'mid' if mid < low, 
        # but we are searching for higher R, so we start from the configuration that achieved 'low')
        # Actually, if we are testing R > low, the previous configuration is NOT feasible.
        # We should start from a configuration that is "close" to feasible for R.
        # A good heuristic: start from the configuration that achieved 'low', but it will have violations for R=mid.
        # The gradient descent will push it out.
        
        success, trial_cents = _try_radius(mid, best_cents, trial_budget)
        
        if success:
            # We found a feasible configuration for R=mid
            low = mid
            best_cents = trial_cents
            # Update best_r with actual score (might be slightly higher than mid due to floating point or slack)
            actual_r = _score(best_cents)
            if actual_r > best_r:
                best_r = actual_r
                low = actual_r
        else:
            # Infeasible
            high = mid
            
    # 3. Final Polish
    # Use remaining time to polish the best configuration found.
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 2.0:
        # Try to squeeze a bit more out of the current best_r
        # Use a very small step size gradient descent to minimize violation for R = best_r * 1.0001
        target_R = best_r * 1.0001
        success, polished_cents = _try_radius(target_R, best_cents, remaining - 1.0)
        if success:
            final_r = _score(polished_cents)
            if final_r > best_r:
                best_r = final_r
                best_cents = polished_cents
        else:
            # Even if not strictly feasible for target_R, the gradient descent might have improved the packing 
            # for the current R by resolving near-collisions.
            # Let's check the score of the polished centers anyway.
            final_r = _score(polished_cents)
            if final_r > best_r:
                best_r = final_r
                best_cents = polished_cents

    return best_cents
