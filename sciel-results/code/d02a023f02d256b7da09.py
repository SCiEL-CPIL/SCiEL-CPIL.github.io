"""
Strategy: Binary search on radius with projected gradient feasibility checks.
Instead of optimizing a soft surrogate that approximates the minimum distance,
we treat the problem as a constraint satisfaction problem:
1. We binary search for the largest radius r such that a feasible configuration exists.
2. For a given r, we check feasibility by minimizing the sum of squared violations
   (overlap between circles and penetration of walls) using L-BFGS-B.
3. If the minimum violation is below a tight tolerance, the radius is feasible.
4. We use a hexagonal grid as the initial guess for each feasibility check.

This is structurally different from the current program because:
- It explicitly targets a specific radius r via binary search.
- The objective is a hard-violation penalty (quadratic) rather than a smooth inverse-distance surrogate.
- It leverages the fact that feasibility is a boolean condition that can be tested efficiently.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
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

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles as a good initial guess."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8]:
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
        # Calculate total vertical span
        # Distance between row centers is r * sqrt(3)
        # Total height = (n_rows - 1) * r * sqrt(3) + 2*r
        # We want to center this in [-0.5, 0.5]
        y_spacing = r * np.sqrt(3.0)
        total_y_span = (n_rows - 1) * y_spacing
        y_start = -0.5 + r + total_y_span / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * y_spacing
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            # Offset odd rows by r
            offset = r if (row_idx % 2 == 1) else 0.0
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

def _violation_loss(x, r):
    """
    Compute the sum of squared violations for a given radius r.
    Violations:
    1. Circle-circle overlap: max(0, 2r - dist_ij)^2
    2. Circle-wall penetration: max(0, r - wall_dist_i)^2
    """
    c = x.reshape(N, 2)
    
    # Wall violations
    wall_x = r - (0.5 - np.abs(c[:, 0]))
    wall_y = r - (0.5 - np.abs(c[:, 1]))
    wall_viol_x = np.maximum(wall_x, 0.0)
    wall_viol_y = np.maximum(wall_y, 0.0)
    loss = np.sum(wall_viol_x**2) + np.sum(wall_viol_y**2)
    
    # Pairwise violations
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Violation is positive if dist < 2r
    pair_viol = np.maximum(2*r - dists, 0.0)
    loss += np.sum(pair_viol**2)
    
    return loss

def _violation_grad(x, r):
    """
    Gradient of the violation loss.
    """
    c = x.reshape(N, 2)
    grad = np.zeros_like(c)
    
    # Wall gradients
    # d/dx (max(0, r - (0.5 - |x|))^2)
    # If x > 0: wall_dist = 0.5 - x. viol = r - 0.5 + x.
    # d/dx (max(0, r - 0.5 + x)^2) = 2 * max(0, r - 0.5 + x) * 1
    # If x < 0: wall_dist = 0.5 + x. viol = r - 0.5 - x.
    # d/dx (max(0, r - 0.5 - x)^2) = 2 * max(0, r - 0.5 - x) * (-1)
    
    wx_pos = r - (0.5 - c[:, 0]) # r - 0.5 + x
    wy_pos = r - (0.5 - c[:, 1])
    wx_neg = r - (0.5 + c[:, 0]) # r - 0.5 - x
    wy_neg = r - (0.5 + c[:, 1])
    
    grad[:, 0] += 2.0 * np.maximum(wx_pos, 0.0) * np.sign(c[:, 0])
    grad[:, 1] += 2.0 * np.maximum(wy_pos, 0.0) * np.sign(c[:, 1])
    
    # Note: np.sign(0) is 0, which is correct as the gradient is 0 at the kink if we consider subgradient, 
    # but actually for x>0 we use wx_pos, for x<0 we use wx_neg.
    # The formula above is slightly incorrect for x<0.
    # Let's be more precise.
    grad[:, 0] = np.zeros(N)
    grad[:, 1] = np.zeros(N)
    
    # For x_i > 0: viol = max(0, r - 0.5 + x_i). grad = 2*viol * 1
    # For x_i < 0: viol = max(0, r - 0.5 - x_i). grad = 2*viol * (-1)
    # For x_i = 0: grad = 0 (subgradient)
    
    mask_pos_x = c[:, 0] > 0
    mask_neg_x = c[:, 0] < 0
    mask_pos_y = c[:, 1] > 0
    mask_neg_y = c[:, 1] < 0
    
    viol_pos_x = np.maximum(r - 0.5 + c[mask_pos_x, 0], 0.0)
    viol_neg_x = np.maximum(r - 0.5 - c[mask_neg_x, 0], 0.0)
    viol_pos_y = np.maximum(r - 0.5 + c[mask_pos_y, 1], 0.0)
    viol_neg_y = np.maximum(r - 0.5 - c[mask_neg_y, 1], 0.0)
    
    grad[mask_pos_x, 0] += 2.0 * viol_pos_x
    grad[mask_neg_x, 0] += -2.0 * viol_neg_x
    grad[mask_pos_y, 1] += 2.0 * viol_pos_y
    grad[mask_neg_y, 1] += -2.0 * viol_neg_y
    
    # Pairwise gradients
    # f = sum_{i<j} max(0, 2r - d_ij)^2
    # d_ij = ||c_i - c_j||
    # d/dc_i f = sum_{j != i} 2 * max(0, 2r - d_ij) * (-1) * (c_i - c_j) / d_ij
    # = -2 * sum_{j != i} max(0, 2r - d_ij) / d_ij * (c_i - c_j)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e-10) # Avoid div by 0
    
    viol = np.maximum(2*r - dists, 0.0)
    # coeff_ij = 2 * viol_ij / d_ij
    coeff = 2.0 * viol / dists
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = - sum_j coeff_ij * (c_i - c_j)
    # Note: coeff is symmetric.
    grad_pairs = -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    grad += grad_pairs
    
    return grad.flatten()

def _try_radius(r, x0, time_limit=10.0):
    """
    Try to find a feasible configuration for radius r starting from x0.
    Returns (success, x_final).
    """
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(
            _violation_loss, 
            x0, 
            args=(r,),
            jac=_violation_grad,
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return False, x0
            
        final_loss = result.fun
        # If loss is very small, we consider it feasible
        if final_loss < 1e-8:
            return True, result.x
        else:
            return False, result.x
    except Exception:
        return False, x0

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Get a good initial guess
    cents0, r0 = _construct_hexagonal()
    x0 = cents0.reshape(-1)
    
    # Binary search range
    # Lower bound: 0
    # Upper bound: 0.5 (max possible for 1 circle, but for 50 it's much smaller)
    # A safe upper bound is 0.1
    r_low = 0.0
    r_high = 0.1
    
    # First, verify r0 is feasible (it should be, with r0 being the actual score)
    # Actually, r0 is the score, so it is feasible by definition (with 0 violation).
    # But our _try_radius might not converge to 0 exactly due to tolerance.
    # Let's just start the binary search.
    
    best_r = r0
    best_cents = cents0.copy()
    
    # Perform binary search
    # We want to find the largest r such that _try_radius(r) returns True.
    # We will do a few iterations.
    
    # Initial check: is r_high feasible?
    success, x_final = _try_radius(r_high, x0, time_limit=15.0)
    if success:
        r_low = r_high
        best_cents = x_final.reshape(N, 2)
        best_r = r_high
    else:
        # If r_high is not feasible, we might need to lower the upper bound or just proceed.
        # Given r0 ~ 0.07, 0.1 is likely infeasible.
        pass
    
    # If r_low is still 0 or low, let's set it to r0 to save time?
    # No, we want to explore above r0.
    # Let's set r_low to r0 * 0.9 to be safe, and r_high to r0 * 1.5
    # Actually, let's just use the previous logic but tighten the range.
    
    # Reset search range based on r0
    r_low = r0 * 0.9
    r_high = min(0.1, r0 * 1.5)
    
    # Check if r_high is feasible
    if time.time() - start_time < deadline - 20:
        success, x_final = _try_radius(r_high, x0, time_limit=15.0)
        if success:
            r_low = r_high
            best_cents = x_final.reshape(N, 2)
            best_r = r_high
        else:
            # r_high is infeasible, keep r_low = r0 * 0.9 (which is feasible)
            # Actually we know r0 is feasible. Let's set r_low = r0.
            r_low = r0

    # Binary search iterations
    # Each iteration takes ~10-15s. We have ~120s left.
    # We can do about 5-8 iterations.
    
    x0 = best_cents.reshape(-1)
    
    for i in range(10):
        if time.time() - start_time > deadline - 15:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible solution for r_mid
        # Use the best known feasible configuration as the starting point
        success, x_final = _try_radius(r_mid, x0, time_limit=12.0)
        
        if success:
            r_low = r_mid
            best_r = r_mid
            best_cents = x_final.reshape(N, 2)
            x0 = x_final
        else:
            r_high = r_mid
            
    # Final check: ensure the best_cents is valid and inside bounds
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    # Calculate final score to be sure
    final_score = _score(best_cents)
    
    return best_cents

# EVOLVE-BLOCK-END
