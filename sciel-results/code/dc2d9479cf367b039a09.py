"""
Strategy: Binary search on the target radius r with violation-minimization feasibility checks.
Instead of minimizing a smooth proxy objective via gradient descent or force-directed annealing,
we treat the problem as a feasibility problem: can we place N circles of radius r in the square?
We binary search on r, and for each candidate r, we minimize the sum of squared violations
(constraints that circles must not overlap and must stay inside the square) using L-BFGS-B.
If the minimum violation is below a tolerance, the radius is feasible; otherwise, it is not.
This directly targets the maximum radius and uses the structure of the constraint satisfaction
problem rather than an indirect energy landscape.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
# Known best radius from Boll, Donovan, Graham, Lubachevsky (2000)
R_BEST = 0.071377103865

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

def _violation_loss(x, r):
    """
    Compute the sum of squared violations for a given radius r.
    x is a flattened array of shape (2N,) representing centers.
    Violations:
    1. Pairwise: d_ij < 2r -> violation = max(0, 2r - d_ij)
    2. Wall: center outside [r, 1-r] in each dimension -> violation = max(0, r - dist_to_wall)
    """
    cents = x.reshape(N, 2)
    loss = 0.0
    grad = np.zeros_like(x)
    
    # Pairwise violations
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For each pair, if dist < 2r, violation = 2r - dist
    viol_matrix = np.maximum(0, 2*r - dists)
    loss += np.sum(viol_matrix**2)
    
    # Gradient for pairwise
    # d/du (max(0, 2r - u))^2 = 2*(2r-u)*(-1) if u < 2r else 0
    # where u = dist
    # d/dx_i = sum_j 2*(2r - d_ij) * (-1) * (x_i - x_j) / d_ij  (for i != j)
    # We compute this efficiently
    # grad_pair[i] = -2 * sum_{j!=i} (2r - d_ij) * (cents[i] - cents[j]) / d_ij  (if d_ij < 2r)
    
    # To avoid division by zero and handle only active constraints:
    active = dists < 2*r
    # Set diagonal to False
    np.fill_diagonal(active, False)
    
    if np.any(active):
        # weights[i,j] = 2 * (2r - dists[i,j]) / dists[i,j] if active else 0
        weights = np.zeros_like(dists)
        weights[active] = 2 * (2*r - dists[active]) / dists[active]
        
        # grad_c[i] = - sum_j weights[i,j] * (cents[i] - cents[j])
        # = - (sum_j weights[i,j] * cents[i] - sum_j weights[i,j] * cents[j])
        # = - cents[i] * sum_j weights[i,j] + sum_j weights[i,j] * cents[j]
        sum_weights = weights.sum(axis=1)
        weighted_cents = weights @ cents
        grad_c = - (cents * sum_weights[:, np.newaxis] - weighted_cents)
        grad = grad_c.flatten()
    
    # Wall violations
    # For x dimension:
    # left wall: x < r -> viol = r - x
    # right wall: x > 1-r -> viol = x - (1-r)
    # In our coordinate system [-0.5, 0.5], walls are at -0.5 and 0.5
    # So:
    # left: x < -0.5 + r -> viol = (-0.5 + r) - x
    # right: x > 0.5 - r -> viol = x - (0.5 - r)
    
    cx = cents[:, 0]
    cy = cents[:, 1]
    
    viol_x_left = np.maximum(0, (-0.5 + r) - cx)
    viol_x_right = np.maximum(0, cx - (0.5 - r))
    viol_y_left = np.maximum(0, (-0.5 + r) - cy)
    viol_y_right = np.maximum(0, cy - (0.5 - r))
    
    loss += np.sum(viol_x_left**2) + np.sum(viol_x_right**2)
    loss += np.sum(viol_y_left**2) + np.sum(viol_y_right**2)
    
    # Gradients for walls
    grad_c = np.zeros_like(cents)
    grad_c[:, 0] += -2 * viol_x_left + 2 * viol_x_right
    grad_c[:, 1] += -2 * viol_y_left + 2 * viol_y_right
    grad += grad_c.flatten()
    
    return loss, grad

def _feasible(r, x0, time_limit=10.0):
    """
    Check if a configuration with radius r is feasible.
    Returns (is_feasible, best_x, best_loss)
    """
    start_time = time.time()
    
    # Use L-BFGS-B to minimize violation loss
    res = minimize(
        _violation_loss,
        x0,
        args=(r,),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
    )
    
    if time.time() - start_time > time_limit:
        return False, x0, res.fun if hasattr(res, 'fun') else 1e10
    
    # Consider feasible if loss is very small
    tol = 1e-10
    return res.fun < tol, res.x, res.fun

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
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
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
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
    deadline = 145.0  # Safety margin before 150s kill
    
    # Step 1: Get initial configuration
    cents0, r0 = _construct_hexagonal()
    x0 = cents0.flatten()
    
    best_r = r0
    best_x = x0.copy()
    
    # Step 2: Binary search on radius
    # Lower bound: current best radius
    # Upper bound: known best radius (slightly above to ensure we can find it)
    r_low = best_r
    r_high = R_BEST * 1.001  # Slightly above known best
    
    # We want to find the maximum feasible radius
    # Binary search for ~30 iterations or until time limit
    for i in range(30):
        if time.time() > deadline - 10:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        # Use the previous best configuration as a starting point if possible
        x_start = best_x if r_mid <= best_r else x0
        
        feasible, x_opt, loss = _feasible(r_mid, x_start, time_limit=5.0)
        
        if feasible:
            # r_mid is feasible, try larger
            r_low = r_mid
            best_r = r_mid
            best_x = x_opt.copy()
        else:
            # r_mid is not feasible, try smaller
            r_high = r_mid
            
        # Check convergence
        if r_high - r_low < 1e-8:
            break
    
    # Final refinement: try to improve the best configuration found
    if time.time() < deadline - 5:
        # Do a final local optimization on the best radius found
        # Minimize violation for the best_r to get a cleaner configuration
        res = minimize(
            _violation_loss,
            best_x,
            args=(best_r,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        if res.fun < 1e-10:
            best_x = res.x
    
    cents_final = best_x.reshape(N, 2)
    
    # Ensure valid
    if not np.all(np.isfinite(cents_final)):
        cents_final = cents0
    
    # Clip to be safe
    cents_final = np.clip(cents_final, -0.5, 0.5)
    
    # Compute final score
    final_r = _score(cents_final)
    
    return cents_final
