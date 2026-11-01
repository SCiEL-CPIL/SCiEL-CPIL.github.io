import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50

def _score(cents):
    """
    Compute the packing radius (score).
    """
    n = len(cents)
    if n == 0:
        return 0.0
        
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1]))
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distances
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0
    
    # Try multiple row configurations
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        # Estimate r to fit
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

def _feasibility_check(r_target, initial_cents, time_limit=5.0):
    """
    Check if circles of radius r_target can fit.
    Uses L-BFGS-B to minimize a violation function.
    Returns (is_feasible, best_cents_found)
    """
    n = len(initial_cents)
    
    # Objective: sum of violations
    # Violation for pair (i,j): max(0, 2*r_target - dist(i,j))
    # Violation for wall i: max(0, r_target - wall_dist(i))
    
    def objective_and_grad(x):
        cents = x.reshape(-1, 2)
        
        # Wall violations
        wall_dists = np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1]))
        wall_viol = np.maximum(0, r_target - wall_dists)
        
        # Pair violations
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        pair_viol_matrix = np.maximum(0, 2 * r_target - dists)
        pair_viol = np.sum(pair_viol_matrix)
        
        total_viol = np.sum(wall_viol) + pair_viol
        
        # Gradients
        grad = np.zeros_like(cents)
        
        # Wall grads
        # d/dx_i max(0, r - (0.5 - |x_i|))
        # If x_i > 0: d/dx (0.5 - x_i) = -1. So d/dx viol = -1 * (if active)
        # If x_i < 0: d/dx (0.5 + x_i) = 1. So d/dx viol = 1 * (if active)
        # Basically, gradient points towards the wall.
        # Let's compute carefully.
        # w_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # v_i = max(0, r - w_i)
        # If v_i > 0, we want to increase w_i.
        # grad_x w_i = -sign(x_i) if |x_i| < |y_i| else 0 (approx, assuming x is the min term)
        # Actually, let's just use subgradient.
        # If x_i > 0, moving x_i left increases 0.5-x_i. So grad x is -1.
        # If x_i < 0, moving x_i right increases 0.5+x_i. So grad x is 1.
        # So grad_x = -sign(x_i) if x is the limiting dimension.
        
        for i in range(n):
            if wall_viol[i] > 0:
                x, y = cents[i]
                dist_x = 0.5 - abs(x)
                dist_y = 0.5 - abs(y)
                if dist_x <= dist_y:
                    grad[i, 0] -= np.sign(x) if x != 0 else 1.0 # If x==0, either direction works, pick +1? No, 0.5-0 is max. Gradient is 0? No, subgradient is [-1, 1]. Let's use 0 to be safe? Or push to center?
                    # Actually if x=0, dist_x=0.5. If r>0.5, impossible. But r is small.
                    # If x=0, we can move in any direction. Let's say 0.
                    if x == 0: grad[i, 0] = 0
                else:
                    grad[i, 1] -= np.sign(y) if y != 0 else 0.0
                    if y == 0: grad[i, 1] = 0
                    
                # Note: The objective is to MINIMIZE violation.
                # Violation = r - dist.
                # d/dx (r - (0.5 - x)) = 1 (if x>0).
                # So we want to DECREASE x.
                # Gradient of violation wrt x is +1 (if x>0).
                # So we should subtract gradient.
                # My previous logic: "grad points towards wall" -> if x>0, wall is right, so +x.
                # Wait. If x>0, dist = 0.5-x.
                # viol = r - (0.5-x) = r - 0.5 + x.
                # d viol / dx = 1.
                # To minimize viol, we need to move in direction -grad = -1. i.e., decrease x.
                # So grad[i, 0] should be +1 if x>0.
                
                # Let's re-evaluate.
                # I will re-calculate grads in the loop below properly.
                pass
                
        # Let's do a clean gradient calculation
        grad[:] = 0.0
        
        for i in range(n):
            if wall_viol[i] > 0:
                x, y = cents[i]
                dist_x = 0.5 - abs(x)
                dist_y = 0.5 - abs(y)
                
                # Gradient of wall violation wrt x, y
                # If dist_x is the min:
                #   if x > 0: viol = r - (0.5 - x) => d/dx = 1
                #   if x < 0: viol = r - (0.5 + x) => d/dx = -1
                #   if x == 0: subgrad [-1, 1]. Let's use 0.
                # If dist_y is the min: similar for y.
                
                if dist_x <= dist_y:
                    if x > 0: grad[i, 0] += 1.0
                    elif x < 0: grad[i, 0] -= 1.0
                    else: grad[i, 0] += 0.0
                else:
                    if y > 0: grad[i, 1] += 1.0
                    elif y < 0: grad[i, 1] -= 1.0
                    else: grad[i, 1] += 0.0

        # Pair gradients
        # viol_ij = max(0, 2r - dist_ij)
        # d/dx_i viol_ij = - (x_i - x_j) / dist_ij * (1 if active else 0)
        # Wait. dist = ||c_i - c_j||.
        # d dist / d c_i = (c_i - c_j) / dist.
        # d viol / d c_i = - d dist / d c_i = -(c_i - c_j) / dist.
        # So if active, we want to move c_i away from c_j.
        # Direction c_i - c_j is away from c_j.
        # So gradient should be in direction c_j - c_i?
        # Let's check: minimize viol.
        # If c_i is too close to c_j, we want to increase dist.
        # Increasing dist means moving c_i in direction c_i - c_j.
        # So the "force" is c_i - c_j.
        # Gradient of objective (viol) is opposite to force?
        # Objective f = 2r - dist.
        # grad f = - grad dist = -(c_i - c_j)/dist.
        # So grad points from i to j?
        # c_i - c_j points from j to i.
        # So -(c_i - c_j) points from i to j.
        # Yes, gradient points towards j.
        # So we subtract gradient -> move away from j. Correct.
        
        for i in range(n):
            for j in range(i + 1, n):
                if pair_viol_matrix[i, j] > 0:
                    d = dists[i, j]
                    if d < 1e-12:
                        # Random push
                        grad[i, 0] += 1.0
                        grad[j, 0] -= 1.0
                    else:
                        dx = cents[i, 0] - cents[j, 0]
                        dy = cents[i, 1] - cents[j, 1]
                        factor = 1.0 / d
                        # grad_i = - (c_i - c_j) / d
                        grad[i, 0] -= dx * factor
                        grad[i, 1] -= dy * factor
                        # grad_j = + (c_i - c_j) / d
                        grad[j, 0] += dx * factor
                        grad[j, 1] += dy * factor
                        
        return total_viol, grad.flatten()

    # Bounds
    # Centers must be in [-0.5, 0.5]
    # Actually, to be safe, we can bound them tighter?
    # No, just [-0.5, 0.5] is the physical constraint.
    # But if we allow centers at exactly 0.5, and r>0, it's invalid.
    # The objective handles the penalty.
    # L-BFGS-B bounds:
    bounds = [(-0.5, 0.5) for _ in range(2 * n)]
    
    x0 = initial_cents.flatten()
    
    start_time = time.time()
    
    try:
        res = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 5000,
                # 'disp': True
            }
        )
        
        # Check time
        if time.time() - start_time > time_limit:
            # Should not happen if maxiter is low, but just in case
            pass
            
        final_viol = res.fun
        final_cents = res.x.reshape(-1, 2)
        
        # If violation is very small, consider it feasible
        # Threshold depends on r. 1e-5 is reasonable for double precision
        if final_viol < 1e-5:
            return True, final_cents
        else:
            return False, final_cents
            
    except Exception as e:
        # If optimization fails, return infeasible
        return False, initial_cents

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Step 1: Create hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # Step 2: Binary search on radius
    # Lower bound: current best
    # Upper bound: theoretical max?
    # For N circles in unit square, max r is roughly 1/(2*sqrt(N)) * some factor.
    # For N=50, sqrt(50)~7.07. 1/(2*7.07) ~ 0.07.
    # Known best is 0.0713.
    # Let's set upper bound to 0.075.
    
    r_low = best_r
    r_high = 0.075
    
    # We will try to find the maximum feasible r.
    # Number of iterations: log2((r_high - r_low) / precision)
    # Precision 1e-6.
    # Range ~ 0.004. log2(4000) ~ 12 iterations.
    # Each iteration: 1 feasibility check.
    # Feasibility check: L-BFGS-B.
    # If L-BFGS-B takes 5s, 12 * 5 = 60s. Fits in budget.
    
    # To be robust, we can run a few iterations with different initializations
    # or just stick to the hexagonal one which is good.
    
    # Let's do a simple binary search.
    # We'll keep track of the best feasible configuration found.
    
    current_best_feasible_cents = best_cents
    current_best_feasible_r = best_r
    
    # We can also try to improve the initial guess using the result of the last successful check?
    # Yes, that's a good idea.
    
    # However, binary search usually starts from the middle.
    # If we fail at r_mid, we lower r_high.
    # If we succeed at r_mid, we raise r_low and keep the found configuration as the new initial guess for the next higher r?
    # Actually, the configuration for r_mid is valid for r_mid, but might not be the best starting point for r_mid + delta.
    # But it's a good starting point.
    
    # Let's structure the loop:
    for _ in range(15):  # Max 15 iterations
        if time.time() - start_time > deadline - 10:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Use the best feasible config found so far as initial guess?
        # Or the original hex?
        # Using the best feasible config is better as it's already "tight".
        init_cents = current_best_feasible_cents
        
        is_feas, found_cents = _feasibility_check(r_mid, init_cents, time_limit=5.0)
        
        if is_feas:
            r_low = r_mid
            current_best_feasible_r = r_mid
            current_best_feasible_cents = found_cents
            # Update global best
            # The score of found_cents should be >= r_mid (since it's feasible)
            # Actually, it might be slightly higher if the optimizer over-shoots?
            # No, it minimizes violation. If violation is 0, dist >= 2r_mid.
            # So score >= r_mid.
            # Let's compute actual score to be safe.
            actual_r = _score(found_cents)
            if actual_r > best_r:
                best_r = actual_r
                best_cents = found_cents.copy()
        else:
            r_high = r_mid
            # If infeasible, the found_cents might still be a good starting point for lower r?
            # Yes, but we don't update current_best_feasible_cents because it's not feasible for r_mid.
            # However, it might be feasible for r_low.
            # We can just keep the previous feasible one.
            
    # Final check: ensure we return the best valid configuration
    # best_cents should be valid.
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents

if __name__ == "__main__":
    # Quick test
    c = construct()
    print(_score(c))
