"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    Returns (radius, min_dist_idx) where min_dist_idx is the index of the pair
    or wall constraint that is tightest (for debugging, not strictly needed for score).
    """
    n = len(centers)
    # Wall distances
    wall_dists = np.min(np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # For N=50, O(N^2) is fine, but let's use a simple loop or vectorized
    # Vectorized:
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf to ignore self
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _objective_and_grad(x, r_target=None, penalty_weight=100.0):
    """
    Objective function for optimization.
    We want to maximize the minimum distance.
    A common technique is to minimize the sum of violations of a target radius,
    or use a soft-min.
    
    Here we use a direct approach:
    If r_target is None, we maximize the minimum distance (minimize negative soft-min).
    If r_target is given, we minimize the overlap violations (distance < 2*r_target)
    and wall violations (dist to wall < r_target).
    
    Let's stick to a robust approach:
    Minimize: sum( max(0, 2*r - dist_ij)^2 ) + sum( max(0, r - wall_dist_i)^2 )
    This is a smooth function if we use softplus or just squared hinge.
    But we don't know r.
    
    Alternative:
    Maximize the minimum distance.
    Let's use the soft-min of all constraints (pair dists and wall dists).
    soft_min = -1/beta * log(sum(exp(-beta * v)))
    We minimize -soft_min.
    """
    centers = x.reshape(N, 2)
    
    # Wall distances
    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    
    # Pairwise distances
    # Efficient computation for N=50
    # diff (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    all_vals = np.concatenate([pair_dists, wall_dists.ravel()])
    
    # Soft min
    # To avoid overflow, shift by min
    min_val = np.min(all_vals)
    # We want to maximize min_val.
    # So we minimize -min_val.
    # Using soft min:
    beta = 50.0
    shifted = -beta * (all_vals - min_val)
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        sm = min_val
    else:
        sm = min_val - (1.0 / beta) * np.log(sum_exp)
    
    val = -sm
    
    # Gradient is hard to compute analytically for soft-min easily in this form.
    # L-BFGS-B can use numerical gradients, but it's slow.
    # Let's try to provide an analytical gradient or use a different method.
    
    # Actually, let's use a different objective that is easier to differentiate.
    # Minimize the sum of squared violations of a lower bound?
    # Or simply minimize the sum of distances? No.
    
    # Let's try to return just the value and let L-BFGS-B use numerical gradients.
    # For N=50, 100 variables. Numerical gradient requires 200 evaluations.
    # Each evaluation is O(N^2) = 2500 ops.
    # 200 * 2500 = 500,000 ops per gradient step.
    # L-BFGS-B might take 100-500 steps.
    # 500 * 500,000 = 250,000,000 ops.
    # This might be too slow in Python/NumPy if not optimized.
    
    # Let's try to optimize the distance computation.
    # Or use a different optimizer like SLSQP with analytic gradient?
    # Or use Nelder-Mead? No, too slow.
    
    # Let's try to implement the gradient for the soft-min.
    # f(x) = -soft_min(v(x))
    # soft_min(v) = m - 1/beta * log(sum(exp(-beta*(v_i - m)))) where m = min(v)
    # This is not differentiable at the minimum.
    
    # Alternative: Use a "max-min" formulation via binary search on r.
    # Check if there exists a configuration with radius >= r.
    # This is a feasibility problem.
    # Constraints: dist_ij >= 2r, wall_dist_i >= r.
    # We can minimize the violation:
    # V(x) = sum( max(0, 2r - dist_ij)^2 ) + sum( max(0, r - wall_dist_i)^2 )
    # If V(x) == 0, then r is achievable.
    # We can binary search on r.
    # For each r, we run L-BFGS-B to minimize V(x).
    # The gradient of V(x) is easy to compute.
    
    return val

def _violation_objective(x, r, beta=100.0):
    """
    Minimize violation of radius r.
    V(x) = sum( max(0, 2r - dist_ij)^2 ) + sum( max(0, r - wall_dist_i)^2 )
    We use a soft version to make it differentiable:
    max(0, u)^2 approx by u^2 / (1 + exp(-beta*u)) or similar?
    Or just use the hard version and let the optimizer handle the kink?
    L-BFGS-B might struggle with non-smoothness.
    
    Let's use a smooth approximation:
    h(u) = u^2 if u > 0, 0 otherwise.
    Smooth: h(u) = 0.5 * u * (u + softplus(u, beta)) ? No.
    Simple smooth max: m(u) = 0.5 * (u + sqrt(u^2 + eps))
    Then m(u)^2.
    
    Let's try m(u) = u / (1 + exp(-beta*u)) * beta? No.
    
    Let's just use the hard version first and see if L-BFGS-B works.
    If not, we can switch to SLSQP or COBYLA.
    """
    centers = x.reshape(N, 2)
    
    # Wall distances
    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    wall_viol = np.maximum(0, r - wall_dists)
    wall_cost = np.sum(wall_viol**2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    pair_viol = np.maximum(0, 2*r - pair_dists)
    pair_cost = np.sum(pair_viol**2)
    
    return pair_cost + wall_cost

def _violation_grad(x, r):
    """
    Gradient of the violation objective.
    """
    centers = x.reshape(N, 2)
    grad = np.zeros_like(centers)
    
    # Wall gradient
    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    wall_viol = np.maximum(0, r - wall_dists)
    
    # d/dx (max(0, r - w(x))^2) = 2 * max(0, r-w) * (-dw/dx)
    # w_x = 0.5 - |x|
    # dw/dx = -sign(x)
    # So -dw/dx = sign(x)
    
    # For x coord:
    sign_x = np.sign(centers[:, 0])
    sign_y = np.sign(centers[:, 1])
    
    # Wall dist x: 0.5 - |x|
    # If x > 0, w = 0.5 - x, dw/dx = -1. -dw/dx = 1.
    # If x < 0, w = 0.5 + x, dw/dx = 1. -dw/dx = -1.
    # So -dw/dx = sign(x)
    
    grad[:, 0] += 2 * wall_viol[:, 0] * sign_x
    grad[:, 1] += 2 * wall_viol[:, 1] * sign_y
    
    # Pair gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0) # Avoid division by zero
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_viol = np.maximum(0, 2*r - pair_dists)
    
    # d/dx_i (max(0, 2r - d_ij)^2) = 2 * max(0, 2r-d_ij) * (-d(d_ij)/dx_i)
    # d_ij = ||x_i - x_j||
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    
    # We need to accumulate gradients for each i and j.
    # For each pair (i, j):
    # grad_i += 2 * viol_ij * (-(x_i - x_j)/d_ij)
    # grad_j += 2 * viol_ij * ((x_i - x_j)/d_ij)
    
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        viol = pair_viol[k]
        if viol == 0:
            continue
        d_ij = pair_dists[k]
        vec = centers[i] - centers[j]
        grad_vec = vec / d_ij
        
        grad[i] -= 2 * viol * grad_vec
        grad[j] += 2 * viol * grad_vec
    
    return grad.flatten()

def _construct_initial_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_centers = _construct_initial_grid()
    best_radius = _compute_radius(best_centers)
    
    # Binary search on r
    # Lower bound: 0
    # Upper bound: 0.5 / sqrt(N) approx? Or just 0.5/2 = 0.25?
    # Max possible radius is when circles are in a grid.
    # For N=50, 7x7 grid. Spacing 1/7. Radius 1/14 ~ 0.0714.
    # So upper bound 0.08 is safe.
    
    r_low = 0.0
    r_high = 0.08
    
    # We will perform a few iterations of binary search.
    # For each r, we try to find a configuration that satisfies the constraints.
    # If we find one, r_low = r. Else r_high = r.
    
    # To speed up, we can use the previous solution as the initial guess for the next r.
    current_centers = best_centers.copy()
    
    # Number of binary search steps
    # We have ~140 seconds.
    # Each L-BFGS-B run might take a few seconds.
    # Let's do 10-15 steps.
    
    for step in range(15):
        if time.time() > deadline - 5.0:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to optimize for this r
        x0 = current_centers.flatten()
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = opt.minimize(
                _violation_objective,
                x0,
                jac=_violation_grad,
                args=(r_mid,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            x_res = res.x
            r_actual = _compute_radius(x_res.reshape(N, 2))
            
            if r_actual >= r_mid - 1e-6:
                # Feasible
                r_low = r_mid
                if r_actual > best_radius:
                    best_radius = r_actual
                    best_centers = x_res.reshape(N, 2).copy()
                current_centers = x_res.reshape(N, 2).copy()
            else:
                # Infeasible
                r_high = r_mid
                # Update current centers to the result of the optimization?
                # It might be closer to feasible than the previous one.
                # But if it's infeasible, it might have overlaps.
                # Let's keep the previous best feasible center as the starting point
                # for the next (lower) r.
                # Actually, using the result of the infeasible run as a start for a lower r
                # might be good because it's "pushed" towards the feasible region.
                # Let's use it.
                current_centers = x_res.reshape(N, 2).copy()
                
        except Exception as e:
            # If optimization fails, break or reduce r_high
            r_high = r_mid
            
    # Final polish
    if time.time() < deadline - 2.0:
        # Try to improve the best radius found by local search
        # Use the soft-min approach for final polish
        beta = 100.0
        
        def _soft_min_obj(x):
            centers = x.reshape(N, 2)
            wall_dists = np.column_stack((
                0.5 - np.abs(centers[:, 0]),
                0.5 - np.abs(centers[:, 1])
            ))
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            iu = np.triu_indices(N, k=1)
            pair_dists = dists[iu]
            all_vals = np.concatenate([pair_dists, wall_dists.ravel()])
            min_val = np.min(all_vals)
            shifted = -beta * (all_vals - min_val)
            sum_exp = np.sum(np.exp(shifted))
            if sum_exp == 0:
                sm = min_val
            else:
                sm = min_val - (1.0 / beta) * np.log(sum_exp)
            return -sm
        
        try:
            res = opt.minimize(
                _soft_min_obj,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_res = res.x
            r_res = _compute_radius(x_res.reshape(N, 2))
            if r_res > best_radius:
                best_radius = r_res
                best_centers = x_res.reshape(N, 2).copy()
        except Exception:
            pass
            
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    if not np.all(np.isfinite(best_centers)):
        return _construct_initial_grid()
        
    return best_centers
# EVOLVE-BLOCK-END
