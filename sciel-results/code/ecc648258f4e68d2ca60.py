# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _compute_radius(pts):
    """Compute the packing radius for a given set of points."""
    if len(pts) == 0:
        return 0.0
    # Vectorized distance calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    
    radii = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - np.max(radii)
    
    return min(min_pair, min_bound)

def _hex_init(n_points, seed=0):
    """
    Generate multiple diverse hexagonal lattice initializations 
    and return the best one based on initial packing radius.
    """
    best_pts = None
    best_r = -1.0
    
    # Try different rotations and slight spacing variations
    for attempt in range(10):
        # Base spacing estimation
        # Area of unit circle is pi. 
        # Area of N circles is N * pi * r^2.
        # Hex density is ~0.9069.
        # N * (2r)^2 * (sqrt(3)/2) * 0.9069 ~ pi
        # r ~ sqrt(pi / (N * 4 * sqrt(3)/2 * 0.9069))
        # Let's just start with r ~ 0.053 and adjust
        base_spacing = 2.0 / (np.sqrt(n_points) * 1.1)
        
        # Vary spacing slightly to find a better fit
        spacing = base_spacing * (0.95 + 0.01 * attempt)
        
        # Generate points in a grid that covers the unit circle
        k = int(1.0 / spacing) + 5
        
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                # Standard hex lattice
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                
                # Apply rotation
                angle = attempt * np.pi / 5.0
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                
                if x_rot**2 + y_rot**2 <= (1.0 - spacing*0.5)**2:
                    points.append([x_rot, y_rot])
        
        if len(points) < n_points:
            continue
            
        points = np.array(points)
        
        # Sort by distance from center to get the innermost N points
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n_points]
        pts = points[idx]
        
        # Calculate radius
        r = _compute_radius(pts)
        
        if r > best_r:
            best_r = r
            best_pts = pts

    if best_pts is None:
        # Fallback: Random points scaled to fit
        best_pts = np.random.rand(n_points, 2) * 1.8 - 0.9
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts / norms[:, np.newaxis] * 0.9
        best_r = _compute_radius(best_pts)

    return best_pts

def _objective_and_grad(x, lam):
    """
    Objective: Minimize -soft_min(distances)
    This is equivalent to Maximize soft_min(distances)
    Distances: all pairwise distances and (1 - |c_i|)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # 1. Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # 2. Compute boundary distances
    radii = np.linalg.norm(pts, axis=1)
    bound_dist = 1.0 - radii
    
    # 3. Combine all distances into a single vector for log-sum-exp
    iu = np.triu_indices(n, 1)
    d_pairs = dist[iu]
    all_dists = np.concatenate([d_pairs, bound_dist])
    
    # 4. Objective Value
    # soft_min = min_val - (1/lam) * log(sum(exp(-lam * (d - min_val))))
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    exp_terms = np.exp(-lam * shifted)
    log_sum = np.log(np.sum(exp_terms))
    m_soft = min_val - log_sum / lam
    obj = -m_soft
    
    # 5. Gradient
    # Grad = sum P_k * d d_k/dx
    # P_k = exp(-lam * (d_k - min_val)) / sum(exp(-lam * (d - min_val)))
    S = np.sum(exp_terms)
    P = exp_terms / S
    
    n_pairs = len(d_pairs)
    P_pairs = P[:n_pairs]
    P_bonds = P[n_pairs:]
    
    # Gradient accumulation
    grad = np.zeros_like(pts)
    
    # A. Boundary Gradient
    # d (1 - |c_i|) / d c_i = - c_i / |c_i|
    radii_safe = np.maximum(radii, 1e-8)
    unit_vecs = pts / radii_safe[:, np.newaxis]
    grad_bound = -P_bonds[:, np.newaxis] * unit_vecs
    
    # B. Pairwise Gradient
    # Create weight matrix W
    W = np.zeros((n, n))
    # Avoid division by zero (dist is inf on diagonal, but we only fill upper triangle)
    # d_pairs is valid
    W[iu[0], iu[1]] = P_pairs / d_pairs
    W[iu[1], iu[0]] = P_pairs / d_pairs
    
    # Grad_pair[i] = sum_j W[i, j] * (c_i - c_j)
    row_sums = np.sum(W, axis=1)
    W_pts = W @ pts
    grad_pair = pts * row_sums[:, np.newaxis] - W_pts
    
    total_grad = grad_bound + grad_pair
    
    return obj, total_grad.ravel()

def _slsqp_objective(x, r_target):
    """
    Objective for SLSQP: Maximize minimum distance.
    We minimize -min_dist.
    Since min is non-smooth, we use a soft approximation or just rely on SLSQP's constraint handling.
    Actually, SLSQP can handle non-smoothness poorly. 
    Better approach: Minimize -t subject to dist >= t and bound >= t.
    But we need a single objective function for scipy.optimize.minimize with SLSQP.
    
    Let's use the auxiliary variable approach directly in the objective?
    No, SLSQP takes a vector x. We can append t to the vector.
    x = [c_1, ..., c_N, t]
    Objective: minimize -t
    Constraints:
      1. |c_i - c_j| >= 2*t for all i<j
      2. 1 - |c_i| >= t for all i
      3. t > 0
    
    This is a standard QP-like formulation (though non-convex due to quadratic constraints).
    """
    n = N
    pts = x[:2*n].reshape(-1, 2)
    t = x[2*n]
    
    # Objective: -t
    obj = -t
    
    # Constraints for SLSQP must be in form: g(x) >= 0
    # We define them as g(x) >= 0.
    
    # 1. Boundary constraints: 1 - |c_i| - t >= 0
    radii = np.linalg.norm(pts, axis=1)
    bound_constr = 1.0 - radii - t
    
    # 2. Pairwise constraints: |c_i - c_j|^2 - (2*t)^2 >= 0
    # Using squared distance to avoid sqrt in gradient? 
    # SLSQP uses finite differences for gradient if not provided.
    # Providing gradient for 45000+ constraints is too slow.
    # Let's rely on finite differences but be careful.
    # Actually, SLSQP might be too slow for 300 points with ~45k constraints.
    # Let's check the number of constraints.
    # N=300 -> ~45k pairwise + 300 boundary = ~45k constraints.
    # SLSQP with 600 variables and 45k constraints is likely to be very slow or fail.
    
    # Alternative: Use a smooth approximation of the min in the objective, 
    # but use SLSQP for the constraints? No, SLSQP needs constraints.
    
    # Let's stick to the soft-min objective with L-BFGS-B but improve the annealing schedule
    # and add a final polish with a tighter tolerance and possibly a different optimizer if time permits.
    # The directive suggests SLSQP with QP formulation. Let's try a subset of constraints or a different approach.
    
    # Given the size, SLSQP on the full problem is risky for time.
    # Let's try to implement the SLSQP approach but only for a smaller subset or with a very tight time limit.
    # Or, better, improve the L-BFGS-B annealing.
    
    # Let's return a dummy value here and implement the actual logic in construct().
    # This function is a placeholder for the SLSQP attempt which we will implement carefully.
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (limit is 150s)
    
    # 1. Initialization
    points = _hex_init(N)
    x0 = points.ravel()
    best_x = x0.copy()
    best_r = _compute_radius(points)
    
    # 2. Optimization with Annealing
    # We will run several stages with increasing lambda (sharpness)
    # Start smooth, end sharp
    
    # Improved lambda schedule: more stages, slower increase
    lambdas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_x = x0.copy()
    
    for lam in lambdas:
        if time.time() > deadline - 20:
            break
            
        # Use L-BFGS-B
        # Adjust maxiter based on remaining time
        time_left = deadline - time.time()
        if time_left < 20:
            break
            
        # Smaller lambda needs more iterations to move.
        # Larger lambda is stiffer, fewer iterations.
        if lam < 5.0:
            maxiter = 500
        elif lam < 20.0:
            maxiter = 300
        else:
            maxiter = 200
            
        try:
            result = minimize(
                _objective_and_grad, 
                current_x, 
                args=(lam,),
                method='L-BFGS-B', 
                jac=True, 
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            
            if result.x is not None:
                new_r = _compute_radius(result.x.reshape(-1, 2))
                if new_r > best_r:
                    best_r = new_r
                    best_x = result.x.copy()
                # Update current_x to continue from the new position
                current_x = result.x.copy()
                
        except Exception as e:
            # If optimization fails, break
            break
            
    # 3. Final Polish Pass with SLSQP on a subset or tight L-BFGS-B
    if time.time() < deadline - 5:
        pts = best_x.reshape(-1, 2)
        
        # Identify active constraints (pairs close to min distance and points close to boundary)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        radii = np.linalg.norm(pts, axis=1)
        min_bound = 1.0 - np.max(radii)
        min_pair_sq = np.min(dist_sq)
        
        r_current = min(np.sqrt(min_pair_sq) / 2.0, min_bound)
        
        # Find indices of points involved in min pair or min bound
        # Threshold: within 1.1 * r_current of the minimum
        threshold_sq = (2.0 * r_current * 1.1) ** 2
        active_pairs = np.argwhere(dist_sq < threshold_sq)
        # Remove duplicates and self-pairs
        unique_pairs = set()
        for i, j in active_pairs:
            if i < j:
                unique_pairs.add((i, j))
        
        active_bounds = np.where(radii > 1.0 - 1.1 * r_current)[0]
        
        # If we have a manageable number of active constraints, try SLSQP
        n_active_pairs = len(unique_pairs)
        n_active_bounds = len(active_bounds)
        
        if n_active_pairs + n_active_bounds < 500:
            # Define SLSQP objective and constraints
            def slsqp_obj(x_slsqp):
                pts_s = x_slsqp[:2*N].reshape(-1, 2)
                t_s = x_slsqp[2*N]
                return -t_s
            
            # Constraints: g(x) >= 0
            # We will use a list of constraint dictionaries
            constraints = []
            
            # Boundary constraints
            for i in active_bounds:
                def bound_constr(x_slsqp, i=i):
                    pts_s = x_slsqp[:2*N].reshape(-1, 2)
                    t_s = x_slsqp[2*N]
                    return 1.0 - np.linalg.norm(pts_s[i]) - t_s
                constraints.append({'type': 'ineq', 'fun': bound_constr})
            
            # Pairwise constraints
            for i, j in unique_pairs:
                def pair_constr(x_slsqp, i=i, j=j):
                    pts_s = x_slsqp[:2*N].reshape(-1, 2)
                    t_s = x_slsqp[2*N]
                    return np.sum((pts_s[i] - pts_s[j])**2) - (2.0 * t_s)**2
                constraints.append({'type': 'ineq', 'fun': pair_constr})
            
            # Initial guess for t
            t_init = r_current * 0.99
            x_init = np.concatenate([best_x, [t_init]])
            
            # Bounds for SLSQP
            # x: -1 to 1, t: 0 to 0.5
            bounds_slsqp = [(-1.0, 1.0)] * (2 * N) + [(0.0, 0.5)]
            
            try:
                result_slsqp = minimize(
                    slsqp_obj,
                    x_init,
                    method='SLSQP',
                    bounds=bounds_slsqp,
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                
                if result_slsqp.x is not None:
                    final_pts = result_slsqp.x[:2*N].reshape(-1, 2)
                    final_r = _compute_radius(final_pts)
                    if final_r > best_r:
                        best_r = final_r
                        best_x = result_slsqp.x[:2*N].copy()
                        
            except Exception:
                pass
        
        # If SLSQP was not run or failed, do a final high-precision L-BFGS-B polish
        if time.time() < deadline - 2:
            lam_final = 500.0
            try:
                result_final = minimize(
                    _objective_and_grad, 
                    best_x, 
                    args=(lam_final,),
                    method='L-BFGS-B', 
                    jac=True, 
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
                )
                
                if result_final.x is not None:
                    final_r = _compute_radius(result_final.x.reshape(-1, 2))
                    if final_r > best_r:
                        best_r = final_r
                        best_x = result_final.x.copy()
            except Exception:
                pass

    # Final validity check and clipping
    final_pts = best_x.reshape(-1, 2)
    radii = np.linalg.norm(final_pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        final_pts[mask] = final_pts[mask] / radii[mask, np.newaxis] * 0.999999
        
    return final_pts
# EVOLVE-BLOCK-END
