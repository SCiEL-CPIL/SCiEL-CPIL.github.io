import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest triangle area
    to the convex hull area using physics-based relaxation and smooth surrogate optimization.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Safety margin
    
    # Precompute indices for triangles
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    # tri_i, tri_j, tri_k are arrays of length 286

    def get_points_from_angles(angles):
        """Convert 13 angles to 13 (x,y) points on unit circle."""
        x = np.cos(angles)
        y = np.sin(angles)
        return np.column_stack((x, y))

    def calculate_metric(points, beta):
        """
        Calculate the smooth surrogate metric.
        beta controls the sharpness of the min approximation.
        """
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-9:
                return 0.0
        except Exception:
            return 0.0
            
        p_i = points[tri_i]
        p_j = points[tri_j]
        p_k = points[tri_k]
        
        # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        if np.min(areas) < 1e-9:
            return 0.0
            
        # Stable soft-min approximation
        m = np.min(areas)
        shifted = areas - m
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            soft_min = m
        else:
            soft_min = m - (1.0 / beta) * np.log(sum_exp)
            
        return soft_min / A_hull

    def neg_objective_smooth(angles, beta):
        points = get_points_from_angles(angles)
        val = calculate_metric(points, beta)
        return -val

    def physics_relaxation(angles, n_steps=500, lr_init=0.01, lr_final=0.001):
        """
        Physics-based relaxation: Laplacian smoothing with repulsive forces
        to push points apart and maximize minimum triangle area.
        """
        angles = angles.copy()
        
        for step in range(n_steps):
            # Linearly interpolate learning rate
            t = step / max(n_steps - 1, 1)
            lr = lr_init * (1 - t) + lr_final * t
            
            points = get_points_from_angles(angles)
            
            # Compute forces
            forces = np.zeros_like(angles)
            
            # 1. Repulsive forces between all pairs (inverse distance squared)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = points[j, 0] - points[i, 0]
                    dy = points[j, 1] - points[i, 1]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-6:
                        continue
                    # Force magnitude: k / dist^2, direction along the vector
                    force_mag = 0.1 / (dist_sq + 0.01)
                    # Tangential component (project onto tangent at point i)
                    # Tangent at angle theta is (-sin(theta), cos(theta))
                    tangent_i = np.array([-np.sin(angles[i]), np.cos(angles[i])])
                    force_vec = force_mag * np.array([dx, dy])
                    tangent_force = np.dot(force_vec, tangent_i)
                    forces[i] += tangent_force
                    forces[j] -= tangent_force
            
            # 2. Laplacian smoothing: each point moves toward the average of its neighbors
            # (in angular space, neighbors are i-1 and i+1)
            for i in range(n):
                prev_i = (i - 1) % n
                next_i = (i + 1) % n
                # Circular mean of neighbors
                # Handle angle wrapping
                d_prev = angles[prev_i] - angles[i]
                d_next = angles[next_i] - angles[i]
                # Wrap to [-pi, pi]
                d_prev = (d_prev + np.pi) % (2 * np.pi) - np.pi
                d_next = (d_next + np.pi) % (2 * np.pi) - np.pi
                laplacian = (d_prev + d_next) / 2
                forces[i] -= 0.5 * laplacian
            
            # Update angles with gradient descent (negative of force)
            angles -= lr * forces
            
            # Normalize angles to [0, 2*pi)
            angles = angles % (2 * np.pi)
            
            # Check time
            if time.time() - start_time > time_limit - 60:
                break
        
        return angles

    # Initial Guess: Regular 13-gon
    theta0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    best_score = -1.0
    best_angles = theta0.copy()
    
    # First, apply physics relaxation to get a better starting point
    relaxed_angles = physics_relaxation(theta0, n_steps=300, lr_init=0.02, lr_final=0.002)
    
    # Try multiple starting points
    rng = np.random.default_rng(seed=42)
    starts = [theta0.copy(), relaxed_angles.copy()]
    
    # Add perturbed starts
    for _ in range(3):
        theta_pert = theta0 + rng.normal(0, 0.15, n)
        starts.append(theta_pert.copy())
    
    # Simulated Annealing / Gradient Descent Loop with better cooling schedule
    betas = [5, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    current_angles = relaxed_angles.copy()
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            break
            
        # Try each start (but use current_angles as primary)
        best_candidate = None
        best_candidate_score = -np.inf
        
        for start in starts[:2]:  # Limit to first 2 starts for speed
            if beta > 5:
                x0 = current_angles.copy()
            else:
                x0 = start.copy()
                
            try:
                res = scipy.optimize.minimize(
                    neg_objective_smooth, 
                    x0, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                # Calculate hard metric for this candidate
                cand_points = get_points_from_angles(res.x)
                hard_val = calculate_metric(cand_points, beta=10000)
                
                if hard_val > best_candidate_score:
                    best_candidate_score = hard_val
                    best_candidate = res.x.copy()
                    
            except Exception:
                pass
        
        if best_candidate is not None and best_candidate_score > best_score:
            best_score = best_candidate_score
            best_angles = best_candidate.copy()
            current_angles = best_candidate.copy()
        elif best_candidate is not None:
            # Even if not better, use it as current for next iteration
            current_angles = best_candidate.copy()
            
        # Apply light physics relaxation after each beta level
        if time.time() - start_time < time_limit - 60:
            current_angles = physics_relaxation(current_angles, n_steps=100, lr_init=0.005, lr_final=0.001)
            # Re-evaluate
            cand_points = get_points_from_angles(current_angles)
            hard_val = calculate_metric(cand_points, beta=10000)
            if hard_val > best_score:
                best_score = hard_val
                best_angles = current_angles.copy()

    # Final Polish with Nelder-Mead on the hard objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            def neg_objective_hard(angles):
                points = get_points_from_angles(angles)
                return -calculate_metric(points, beta=10000)
                
            res_final = scipy.optimize.minimize(
                neg_objective_hard, 
                best_angles, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            
            final_hard_val = -res_final.fun
            if final_hard_val > best_score:
                best_score = final_hard_val
                best_angles = res_final.x.copy()
                
        except Exception:
            pass

    final_points = get_points_from_angles(best_angles)
    
    # Ensure validity
    if not np.all(np.isfinite(final_points)):
        # Fallback to regular polygon
        final_points = get_points_from_angles(theta0)

    return final_points

# EVOLVE-BLOCK-END
