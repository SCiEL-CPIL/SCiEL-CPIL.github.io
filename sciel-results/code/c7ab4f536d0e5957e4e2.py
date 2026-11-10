import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 10  # 590 seconds to be safe

    def get_score(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        pts = points.reshape(n, d)
        dists = scipy.spatial.distance.pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize(pts):
        dists = scipy.spatial.distance.pdist(pts)
        d_max = np.max(dists)
        if d_max > 0:
            return pts / d_max
        return pts

    # Generate diverse initial candidates
    candidates = []
    
    # Candidate 1: 4x4 Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    candidates.append(normalize(grid_pts))
    
    # Candidate 2: Regular 16-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(normalize(circle_pts))
    
    # Candidate 3: Hexagonal-like packing
    hex_pts = grid_pts.copy()
    for i in range(1, 4, 2):
        hex_pts[i*4:(i+1)*4, 0] += 0.5
    candidates.append(normalize(hex_pts))
    
    # Candidate 4: Multi-ring (1 center, 5 inner, 10 outer? 1+5+10=16)
    c1 = np.array([0, 0])
    a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    r2 = 1.0
    inner = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
    a3 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    r3 = 2.0
    outer = np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
    ring_pts = np.vstack([c1, inner, outer])
    candidates.append(normalize(ring_pts))
    
    # Add 10 random candidates
    np.random.seed(42)
    for _ in range(10):
        rand_pts = np.random.uniform(-2, 2, (n, d))
        candidates.append(normalize(rand_pts))
    
    best_points = None
    best_score = -np.inf
    
    # Evaluate initial candidates
    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            
    # SCP + Bisection Strategy
    # We want to maximize d_min / d_max.
    # This is equivalent to finding the largest r such that there exist points with d_min >= r and d_max <= 1 (normalized).
    # Or, more generally, for a fixed d_max = 1, maximize d_min.
    # We can use bisection on d_min = r.
    # Feasibility check: Does there exist a configuration of 16 points with d_max <= 1 and d_min >= r?
    # This can be formulated as minimizing d_max subject to d_ij >= r for all i < j.
    # The constraint d_ij >= r is non-convex (union of half-spaces in distance space, but actually it's the outside of a sphere in point space).
    # However, the objective min d_max is convex.
    # Wait, minimizing the maximum distance subject to minimum distance constraints is a classic problem.
    # The feasible region defined by d_ij >= r is non-convex.
    # But we can use a penalty method or barrier method.
    # Alternatively, we can use the fact that for a fixed r, we can try to find a configuration.
    # Let's use a soft constraint approach within an optimization loop.
    
    def scp_objective(x, r_target, mu=100.0):
        """
        Objective function for SCP.
        Minimize d_max + mu * sum(max(0, r_target - d_ij)^2)
        This encourages d_max to be small while penalizing violations of d_min >= r_target.
        """
        points = x.reshape(n, d)
        dists = scipy.spatial.distance.pdist(points)
        d_max = np.max(dists)
        
        # Penalty for distances less than r_target
        violations = np.maximum(0, r_target - dists)
        penalty = np.sum(violations ** 2)
        
        return d_max + mu * penalty

    def scp_grad(x, r_target, mu=100.0):
        """
        Gradient of the SCP objective.
        """
        points = x.reshape(n, d)
        dists = scipy.spatial.distance.pdist(points)
        
        # Gradient of d_max
        # d_max is the max of d_ij. The gradient is the gradient of the max distance pair.
        # There can be multiple pairs. We can use a soft-max gradient or just pick the max.
        # For simplicity and robustness, we'll use a soft-max approximation for the gradient.
        # Or, we can just compute the gradient of the actual max.
        # Let's use the subgradient of the max function.
        # If we use L-BFGS-B, we don't need to provide the gradient explicitly if we don't want to, 
        # but providing it is faster.
        # Let's try to compute the gradient of the penalty term.
        
        # Gradient of penalty:
        # P = sum(max(0, r - d_ij)^2)
        # dP/dx = sum( 2 * max(0, r - d_ij) * (-d d_ij / dx) )
        # d d_ij / dx = (x_i - x_j) / d_ij
        
        grad = np.zeros_like(x)
        points_flat = x.reshape(n, d)
        
        # Compute pairwise differences
        # This is O(n^2 * d)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points_flat[i] - points_flat[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                
                # Contribution to d_max gradient
                # We only need the gradient of the max distance.
                # To be safe and smooth, let's use a soft-max for the d_max part as well.
                # But for now, let's just use the hard max gradient for the d_max part.
                # This might be noisy.
                
                # Let's stick to a simpler approach: use numerical gradient or a simpler objective.
                # Actually, L-BFGS-B can handle non-smooth functions reasonably well with finite differences,
                # but it's slow.
                
                # Let's use a differentiable objective:
                # Minimize log-sum-exp of distances (approximates max) + penalty for min.
                # This is fully differentiable.
                
                pass # We will rewrite the objective to be fully differentiable.
                
        # Let's rewrite the SCP step to use a differentiable objective.
        # Objective: f(x) = logsumexp(beta * dists) / beta + mu * sum(max(0, r - dists)^2)
        # This is differentiable.
        
        return None # Not used, we will use a different function.

    def differentiable_scp_objective(x, r_target, beta=20.0, mu=100.0):
        """
        Differentiable objective for SCP.
        f(x) = soft_max(dists) + mu * sum(max(0, r_target - dists)^2)
        """
        points = x.reshape(n, d)
        dists = scipy.spatial.distance.pdist(points)
        
        # Soft max
        # To avoid overflow, subtract max
        max_d = np.max(dists)
        if max_d < 1e-12:
            return 0.0
        soft_max = max_d + (1.0 / beta) * np.log(np.sum(np.exp(beta * (dists - max_d))))
        
        # Penalty
        violations = np.maximum(0, r_target - dists)
        penalty = np.sum(violations ** 2)
        
        return soft_max + mu * penalty

    # Bisection on r_target
    # We know r_target is in [0, 1] (since d_max is normalized to 1, d_min <= 1)
    # Actually, d_min can be slightly larger than 1 if d_max is larger, but we normalize d_max=1.
    # So we are looking for the max r such that there exists a config with d_max <= 1 and d_min >= r.
    
    r_low = 0.0
    r_high = 1.0
    
    # Initial guess for r_high: use the best score found so far.
    # best_score = (d_min/d_max)^2. So d_min/d_max = sqrt(best_score).
    # If d_max=1, then d_min = sqrt(best_score).
    if best_score > 0:
        r_high = np.sqrt(best_score) + 0.1 # Add a margin
    
    # Bisection loop
    max_bisection_iters = 10
    for iter_b in range(max_bisection_iters):
        if time.time() > deadline - 30:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a configuration with d_min >= r_mid and d_max <= 1.
        # We minimize the differentiable_scp_objective.
        # If the minimum value is close to 1 (d_max) and penalty is 0, it's feasible.
        
        # We start from the best known configuration, or a random one.
        # Let's start from the best known configuration.
        x0 = best_points.flatten() if best_points is not None else np.random.uniform(-1, 1, n*d)
        
        try:
            res = scipy.optimize.minimize(
                differentiable_scp_objective,
                x0,
                args=(r_mid, 20.0, 100.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                pts = res.x.reshape(n, d)
                dists = scipy.spatial.distance.pdist(pts)
                d_min = np.min(dists)
                d_max = np.max(dists)
                
                # Check feasibility
                # If d_min >= r_mid and d_max is not too large, it's feasible.
                # Actually, we just want to know if we can achieve d_min >= r_mid.
                # If the optimizer found a point with d_min >= r_mid, then r_mid is feasible.
                # We can check the actual d_min of the result.
                
                if d_min >= r_mid - 1e-6: # Feasible
                    r_low = r_mid
                    # Update best points if this configuration is better
                    if d_max > 0:
                        score = (d_min / d_max) ** 2
                        if score > best_score:
                            best_score = score
                            best_points = pts.copy()
                else:
                    r_high = r_mid
                    
        except Exception:
            pass

    # Final polish: Use the best r_low to do a final optimization without penalty, just maximizing d_min/d_max.
    if best_points is not None and time.time() < deadline - 10:
        # We can try to improve the best point found using the original smoothed objective.
        def smoothed_objective(x, beta=50.0):
            points = x.reshape(n, d)
            dists = scipy.spatial.distance.pdist(points)
            max_d = np.max(dists)
            if max_d < 1e-12:
                return 0.0
            term_min = np.exp(beta * (max_d - dists))
            d_min_soft = max_d - (1.0 / beta) * np.log(np.sum(term_min))
            term_max = np.exp(beta * (dists - max_d))
            d_max_soft = max_d + (1.0 / beta) * np.log(np.sum(term_max))
            if d_max_soft == 0:
                return 0.0
            return -(d_min_soft / d_max_soft)

        try:
            res_final = scipy.optimize.minimize(
                smoothed_objective,
                best_points.flatten(),
                args=(100.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res_final.success:
                final_pts = res_final.x.reshape(n, d)
                final_score = get_score(final_pts)
                if final_score > best_score:
                    best_points = final_pts
        except Exception:
            pass

    if best_points is None:
        best_points = candidates[0]
        
    return best_points
