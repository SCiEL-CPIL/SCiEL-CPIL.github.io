# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a two-stage optimization:
    1. Smooth surrogate optimization (sum of inverse squared distances) with L-BFGS-B.
    2. Exact metric polish with Nelder-Mead.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580.0  # 20s margin

    def _objective_exact(flat_points):
        """
        Objective to minimize: (d_max / d_min)^2
        Equivalent to maximizing (d_min / d_max)^2
        """
        points = flat_points.reshape(n, d)
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max <= 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 1e10
        return (d_max / d_min) ** 2

    def _objective_smooth(flat_points):
        """
        Smooth surrogate: sum of 1/d_ij^2
        Minimizing this tends to maximize d_min.
        We also need to control scale. We normalize points to have unit norm sum or similar?
        Actually, the ratio is scale invariant. The smooth objective is NOT scale invariant.
        If we scale points by k, d_ij scales by k, 1/d_ij^2 scales by 1/k^2.
        So we must fix the scale.
        We can add a penalty for deviation from a fixed scale, e.g., sum(||p_i||^2) = n.
        Or simply normalize the points after each step? 
        If we normalize inside the objective, the gradient is complicated.
        Better: Constrain the optimization to the manifold sum(||p_i||^2) = n?
        L-BFGS-B doesn't handle equality constraints well (only bounds).
        
        Alternative: 
        Minimize sum(1/d_ij^2) subject to scale.
        We can just normalize the points in the wrapper? No, the optimizer needs a smooth landscape.
        
        Let's try a different smooth objective that is scale-invariant?
        No, let's just fix the scale by normalizing the vector of all coordinates to have a fixed norm.
        Or, simpler: Optimize on the sphere? 
        The current code optimizes on the unit sphere. 
        Is the optimal solution on the unit sphere? 
        For Tammes problem (maximize min distance on sphere), yes.
        For max(min/max) in free space, the points will likely form a shape that is "spread out". 
        If we allow free space, we can always scale. 
        Let's assume the optimal shape is similar to the spherical code.
        Let's stick to the sphere constraint for the first stage, as it's simpler and the current code does it.
        If we optimize on the sphere, d_max is bounded by 2 (diameter).
        
        Let's define the smooth objective on the sphere.
        We can use the same repulsion force idea but use L-BFGS-B? 
        L-BFGS-B works on Euclidean space. 
        We can parameterize the sphere using angles (spherical coordinates) or quaternions? 
        Or we can just use the Euclidean coordinates and add a large penalty for deviation from the unit sphere.
        
        Penalty method:
        J_smooth(p) = sum(1/d_ij^2) + lambda * sum(||p_i||^2 - 1)^2
        
        This is smooth and differentiable.
        """
        points = flat_points.reshape(n, d)
        
        # Distance matrix
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero
        
        # Avoid division by zero for very close points
        dists_sq = np.maximum(dists_sq, 1e-10)
        
        # Sum of inverse squared distances
        # Exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        inv_d_sq_sum = np.sum(1.0 / dists_sq[mask])
        
        # Penalty for sphere constraint
        norms_sq = np.sum(points**2, axis=1)
        sphere_penalty = np.sum((norms_sq - 1.0)**2)
        
        return inv_d_sq_sum + 100.0 * sphere_penalty

    def _evaluate_metric(points):
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max <= 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Generate initial guesses
    # 1. Fibonacci sphere
    def fibonacci_sphere(n):
        points = np.zeros((n, d))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - (2 * i + 1) / n
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / golden_ratio
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
            points[i, 2] = z
        return points

    # 2. Random points on sphere
    np.random.seed(42)
    def random_sphere(n):
        points = np.random.randn(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

    initial_guesses = [fibonacci_sphere(n)]
    for _ in range(5):
        initial_guesses.append(random_sphere(n))

    best_points = None
    best_score = -1.0

    for guess in initial_guesses:
        if time.time() > deadline - 30.0:
            break
            
        flat_guess = guess.flatten()
        
        # Stage 1: L-BFGS-B on smooth surrogate
        try:
            res = minimize(
                _objective_smooth, 
                flat_guess, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            optimized_flat = res.x
            optimized_points = optimized_flat.reshape(n, d)
            
            # Normalize to sphere to remove penalty drift
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            optimized_points = optimized_points / norms
            
        except Exception:
            optimized_points = guess

        # Stage 2: Nelder-Mead on exact metric
        # Nelder-Mead is derivative-free, good for non-smooth
        # It can be slow, so we limit iterations
        try:
            res2 = minimize(
                _objective_exact, 
                optimized_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            polished_flat = res2.x
            polished_points = polished_flat.reshape(n, d)
            
            # Check if it's still on the sphere (Nelder-Mead doesn't know about the constraint)
            # If it drifted, project back to sphere? 
            # The exact metric is scale invariant, so drifting off sphere is fine as long as we evaluate correctly.
            # However, if points collapse or go to infinity, it's bad.
            # We should probably keep them on the sphere or at least bounded.
            # Let's project to sphere if they are close, or just use them as is if the score is good.
            
            score = _evaluate_metric(polished_points)
            
            # Compare with the pre-polish score
            score_pre = _evaluate_metric(optimized_points)
            if score > score_pre:
                current_best = polished_points
            else:
                current_best = optimized_points
                
            if _evaluate_metric(current_best) > best_score:
                best_score = _evaluate_metric(current_best)
                best_points = current_best
                
        except Exception:
            score = _evaluate_metric(optimized_points)
            if score > best_score:
                best_score = score
                best_points = optimized_points

    if best_points is None:
        # Fallback
        best_points = fibonacci_sphere(n)

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
