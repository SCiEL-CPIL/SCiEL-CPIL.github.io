import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    n = 8
    
    # Precompute constants
    log_n_n = n * math.log(n)
    # Score = (Product of 56 distances) / (Diameter^56 * n^8)
    # We scale points to diameter 1 first, then compute product.
    # Actually, the task says: "scale the set to diameter 2".
    # Let D be the diameter of the generated set.
    # Scaled points P' = 2 * P / D.
    # Distance d'_ij = 2 * d_ij / D.
    # Product Delta' = (2/D)^56 * Product(d_ij).
    # Score = Delta' / n^8 = (2^56 / (D^56 * n^8)) * Product(d_ij).
    # Log Score = 56*log(2) - 56*log(D) - log(n^8) + Sum(log(d_ij)).
    
    def get_score_and_points(angles):
        """
        angles: array of 8 angles.
        Returns: (log_score, points_array)
        Points are generated on unit circle first.
        """
        # Generate points on unit circle
        x = np.cos(angles)
        y = np.sin(angles)
        points = np.column_stack((x, y))
        
        # Compute pairwise distances
        # We need all 56 unique distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                dists[i, j] = d
                dists[j, i] = d
                if d < 1e-12:
                    return -np.inf, points
        
        D = np.max(dists)
        if D < 1e-12:
            return -np.inf, points
            
        # Calculate log of product of all 56 distances
        # We only summed unique pairs in the loop, so we need to double the sum for ordered pairs?
        # Task says: "product over ordered pairs".
        # Ordered pairs (i,j) where i!=j. There are n*(n-1) = 56 such pairs.
        # My loop calculated n*(n-1)/2 = 28 unique distances.
        # So Product_ordered = (Product_unique)^2.
        # Log(Product_ordered) = 2 * Sum(log(unique_dists))
        
        mask = ~np.eye(n, dtype=bool)
        valid_dists = dists[mask]
        
        if np.any(valid_dists < 1e-12):
            return -np.inf, points
            
        log_prod_ordered = 2.0 * np.sum(np.log(valid_dists))
        
        log_score = 56 * math.log(2) - 56 * math.log(D) - log_n_n + log_prod_ordered
        
        return log_score, points

    def objective(angles):
        """
        Returns negative log score for minimization.
        Includes penalty for non-distinct points.
        """
        if time.time() > deadline:
            return 1e10
            
        # Ensure angles are in [0, 2pi) for stability, though cos/sin are periodic
        # angles = np.mod(angles, 2 * math.pi)
        
        log_score, _ = get_score_and_points(angles)
        
        if not np.isfinite(log_score):
            return 1e10
            
        return -log_score

    best_log_score = -np.inf
    best_angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    
    # Multi-start Simulated Annealing
    # We will run several independent SA chains.
    
    num_starts = 10
    # Dynamic budget per start
    # Total time available ~ 145s. 
    # Each SA step is cheap (numpy ops on 8 points).
    # Let's estimate: 1 SA step ~ 10-50 microseconds? 
    # Actually, distance calculation is 28 norms. Very fast.
    # We can afford ~100,000 - 1,000,000 steps total.
    
    steps_per_start = 50000
    # Initial temperature
    T_init = 0.1
    # Cooling rate
    alpha = 0.999
    
    for k in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Initialize angles
        # Start 0: Regular octagon
        # Start 1+: Random perturbations or random starts
        if k == 0:
            current_angles = best_angles.copy()
        else:
            # Random start, but ensure distinctness
            current_angles = np.sort(np.random.uniform(0, 2 * math.pi, n))
            # Check distinctness
            diffs = np.diff(np.concatenate([current_angles, [current_angles[0] + 2*math.pi]]))
            if np.min(diffs) < 0.1:
                # Too close, re-roll
                current_angles = np.random.uniform(0, 2 * math.pi, n)
                current_angles.sort()
        
        current_log_score, _ = get_score_and_points(current_angles)
        if not np.isfinite(current_log_score):
            continue
            
        current_obj = -current_log_score
        
        best_local_log_score = current_log_score
        best_local_angles = current_angles.copy()
        
        T = T_init
        for step in range(steps_per_start):
            if time.time() > deadline - 10:
                break
                
            # Generate neighbor
            # Move one random angle by a small amount
            idx = np.random.randint(n)
            delta = np.random.normal(0, 0.1)
            neighbor_angles = current_angles.copy()
            neighbor_angles[idx] += delta
            
            neighbor_log_score, _ = get_score_and_points(neighbor_angles)
            if not np.isfinite(neighbor_log_score):
                # Invalid point (collision), reject
                T *= alpha
                continue
                
            neighbor_obj = -neighbor_log_score
            
            # Acceptance probability
            if neighbor_obj < current_obj:
                accept = True
            else:
                delta_obj = neighbor_obj - current_obj
                if delta_obj == 0:
                    accept = True
                else:
                    # Metropolis criterion
                    accept = (np.random.uniform() < math.exp(-delta_obj / T))
                    
            if accept:
                current_angles = neighbor_angles
                current_log_score = neighbor_log_score
                current_obj = neighbor_obj
                
                if current_log_score > best_local_log_score:
                    best_local_log_score = current_log_score
                    best_local_angles = current_angles.copy()
                    
            T *= alpha
            
        # Update global best
        if best_local_log_score > best_log_score:
            best_log_score = best_local_log_score
            best_angles = best_local_angles
    
    # Refinement: Nelder-Mead on the best found angles
    # Since the landscape is smooth, NM can polish the result.
    # We need to convert angles to a flat array for scipy.optimize.minimize
    # But wait, the objective function takes angles directly.
    
    # Let's use scipy.optimize.minimize with Nelder-Mead
    # We must be careful with the periodic boundary.
    # However, for a local refinement around a good point, it's usually safe.
    
    try:
        from scipy.optimize import minimize
        
        # Define wrapper for minimize
        def obj_for_minimize(angles_flat):
            # angles_flat is 1D array of 8 floats
            if time.time() > deadline - 5:
                return 1e10
            return objective(angles_flat)
        
        # Initial guess
        x0 = best_angles.copy()
        
        res = minimize(
            obj_for_minimize, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if res.fun < -best_log_score:
            best_log_score = -res.fun
            best_angles = res.x
            
    except Exception:
        pass
    
    # Final verification and scaling
    # The task requires returning points [x, y].
    # The score is calculated by the evaluator after scaling to diameter 2.
    # My internal calculation assumed scaling to diameter 2 for the score formula.
    # Let's double check.
    # Task: "score is Delta/n^n, where Delta is the product over ordered pairs of distances after scaling the set to diameter 2"
    # My formula:
    # Let P be points on unit circle. Diameter D_P.
    # Scaled points P' = 2 * P / D_P.
    # Distances d' = 2 * d / D_P.
    # Delta = Product(d') = (2/D_P)^56 * Product(d).
    # Score = Delta / n^8.
    # This matches my `get_score_and_points` logic.
    
    # So, I just need to return the points from `best_angles`.
    # Should I return them on the unit circle, or scaled?
    # The evaluator scales them. So I can return them on the unit circle.
    # However, to be safe and avoid any floating point issues with the evaluator's scaling,
    # I will return them on the unit circle.
    
    x = np.cos(best_angles)
    y = np.sin(best_angles)
    points = np.column_stack((x, y))
    
    # Ensure distinctness
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                # Perturb
                points[j] += np.random.normal(0, 1e-5, 2)
                # Re-normalize to unit circle? 
                # The task doesn't require unit circle, just distinct points.
                # But keeping them on the circle is good for the "angular" interpretation.
                # If I perturb, they leave the circle.
                # Better to just ensure angles are distinct.
                pass
    
    # Actually, if I perturb Cartesian coords, I break the "on unit circle" assumption 
    # used in my score calculation, but the evaluator recalculates score from scratch.
    # So it's fine. But distinctness is key.
    
    return [[float(p[0]), float(p[1])] for p in points]
