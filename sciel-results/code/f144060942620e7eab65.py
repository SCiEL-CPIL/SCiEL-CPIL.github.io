import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = 8
    iu = np.triu_indices(n, k=1)
    num_pairs = len(iu[0])

    def calc_diam(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)

    def log_score(pts):
        """
        Returns log of the product of pairwise distances (sum over i<j).
        """
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        # Avoid log(0)
        d_ij = np.maximum(d_ij, 1e-15)
        return np.sum(np.log(d_ij))

    def compute_score(pts):
        """
        Compute the actual score Delta/n^n.
        Delta is product of ordered pairs distances after scaling to diameter 2.
        """
        pts = np.array(pts, dtype=float)
        if pts.shape != (n, 2):
            return -np.inf
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        
        if np.any(d < 1e-9):
            return -np.inf
            
        diam = np.max(d)
        if diam < 1e-9:
            return -np.inf
        
        pts_scaled = pts * (2.0 / diam)
        
        diff_s = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
        dists_s = np.sqrt(np.sum(diff_s**2, axis=2))
        d_s = dists_s[mask]
        
        log_delta = np.sum(np.log(d_s))
        delta = np.exp(log_delta)
        score = delta / (n ** n)
        
        return score

    # --- Objective and Gradient for L-BFGS-B ---
    # We maximize log(product) subject to diameter <= 2.
    # Using penalty method: minimize -log_score + lambda * penalty(diam > 2)
    
    def objective_and_grad(x, lambda_pen):
        """
        Objective to minimize: -log(product) + penalty
        """
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        
        # Log score part
        # Sum of log(d_ij)
        # We want to maximize this, so minimize -Sum(log(d_ij))
        
        # Gradient of log(d_ij) wrt x:
        # d/dx_i log(||x_i - x_j||) = (x_i - x_j) / ||x_i - x_j||^2
        # d/dx_j log(||x_i - x_j||) = (x_j - x_i) / ||x_i - x_j||^2 = -(x_i - x_j) / ||x_i - x_j||^2
        
        grad = np.zeros_like(pts)
        val_log = 0.0
        
        # Penalty part
        # P = sum( max(0, d_ij - 2)^2 )
        # dP/dx_i = 2 * max(0, d_ij - 2) * (d_ij - 2 < 2 ? 1 : 0) * d(d_ij)/dx_i
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        
        val_pen = 0.0
        grad_pen = np.zeros_like(pts)
        
        for idx in range(num_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = d_ij[idx]
            if d < 1e-15:
                continue
            
            diff = pts[i] - pts[j]
            # grad log
            g = diff / (d * d)
            grad[i] += g
            grad[j] -= g
            val_log += np.log(d)
            
            # penalty
            if d > 2.0:
                excess = d - 2.0
                # dP = 2 * excess * d(d)
                # d(d) = diff / d
                g_pen = 2.0 * excess * (diff / d)
                grad_pen[i] += g_pen
                grad_pen[j] -= g_pen
                val_pen += excess * excess

        total_val = -val_log + lambda_pen * val_pen
        total_grad = -grad + lambda_pen * grad_pen
        
        return total_val, total_grad

    def run_optimization(x0, lambda_start=1.0, lambda_end=1000.0, steps=10):
        """
        Iteratively increase penalty weight to enforce constraint.
        """
        current_x = x0.copy()
        
        # Geometric progression for lambda
        lambdas = np.geomspace(lambda_start, lambda_end, steps)
        
        for lam in lambdas:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    lambda v: objective_and_grad(v, lam)[0],
                    current_x,
                    jac=lambda v: objective_and_grad(v, lam)[1].flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_x = res.x
            except Exception:
                pass
                
        return current_x

    def evaluate(pts):
        d = calc_diam(pts)
        if d > 2.0 + 1e-6:
            return -np.inf
        # Scale to diameter 2 if slightly off, or just use as is if <= 2
        # The score definition says "after scaling the set to diameter 2".
        # So we must scale.
        if d < 1e-10:
            return -np.inf
        pts_scaled = pts * (2.0 / d)
        return log_score(pts_scaled)

    best_score = -np.inf
    best_pts = None
    
    def update_best(pts):
        nonlocal best_score, best_pts
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # --- 1. Regular Octagon ---
    theta = 2 * np.pi * np.arange(n) / n
    octagon = np.column_stack([np.cos(theta), np.sin(theta)])
    # Diameter is 2.
    update_best(octagon)
    
    # --- 2. Cage Topology: Two Concentric Squares ---
    # Parameterization: R1, R2, theta2 (rotation of square 2 relative to square 1)
    # Square 1 at angles 0, pi/2, pi, 3pi/2 with radius R1
    # Square 2 at angles theta2 + 0, theta2 + pi/2, ... with radius R2
    
    def cage_points(params):
        R1, R2, theta2 = params
        # Ensure positive radii
        R1 = max(1e-6, R1)
        R2 = max(1e-6, R2)
        
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + theta2
        
        p1 = np.column_stack([R1 * np.cos(angles1), R1 * np.sin(angles1)])
        p2 = np.column_stack([R2 * np.cos(angles2), R2 * np.sin(angles2)])
        
        return np.vstack([p1, p2])
    
    def cage_objective(params):
        pts = cage_points(params)
        # We want to maximize log score, so minimize negative
        # We don't strictly enforce diameter <= 2 in objective here, 
        # but we check in evaluation. 
        # For optimization, we can just maximize log score, and let the 
        # scaling handle the normalization? 
        # No, the diameter constraint is structural for the "cage" shape 
        # if we want to compare fairly? 
        # Actually, the score is scale invariant. 
        # So we can just maximize log_score(pts) without diameter constraint 
        # in the objective, because scaling doesn't change the relative 
        # geometry or the final normalized score?
        # Wait. 
        # Score = Delta / n^n. Delta is product of distances after scaling to diam=2.
        # If we scale pts by k, distances scale by k. 
        # Product scales by k^(n(n-1)).
        # If we scale to diam=2, the factor is fixed.
        # So maximizing log_score(pts) is equivalent to maximizing the final score,
        # regardless of the initial scale, AS LONG AS we can scale to diam=2.
        # Since any set of distinct points has a diameter, we can always scale.
        # So we can just optimize the shape (parameters) to maximize log_score.
        
        return -log_score(pts)

    # Grid search over cage parameters
    # R1 in [0.5, 1.5], R2 in [0.1, 1.5], theta2 in [0, pi/2]
    # Since the shape is what matters, we can fix R1=1 and optimize R2 and theta2?
    # No, the relative size matters. But scale invariance means (R1, R2) is equivalent to (1, R2/R1).
    # So we can fix R1=1 and optimize R2 (ratio) and theta2.
    
    # Let's do a coarse grid search first
    grid_results = []
    r2_vals = np.linspace(0.1, 1.5, 15)
    theta_vals = np.linspace(0, np.pi/2, 15)
    
    for r2 in r2_vals:
        for th in theta_vals:
            if time.time() > deadline - 20:
                break
            params = np.array([1.0, r2, th])
            pts = cage_points(params)
            # Quick check for distinctness
            d = calc_diam(pts)
            if d < 1e-6:
                continue
            # Scale to diam 2 for evaluation consistency? 
            # No, just evaluate log score.
            score = evaluate(pts)
            grid_results.append((score, params))
    
    if grid_results:
        grid_results.sort(key=lambda x: x[0], reverse=True)
        # Take top few for refinement
        for score, params in grid_results[:5]:
            if time.time() > deadline - 10:
                break
            # Refine with L-BFGS-B
            # We need to optimize in 3D space.
            # But L-BFGS-B expects 1D vector.
            # We can wrap it.
            
            def cage_opt_obj(v):
                return cage_objective(v)
            
            def cage_opt_grad(v):
                # Numerical gradient or analytical?
                # For 3 params, numerical is fast enough and robust.
                eps = 1e-8
                grad = np.zeros(3)
                f0 = cage_objective(v)
                for i in range(3):
                    v_plus = v.copy()
                    v_plus[i] += eps
                    f_plus = cage_objective(v_plus)
                    grad[i] = (f_plus - f0) / eps
                return grad
            
            try:
                res = minimize(cage_opt_obj, params, jac=cage_opt_grad, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12})
                best_cage_params = res.x
                best_cage_pts = cage_points(best_cage_params)
                update_best(best_cage_pts)
            except Exception:
                pass

    # --- 3. Differential Evolution over full 16D space ---
    # This is the heavy hitter. 
    # We need to map 16D to points, ensuring distinctness and reasonable bounds.
    # Bounds: x, y in [-2, 2] is sufficient for diameter <= 4, then we scale.
    # Actually, since we scale to diameter 2, the absolute scale doesn't matter.
    # But DE works better with bounded search.
    # Let's use bounds [-2, 2] for each coordinate.
    
    bounds = [(-2, 2)] * (n * 2)
    
    def de_objective(x):
        pts = x.reshape(n, 2)
        # Check distinctness
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        if np.any(d < 1e-6):
            return 1e10 # Penalty for collision
        # Maximize log score -> Minimize negative
        return -log_score(pts)
    
    # Run DE if time permits
    if time.time() < deadline - 30:
        try:
            de_res = differential_evolution(
                de_objective, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=1e-4, 
                update_interval=10,
                polish=True,
                seed=42
            )
            if de_res.success or de_res.fun < 1e9:
                de_pts = de_res.x.reshape(n, 2)
                update_best(de_pts)
        except Exception:
            pass

    # --- 4. Final Polish ---
    if best_pts is not None:
        # Run a final L-BFGS-B refinement on the best points found
        # This helps if DE or Cage search landed near a local optimum but not exactly on it
        if time.time() < deadline - 5:
            final_x = run_optimization(best_pts.flatten(), lambda_start=100.0, lambda_end=10000.0, steps=5)
            final_pts = final_x.reshape(n, 2)
            update_best(final_pts)

    # Ensure distinctness and bounds
    if best_pts is None:
        best_pts = octagon

    # Scale to diameter 2
    d = calc_diam(best_pts)
    if d > 1e-10:
        best_pts = best_pts * (2.0 / d)

    # Check distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-8:
                best_pts[j] += np.array([1e-6, 1e-6])

    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e5:
        best_pts = best_pts * (1e5 / max_coord)
        d = calc_diam(best_pts)
        if d > 1e-10:
            best_pts = best_pts * (2.0 / d)

    return [[float(p[0]), float(p[1])] for p in best_pts]
