import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n = 8
    # Regular octagon as fallback
    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def score(points):
        # points: (n, 2)
        # Scale to diameter 2
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D == 0:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        # Product of distances for i < j
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    start_time = time.time()
    timeout = 135  # seconds, keep 15s margin for safety
    
    best_score = score(regular_octagon())
    best_points = regular_octagon()
    
    # Parametric optimization:
    # Fix p0 = (-1, 0), p1 = (1, 0) as the diameter pair.
    # Optimize p2..p7.
    # Constraints: d(pi, pj) <= 2 for all i,j.
    
    def objective(x, penalty_weight=1e4):
        # x: (12,) representing p2..p7
        pts = np.zeros((n, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(n-2, 2)
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Check validity (distinct points)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        min_d = np.min(dists[mask])
        if min_d < 1e-8:
            return 1e10
        
        # Log sum of distances
        log_sum = np.sum(np.log(np.maximum(dists[mask], 1e-15)))
        
        # Penalty for constraint violations (smooth)
        # For each pair, if distance > 2, penalize (d-2)^2
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d > 2.0:
                    penalty += (d - 2.0) ** 2
        
        return -log_sum + penalty_weight * penalty

    def objective_no_penalty(x):
        return objective(x, penalty_weight=0.0)

    # Constraints for SLSQP: g(x) >= 0
    # We want d(pi, pj) <= 2, so 2 - d(pi, pj) >= 0
    def make_constraint(i, j):
        def constraint(x):
            pts = np.zeros((n, 2))
            pts[0] = [-1.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = x.reshape(n-2, 2)
            d = np.linalg.norm(pts[i] - pts[j])
            return 2.0 - d
        return constraint

    constraints = []
    # Pairs among 2..7
    for i in range(2, n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    # Pairs between 0 and 2..7
    for i in range(2, n):
        constraints.append({'type': 'ineq', 'fun': make_constraint(0, i)})
    # Pairs between 1 and 2..7
    for i in range(2, n):
        constraints.append({'type': 'ineq', 'fun': make_constraint(1, i)})
    
    # Bounds for x (coordinates of p2..p7)
    # They must be within distance 2 of p0 and p1.
    # x in [-3, 3], y in [-3, 3] is a safe box.
    bounds = [(-3.0, 3.0)] * (2 * (n-2))
    
    def evaluate_and_update(pts):
        nonlocal best_score, best_points
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D > 2.0 + 1e-6:
            return False
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if np.min(dists[mask]) < 1e-6:
            return False
        cand_score = score(pts)
        if cand_score > best_score:
            best_score = cand_score
            best_points = pts.copy()
            return True
        return False

    # Initial guesses
    # 1. Regular octagon perturbed to match p0, p1
    oct_pts = regular_octagon()
    # In np.linspace(0, 2pi, 8, endpoint=False):
    # 0: (1,0), 1: (0.707, 0.707), 2: (0, 1), 3: (-0.707, 0.707),
    # 4: (-1, 0), 5: (-0.707, -0.707), 6: (0, -1), 7: (0.707, -0.707)
    # p0 = (-1,0) = index 4, p1 = (1,0) = index 0
    # Remaining: indices 1,2,3,5,6,7
    init_x = np.array([
        oct_pts[1][0], oct_pts[1][1],
        oct_pts[2][0], oct_pts[2][1],
        oct_pts[3][0], oct_pts[3][1],
        oct_pts[5][0], oct_pts[5][1],
        oct_pts[6][0], oct_pts[6][1],
        oct_pts[7][0], oct_pts[7][1]
    ])
    
    # Run SLSQP with penalty-based objective first (smoother)
    try:
        res = minimize(
            lambda x: objective(x, penalty_weight=1e4),
            init_x, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        
        if time.time() - start_time < timeout and res.success:
            pts = np.zeros((n, 2))
            pts[0] = [-1.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = res.x.reshape(n-2, 2)
            evaluate_and_update(pts)
            
    except Exception:
        pass

    # Try random initializations
    if time.time() - start_time < timeout:
        np.random.seed(42)
        for trial in range(15):
            if time.time() - start_time > timeout:
                break
                
            pts = np.zeros((n, 2))
            pts[0] = [-1.0, 0.0]
            pts[1] = [1.0, 0.0]
            for i in range(2, n):
                max_attempts = 1000
                for attempt in range(max_attempts):
                    p = np.random.uniform(-2.5, 2.5, 2)
                    if (np.linalg.norm(p - pts[0]) <= 2.0 and 
                        np.linalg.norm(p - pts[1]) <= 2.0 and
                        all(np.linalg.norm(p - pts[j]) > 0.1 for j in range(2, i))):
                        pts[i] = p
                        break
                else:
                    # Fallback: place on boundary
                    angle = np.random.uniform(0, 2*np.pi)
                    pts[i] = [np.cos(angle), np.sin(angle)]
            
            x0 = pts[2:].flatten()
            
            try:
                # Use penalty-based objective for robustness
                res = minimize(
                    lambda x: objective(x, penalty_weight=1e4),
                    x0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-10}
                )
                if time.time() - start_time < timeout and res.success:
                    pts[2:] = res.x.reshape(n-2, 2)
                    evaluate_and_update(pts)
            except Exception:
                pass

    # Final high-precision polish pass using the best solution found
    if time.time() - start_time < timeout - 5:
        # Extract current best points
        best_x = best_points[2:].flatten()
        
        # Polish with very tight tolerance and no penalty (assuming constraints are satisfied)
        try:
            res_polish = minimize(
                objective_no_penalty,
                best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-14, 'maxls': 100}
            )
            if res_polish.success:
                pts_polish = np.zeros((n, 2))
                pts_polish[0] = [-1.0, 0.0]
                pts_polish[1] = [1.0, 0.0]
                pts_polish[2:] = res_polish.x.reshape(n-2, 2)
                evaluate_and_update(pts_polish)
        except Exception:
            pass
        
        # Additional polish with L-BFGS-B using penalty (unconstrained)
        if time.time() - start_time < timeout - 2:
            try:
                res_lbfgs = minimize(
                    lambda x: objective(x, penalty_weight=1e6),
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                pts_lbfgs = np.zeros((n, 2))
                pts_lbfgs[0] = [-1.0, 0.0]
                pts_lbfgs[1] = [1.0, 0.0]
                pts_lbfgs[2:] = res_lbfgs.x.reshape(n-2, 2)
                
                # Check if constraints are satisfied
                diffs = pts_lbfgs[:, np.newaxis, :] - pts_lbfgs[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                if np.max(dists) <= 2.0 + 1e-8:
                    evaluate_and_update(pts_lbfgs)
            except Exception:
                pass

    # Convex polygon parametrization: 8 points on a convex hull with diameter 2
    # Parametrize by 8 angles theta_i in [0, 2pi) sorted, and radii r_i in (0, 1]
    # Actually, for a convex polygon with diameter 2, we can use the fact that
    # all points lie within the unit disk (if we scale so diameter=2, the points
    # fit in a circle of radius 1... wait, no. Diameter 2 means the max distance is 2.
    # The points can be anywhere as long as max distance is 2.
    
    # Better approach: use the fact that for a set with diameter 2, all points lie
    # in the intersection of disks of radius 2 centered at each other point.
    # But a simpler parametrization: place points on the boundary of the convex hull.
    # For a convex polygon with n vertices and diameter D, we can parametrize by
    # the angles of the vertices from the centroid, but this doesn't guarantee diameter D.
    
    # Alternative: use the "diameter pair" approach but with a different constraint handling.
    # Since we already have a good solution, let's try to refine it using a different
    # local optimizer or a different parametrization.
    
    # Let's try: fix the diameter pair p0=(-1,0), p1=(1,0), and parametrize the other
    # 6 points as (x_i, y_i) where we use the constraint that each point is within
    # distance 2 of both p0 and p1. This means each point lies in the intersection
    # of two disks of radius 2 centered at (-1,0) and (1,0).
    # The intersection is a lens shape. We can parametrize points in this lens using
    # polar coordinates from the origin: (r*cos(theta), r*sin(theta)) where r is in [0, r_max(theta)]
    # and r_max(theta) = min(sqrt(4 - (r*cos(theta)+1)^2 - (r*sin(theta))^2), ...)
    # This is getting complicated. Let's stick with the current approach but improve the search.
    
    # Try a few more random starts with different seeds
    if time.time() - start_time < timeout - 10:
        for seed in [123, 456, 789, 101112]:
            if time.time() - start_time > timeout - 10:
                break
            np.random.seed(seed)
            for trial in range(5):
                if time.time() - start_time > timeout - 10:
                    break
                pts = np.zeros((n, 2))
                pts[0] = [-1.0, 0.0]
                pts[1] = [1.0, 0.0]
                for i in range(2, n):
                    max_attempts = 500
                    for attempt in range(max_attempts):
                        p = np.random.uniform(-2.0, 2.0, 2)
                        if (np.linalg.norm(p - pts[0]) <= 2.0 and 
                            np.linalg.norm(p - pts[1]) <= 2.0 and
                            all(np.linalg.norm(p - pts[j]) > 0.05 for j in range(2, i))):
                            pts[i] = p
                            break
                    else:
                        angle = np.random.uniform(0, 2*np.pi)
                        pts[i] = [np.cos(angle), np.sin(angle)]
                
                x0 = pts[2:].flatten()
                try:
                    res = minimize(
                        lambda x: objective(x, penalty_weight=1e4),
                        x0, 
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=constraints,
                        options={'maxiter': 200, 'ftol': 1e-10}
                    )
                    if time.time() - start_time < timeout - 10 and res.success:
                        pts[2:] = res.x.reshape(n-2, 2)
                        evaluate_and_update(pts)
                except Exception:
                    pass
    
    return best_points.tolist()
