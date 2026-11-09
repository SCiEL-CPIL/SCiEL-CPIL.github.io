import numpy as np
from scipy.optimize import differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8

    def compute_score(points):
        # points is shape (n, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        mask = ~np.eye(n, dtype=bool)
        max_dist = np.max(dists[mask])
        
        if max_dist == 0:
            return -np.inf
            
        log_prod_ordered = np.sum(np.log(dists[mask]))
        
        # Objective to maximize: log(product) - 56 * log(diameter)
        return log_prod_ordered - 56 * np.log(max_dist)

    def objective_full(points):
        s = compute_score(points)
        if not np.isfinite(s):
            return 1e10
        return -s

    # Strategy: Exploit D4 symmetry.
    # The optimal configuration for n=8 is expected to be two concentric squares.
    # Let the outer square have vertices at radius R1 at angles 0, pi/2, pi, 3pi/2.
    # Let the inner square have vertices at radius R2 at angles pi/4, 3pi/4, 5pi/4, 7pi/4.
    # We can parameterize this by R1, R2 and the relative rotation angle theta.
    # Actually, due to D4 symmetry, the "squares" must be aligned or rotated by 45 degrees?
    # If we assume D4 symmetry (8-fold rotational symmetry is broken to 4-fold),
    # the points come in 4 pairs under 90-degree rotation.
    # A general D4-symmetric set of 8 points can be described by 4 points in the first quadrant,
    # but if we assume they form two squares, we only need 2 parameters (radii) if aligned,
    # or 3 parameters (R1, R2, theta) if the inner square is rotated.
    
    # Let's use a slightly more general D4 symmetric parameterization:
    # 4 points in the first quadrant? No, D4 means invariant under 90 deg rotation.
    # So we define 2 points in the first quadrant (since 8 points / 4 quadrants = 2 points per quadrant).
    # Let p1 = (x1, y1) and p2 = (x2, y2) with x1,y1,x2,y2 > 0.
    # The 8 points are:
    # (x1, y1), (-y1, x1), (-x1, -y1), (y1, -x1)  -> Square 1 rotated by angle atan(y1/x1)
    # (x2, y2), (-y2, x2), (-x2, -y2), (y2, -x2)  -> Square 2 rotated by angle atan(y2/x2)
    
    # This reduces the problem to 4 parameters: x1, y1, x2, y2.
    # We can further reduce by noting that scaling all coordinates scales the score?
    # No, the score is scale-invariant (Delta/n^n after scaling to diameter 2).
    # So we can fix one parameter, e.g., x1 = 1, and optimize over y1, x2, y2.
    # Or just optimize over 4 parameters and let the optimizer handle scale.
    # Fixing scale is better for conditioning. Let's fix x1 = 1.
    # Then we optimize over y1, x2, y2.
    
    def objective_3d(params):
        y1, x2, y2 = params
        x1 = 1.0
        
        # Ensure positive
        y1 = abs(y1) + 1e-6
        x2 = abs(x2) + 1e-6
        y2 = abs(y2) + 1e-6
        
        p1 = np.array([x1, y1])
        p2 = np.array([-y1, x1])
        p3 = np.array([-x1, -y1])
        p4 = np.array([y1, -x1])
        
        p5 = np.array([x2, y2])
        p6 = np.array([-y2, x2])
        p7 = np.array([-x2, -y2])
        p8 = np.array([y2, -x2])
        
        points = np.vstack([p1, p2, p3, p4, p5, p6, p7, p8])
        
        return objective_full(points)

    # Initial best from regular octagon
    angles = 2 * np.pi * np.arange(n) / n
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = -compute_score(p_oct) # Negative because objective returns -score
    best_points = p_oct

    # Check time
    if time.time() > deadline:
        return [[float(x), float(y)] for x, y in best_points]

    # Differential Evolution on the 3D parameter space
    # Bounds: y1, x2, y2 can be anywhere in a reasonable range.
    # Since x1=1, the diameter is determined by the points.
    # Let's search in [0.1, 2.0] for each.
    bounds = [(0.1, 2.0), (0.1, 2.0), (0.1, 2.0)]
    
    try:
        result = differential_evolution(objective_3d, bounds, seed=42, maxiter=100, tol=1e-8, polish=True, workers=-1)
        if result.fun < best_score:
            best_score = result.fun
            y1, x2, y2 = result.x
            x1 = 1.0
            p1 = np.array([x1, y1])
            p2 = np.array([-y1, x1])
            p3 = np.array([-x1, -y1])
            p4 = np.array([y1, -x1])
            p5 = np.array([x2, y2])
            p6 = np.array([-y2, x2])
            p7 = np.array([-x2, -y2])
            p8 = np.array([y2, -x2])
            best_points = np.vstack([p1, p2, p3, p4, p5, p6, p7, p8])
    except Exception as e:
        pass

    # If time remains, try a more general D4 search with 4 parameters (no fixed scale)
    # or try full 16D Nelder-Mead starting from this good D4 solution.
    # The full 16D search might refine the D4 solution if the true optimum breaks D4 slightly.
    # However, D4 is a strong candidate. Let's do a quick Nelder-Mead polish on the full 16D space
    # starting from the D4 result.
    
    if time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            from scipy.optimize import minimize
            res = minimize(objective_full, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    return [[float(x), float(y)] for x, y in best_points]
