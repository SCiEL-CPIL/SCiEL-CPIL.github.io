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
    timeout = 140  # seconds, keep 10s margin
    
    best_score = score(regular_octagon())
    best_points = regular_octagon()
    
    # Symmetric optimization:
    # Fix p0 = (-1, 0), p1 = (1, 0).
    # The other 6 points are assumed to be symmetric with respect to the x-axis.
    # Let the 3 points in the upper half-plane be (x1, y1), (x2, y2), (x3, y3) with yi > 0.
    # The full set is {(-1,0), (1,0), (x1,y1), (x1,-y1), (x2,y2), (x2,-y2), (x3,y3), (x3,-y3)}.
    
    def build_points(x):
        # x: [x1, y1, x2, y2, x3, y3]
        pts = np.zeros((8, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2] = [x[0], x[1]]
        pts[3] = [x[0], -x[1]]
        pts[4] = [x[2], x[3]]
        pts[5] = [x[2], -x[3]]
        pts[6] = [x[4], x[5]]
        pts[7] = [x[4], -x[5]]
        return pts

    def objective_sym(x):
        pts = build_points(x)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Check validity (distinct points)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if np.min(dists[mask]) < 1e-8:
            return 1e10
            
        # Log sum of distances
        log_sum = np.sum(np.log(dists[mask]))
        return -log_sum

    # Constraints for SLSQP: g(x) >= 0
    # We want d(pi, pj) <= 2, so 2 - d(pi, pj) >= 0
    
    def make_constraint_sym(i, j):
        def constraint(x):
            pts = build_points(x)
            d = np.linalg.norm(pts[i] - pts[j])
            return 2.0 - d
        return constraint

    constraints = []
    # We only need to check unique pairs.
    # Indices:
    # 0: (-1,0)
    # 1: (1,0)
    # 2: (x1, y1)
    # 3: (x1, -y1)
    # 4: (x2, y2)
    # 5: (x2, -y2)
    # 6: (x3, y3)
    # 7: (x3, -y3)
    
    # Pairs involving 0 and 1
    # d(0, 2) <= 2, d(0, 4) <= 2, d(0, 6) <= 2
    # d(1, 2) <= 2, d(1, 4) <= 2, d(1, 6) <= 2
    # By symmetry, d(0,3)=d(1,2), d(0,5)=d(1,4), d(0,7)=d(1,6)
    # d(1,3)=d(0,2), etc.
    # So we just need to constrain distances from 0 to upper points, and 1 to upper points.
    for i in [2, 4, 6]:
        constraints.append({'type': 'ineq', 'fun': make_constraint_sym(0, i)})
        constraints.append({'type': 'ineq', 'fun': make_constraint_sym(1, i)})
        
    # Pairs among upper points (2,4,6)
    # d(2,4), d(2,6), d(4,6)
    constraints.append({'type': 'ineq', 'fun': make_constraint_sym(2, 4)})
    constraints.append({'type': 'ineq', 'fun': make_constraint_sym(2, 6)})
    constraints.append({'type': 'ineq', 'fun': make_constraint_sym(4, 6)})
    
    # Pairs between upper and lower (mirrored)
    # d(2,5) = dist((x1,y1), (x2,-y2))
    # d(2,7) = dist((x1,y1), (x3,-y3))
    # d(4,3) = dist((x2,y2), (x1,-y1)) - same as d(2,5)
    # d(4,7) = dist((x2,y2), (x3,-y3))
    # d(6,3) = dist((x3,y3), (x1,-y1)) - same as d(2,7)
    # d(6,5) = dist((x3,y3), (x2,-y2)) - same as d(4,7)
    constraints.append({'type': 'ineq', 'fun': make_constraint_sym(2, 5)})
    constraints.append({'type': 'ineq', 'fun': make_constraint_sym(2, 7)})
    constraints.append({'type': 'ineq', 'fun': make_constraint_sym(4, 7)})
    
    # Vertical distance within same x: d(2,3) = 2*y1 <= 2 => y1 <= 1
    # d(4,5) = 2*y2 <= 2 => y2 <= 1
    # d(6,7) = 2*y3 <= 2 => y3 <= 1
    def constraint_y1(x): return 1.0 - x[1]
    def constraint_y2(x): return 1.0 - x[3]
    def constraint_y3(x): return 1.0 - x[5]
    constraints.append({'type': 'ineq', 'fun': constraint_y1})
    constraints.append({'type': 'ineq', 'fun': constraint_y2})
    constraints.append({'type': 'ineq', 'fun': constraint_y3})
    
    # Bounds
    # x_i in [-1.5, 1.5], y_i in [0.01, 1.0] (y must be > 0 for distinctness from mirror, but constraint handles y<=1)
    # Actually y can be close to 0, but points must be distinct.
    # If y=0, point coincides with mirror. We want y > 0.
    # SLSQP bounds:
    bounds = [(-1.5, 1.5), (1e-4, 1.0),
              (-1.5, 1.5), (1e-4, 1.0),
              (-1.5, 1.5), (1e-4, 1.0)]
    
    # Initial guesses
    # 1. Regular octagon symmetric subset
    # Octagon points:
    # (1,0), (0.707, 0.707), (0,1), (-0.707, 0.707), (-1,0), ...
    # Fixed: (-1,0), (1,0).
    # Remaining upper: (0.707, 0.707), (0, 1), (-0.707, 0.707).
    # So x1=0.707, y1=0.707; x2=0, y2=1; x3=-0.707, y3=0.707.
    init_x = np.array([0.707, 0.707, 0.0, 1.0, -0.707, 0.707])
    
    # Run SLSQP
    try:
        res = minimize(
            objective_sym, 
            init_x, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if time.time() - start_time < timeout:
            if res.success:
                pts = build_points(res.x)
                # Verify
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                D = np.max(dists)
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                if np.min(dists[mask]) > 1e-5 and D <= 2.0 + 1e-5:
                    cand_score = score(pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = pts
    except Exception as e:
        pass

    # Try other initializations for symmetric case
    # Random symmetric points
    if time.time() - start_time < timeout:
        np.random.seed(42)
        for _ in range(20):
            x0 = np.random.uniform(-1.0, 1.0, 3)
            y0 = np.random.uniform(0.1, 0.9, 3)
            init_rand = np.concatenate([x0, y0])
            
            try:
                res = minimize(
                    objective_sym, 
                    init_rand, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                if time.time() - start_time < timeout:
                    if res.success:
                        pts = build_points(res.x)
                        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                        dists = np.linalg.norm(diffs, axis=2)
                        D = np.max(dists)
                        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                        if np.min(dists[mask]) > 1e-5 and D <= 2.0 + 1e-5:
                            cand_score = score(pts)
                            if cand_score > best_score:
                                best_score = cand_score
                                best_points = pts
            except Exception:
                pass

    return best_points.tolist()
