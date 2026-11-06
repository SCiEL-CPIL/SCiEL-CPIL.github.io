"""
Strategy:
- Parameterize 8 points as two concentric squares with independent radii and relative rotation.
- This enforces a "cage" topology that avoids the regular octagon local optimum.
- Use a global grid search over the 3 parameters, followed by local refinement.
- Also try a few different parameterizations to cover more configurations.
"""

import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 8
    
    def get_score(pts):
        # pts is (8, 2)
        # Calculate product of unordered distances U
        log_U = 0.0
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-12:
                    return -1e9
                log_U += np.log(d)
                if d > max_d:
                    max_d = d
        if max_d < 1e-12:
            return -1e9
        return log_U - 28.0 * np.log(max_d)
    
    def make_points_t1(r1, r2, theta):
        # Two concentric squares: inner radius r1, outer radius r2, relative rotation theta
        pts = []
        for k in range(4):
            angle = 2 * np.pi * k / 4 + np.pi / 8
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        for k in range(4):
            angle = 2 * np.pi * k / 4 + np.pi / 8 + theta
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        return np.array(pts)
    
    def make_points_t2(r1, r2, theta):
        # Alternative: one square at radius r1, another at r2 with rotation
        pts = []
        for k in range(4):
            angle = 2 * np.pi * k / 4
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        for k in range(4):
            angle = 2 * np.pi * k / 4 + theta
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        return np.array(pts)
    
    def make_points_t3(r1, r2, theta):
        # Another variant: staggered squares
        pts = []
        for k in range(4):
            angle = 2 * np.pi * k / 4 + np.pi / 4
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        for k in range(4):
            angle = 2 * np.pi * k / 4 + theta
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        return np.array(pts)
    
    def objective_3(x, make_fn):
        r1, r2, theta = x
        pts = make_fn(r1, r2, theta)
        score = get_score(pts)
        if score < -1e8:
            return 1e9
        return -score
    
    def grad_3(x, make_fn):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(3)
        base = -objective_3(x, make_fn)
        for i in range(3):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (-objective_3(x_plus, make_fn) - base) / eps
        return grad
    
    best_score = -np.inf
    best_pts = None
    
    # Try multiple parameterizations
    make_fns = [make_points_t1, make_points_t2, make_points_t3]
    
    for make_fn in make_fns:
        if time.time() > deadline - 30:
            break
        
        # Grid search
        r1_vals = np.linspace(0.1, 1.5, 30)
        r2_vals = np.linspace(0.5, 2.0, 30)
        theta_vals = np.linspace(0.0, np.pi / 4, 20)
        
        grid_best_score = -np.inf
        grid_best_x = None
        
        for r1 in r1_vals:
            if time.time() > deadline - 15:
                break
            for r2 in r2_vals:
                for theta in theta_vals:
                    x = np.array([r1, r2, theta])
                    pts = make_fn(r1, r2, theta)
                    score = get_score(pts)
                    if score > grid_best_score:
                        grid_best_score = score
                        grid_best_x = x
        
        if grid_best_x is not None and grid_best_score > best_score:
            best_score = grid_best_score
            best_pts = make_fn(*grid_best_x)
        
        # Local refinement from best grid point
        if grid_best_x is not None:
            try:
                res = minimize(
                    lambda x: objective_3(x, make_fn),
                    grid_best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    refined_pts = make_fn(*res.x)
                    refined_score = get_score(refined_pts)
                    if refined_score > best_score:
                        best_score = refined_score
                        best_pts = refined_pts
            except Exception:
                pass
    
    # Also try a regular octagon as baseline
    octagon = np.array([[np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(8)])
    octagon_score = get_score(octagon)
    if octagon_score > best_score or best_pts is None:
        best_score = octagon_score
        best_pts = octagon
    
    # Final scaling to ensure diameter is exactly 2
    if best_pts is not None:
        d_final = 0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d > d_final:
                    d_final = d
        if d_final > 1e-6:
            best_pts = best_pts * (2.0 / d_final)
        return [[float(x), float(y)] for x, y in best_pts]
    
    return [[float(octagon[k][0]), float(octagon[k][1])] for k in range(8)]
