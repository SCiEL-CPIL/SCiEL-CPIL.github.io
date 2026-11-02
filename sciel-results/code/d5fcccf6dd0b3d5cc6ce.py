"""Circle packing in unit square using Delaunay-based constrained optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

def _delaunay_constraints(points):
    """
    Given an array of points (n, 2), return a list of edge indices (i, j)
    from the Delaunay triangulation.
    """
    if len(points) < 3:
        return []
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a = simplex[i]
                b = simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]

def _check_validity(pts, tol=1e-6):
    """
    Check if the configuration is valid (no overlaps, inside bounds).
    """
    n = len(pts)
    for i in range(n):
        xi, yi, ri = pts[i]
        if ri < 0:
            return False
        if xi - ri < -tol or yi - ri < -tol:
            return False
        if xi + ri > 1 + tol or yi + ri > 1 + tol:
            return False
            
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = pts[i]
            xj, yj, rj = pts[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if ri + rj > dist + tol:
                return False
    return True

def _make_initial_guess(seed):
    """
    Create a jittered grid initial guess.
    """
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    pts = np.zeros((N, 3))
    
    for i in range(N):
        row = i // side
        col = i % side
        
        x_base = (col + 0.5) / side
        y_base = (row + 0.5) / side
        
        jitter_x = rng.uniform(-0.1, 0.1)
        jitter_y = rng.uniform(-0.1, 0.1)
        
        x = x_base + jitter_x
        y = y_base + jitter_y
        
        x = np.clip(x, 0.1, 0.9)
        y = np.clip(y, 0.1, 0.9)
        
        r = 0.3 / side
        r = r * rng.uniform(0.8, 1.2)
        
        pts[i] = [x, y, r]
        
    return pts

def construct():
    start_time = time.time()
    time_limit = 110
    
    best_obj = np.inf
    best_x = None
    
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        seed = 42 + restart
        pts_init = _make_initial_guess(seed)
        x0 = pts_init.flatten()
        
        centers0 = pts_init[:, :2]
        edges = _delaunay_constraints(centers0)
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((1e-6, 0.5))
            
        def obj_func(x):
            radii = x[2::3]
            return -np.sum(radii)
        
        constraints = []
        
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
        
        for (i, j) in edges:
            def _nonoverlap(x, i=i, j=j):
                xi, yi = x[3*i], x[3*i+1]
                xj, yj = x[3*j], x[3*j+1]
                ri = x[3*i+2]
                rj = x[3*j+2]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                return dist - ri - rj
            constraints.append({'type': 'ineq', 'fun': _nonoverlap})
            
        try:
            result = minimize(
                obj_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            x_sol = result.x
            pts_sol = x_sol.reshape(N, 3)
            
            if _check_validity(pts_sol):
                obj_val = -np.sum(pts_sol[:, 2])
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = x_sol.copy()
                    
        except Exception:
            pass
            
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        pts = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.2 / side
            pts[i] = [x, y, r]
        return pts

    pts_final = best_x.reshape(N, 3)
    
    for i in range(N):
        xi, yi, ri = pts_final[i]
        ri = min(ri, xi, yi, 1 - xi, 1 - yi)
        ri = max(ri, 1e-6)
        pts_final[i] = [xi, yi, ri]
        
    return pts_final
# EVOLVE-BLOCK-END
