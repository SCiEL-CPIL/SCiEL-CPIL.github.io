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
    from the Delaunay triangulation. These are the pairs that need to be
    checked for non-overlap (plus boundary).
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
        # Fallback: all pairs
        return [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]


def _objective_and_constraints(x, N, edges):
    """
    x: flat array of length 3*N: [x0, y0, r0, x1, y1, r1, ...]
    Returns (objective, constraints_list)
    """
    pts = x.reshape(N, 3)
    radii = pts[:, 2]
    centers = pts[:, :2]
    
    # Objective: maximize sum of radii -> minimize negative
    obj = -np.sum(radii)
    
    constraints = []
    
    # Boundary constraints:
    # x_i - r_i >= 0  =>  r_i - x_i <= 0
    # y_i - r_i >= 0  =>  r_i - y_i <= 0
    # 1 - x_i - r_i >= 0  =>  x_i + r_i - 1 <= 0
    # 1 - y_i - r_i >= 0  =>  y_i + r_i - 1 <= 0
    
    for i in range(N):
        xi, yi, ri = pts[i]
        # Left: r_i - x_i <= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2] - x[3*i]})
        # Bottom: r_i - y_i <= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2] - x[3*i+1]})
        # Right: x_i + r_i - 1 <= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] + x[3*i+2] - 1})
        # Top: y_i + r_i - 1 <= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] + x[3*i+2] - 1})
    
    # Non-overlap constraints for Delaunay edges:
    # For each edge (i, j): dist(i,j) - r_i - r_j >= 0
    # => r_i + r_j - dist(i,j) <= 0
    for (i, j) in edges:
        def _nonoverlap(x, i=i, j=j):
            xi, yi = x[3*i], x[3*i+1]
            xj, yj = x[3*j], x[3*j+1]
            ri = x[3*i+2]
            rj = x[3*j+2]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            # Constraint: ri + rj - dist <= 0
            return ri + rj - dist
        constraints.append({'type': 'ineq', 'fun': _nonoverlap})
    
    return obj, constraints


def _make_initial_guess(N):
    """
    Create an initial guess for the optimizer.
    Use a grid arrangement as starting point.
    """
    side = int(np.ceil(np.sqrt(N)))
    # Place circles in a grid
    xs = []
    ys = []
    for i in range(N):
        row = i // side
        col = i % side
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        xs.append(x)
        ys.append(y)
    
    # Initial radius: small
    r = 0.5 / side * 0.9
    pts = np.zeros((N, 3))
    for i in range(N):
        pts[i, 0] = xs[i]
        pts[i, 1] = ys[i]
        pts[i, 2] = r
    
    return pts


def construct():
    start_time = time.time()
    time_limit = 110  # leave some margin
    
    # Initial guess
    pts = _make_initial_guess(N)
    x0 = pts.flatten()
    
    # Get initial Delaunay edges
    centers0 = pts[:, :2]
    edges = _delaunay_constraints(centers0)
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5))  # r
    
    best_obj = np.inf
    best_x = x0.copy()
    
    # We'll do a few iterations of: solve with current edges, update edges
    max_outer_iters = 5
    
    for outer_iter in range(max_outer_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Re-compute edges based on current best solution
        centers_cur = best_x.reshape(N, 3)[:, :2]
        edges = _delaunay_constraints(centers_cur)
        
        # Build objective and constraints
        def obj_func(x):
            obj, _ = _objective_and_constraints(x, N, edges)
            return obj
        
        def make_constraint_func(i, j):
            def _c(x):
                xi, yi = x[3*i], x[3*i+1]
                xj, yj = x[3*j], x[3*j+1]
                ri = x[3*i+2]
                rj = x[3*j+2]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                return ri + rj - dist
            return _c
        
        constraints = []
        # Boundary constraints
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2] - x[3*i]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2] - x[3*i+1]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] + x[3*i+2] - 1})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] + x[3*i+2] - 1})
        
        # Non-overlap for edges
        for (i, j) in edges:
            constraints.append({'type': 'ineq', 'fun': make_constraint_func(i, j)})
        
        # Run SLSQP
        try:
            result = minimize(
                obj_func,
                best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success or result.fun < best_obj:
                # Check if solution is valid
                x_sol = result.x
                pts_sol = x_sol.reshape(N, 3)
                
                # Verify all constraints are satisfied
                valid = True
                # Check boundaries
                for i in range(N):
                    xi, yi, ri = pts_sol[i]
                    if ri - xi > 1e-6 or ri - yi > 1e-6:
                        valid = False
                    if xi + ri - 1 > 1e-6 or yi + ri - 1 > 1e-6:
                        valid = False
                
                # Check all pairwise non-overlap (not just Delaunay edges)
                for i in range(N):
                    for j in range(i + 1, N):
                        xi, yi, ri = pts_sol[i]
                        xj, yj, rj = pts_sol[j]
                        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                        if ri + rj - dist > 1e-4:
                            valid = False
                
                if valid:
                    best_obj = result.fun
                    best_x = x_sol.copy()
                else:
                    # If invalid, try to project back or keep old
                    pass
                    
        except Exception:
            pass
    
    # Final solution
    pts_final = best_x.reshape(N, 3)
    
    # Ensure validity by clamping radii if needed
    for i in range(N):
        xi, yi, ri = pts_final[i]
        # Clamp radius to fit within boundaries
        ri = min(ri, xi, yi, 1 - xi, 1 - yi)
        ri = max(ri, 1e-6)
        pts_final[i] = [xi, yi, ri]
    
    # Final pairwise check and shrink if overlapping
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = pts_final[i]
            xj, yj, rj = pts_final[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if ri + rj > dist - 1e-8:
                # Shrink both radii slightly
                overlap = ri + rj - dist
                shrink = overlap / 2 + 1e-6
                ri = max(ri - shrink, 1e-6)
                rj = max(rj - shrink, 1e-6)
                pts_final[i, 2] = ri
                pts_final[j, 2] = rj
    
    # Final boundary check
    for i in range(N):
        xi, yi, ri = pts_final[i]
        ri = min(ri, xi - 1e-6, yi - 1e-6, 1 - xi - 1e-6, 1 - yi - 1e-6)
        ri = max(ri, 1e-6)
        pts_final[i, 2] = ri
    
    return pts_final
# EVOLVE-BLOCK-END
