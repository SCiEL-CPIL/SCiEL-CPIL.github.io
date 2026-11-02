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

def _compute_penalty(x, N, edges):
    """
    Compute a penalty term for constraint violations.
    x: flat array of length 3*N
    edges: list of (i, j) tuples for Delaunay edges
    Returns penalty value (0 if feasible, >0 if infeasible)
    """
    pts = x.reshape(N, 3)
    penalty = 0.0
    
    # Boundary penalties
    for i in range(N):
        xi, yi, ri = pts[i]
        # Left
        if xi - ri < 0:
            penalty += (xi - ri)**2
        # Bottom
        if yi - ri < 0:
            penalty += (yi - ri)**2
        # Right
        if 1 - xi - ri < 0:
            penalty += (1 - xi - ri)**2
        # Top
        if 1 - yi - ri < 0:
            penalty += (1 - yi - ri)**2
        
    # Overlap penalties for Delaunay edges
    for (i, j) in edges:
        xi, yi, ri = pts[i]
        xj, yj, rj = pts[j]
        dist_sq = (xi - xj)**2 + (yi - yj)**2
        dist = np.sqrt(dist_sq)
        required_dist = ri + rj
        if dist < required_dist:
            # Penalty is square of the violation
            violation = required_dist - dist
            penalty += violation**2
            
    return penalty

def _make_initial_guess_hex(N):
    """
    Create an initial guess using a hexagonal lattice pattern.
    This is generally a better starting point than a square grid.
    """
    # Approximate number of columns
    # For N circles, side ~ sqrt(N)
    # Hexagonal packing density is higher, so we can fit slightly more in same area
    # Let's try to fit N circles in a hexagonal grid
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    # Hexagonal spacing
    # If we have 'rows' rows and 'cols' columns
    # Horizontal distance between centers in same row: dx
    # Vertical distance between rows: dy = dx * sqrt(3)/2
    # To fit in [0,1]x[0,1]:
    # (cols-1)*dx <= 1 - 2*r
    # (rows-1)*dy <= 1 - 2*r
    
    # Let's estimate r.
    # Total area N * pi * r^2 <= 1 * packing_density
    # For hex, packing density ~ 0.9069
    # r_est = sqrt(0.9069 / (N * pi))
    r_est = np.sqrt(0.9 / (N * np.pi))
    
    # Adjust dx, dy to fit
    # dx = 1 / (cols + 1) roughly?
    # Let's just place them in a grid with hex offsets
    pts = np.zeros((N, 3))
    
    idx = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx >= N:
                break
            
            # Hexagonal offset: shift odd rows by half dx
            x = (c_idx + 0.5) / cols
            if r_idx % 2 == 1:
                x += 0.5 / cols
            y = (r_idx + 0.5) / rows
            
            # Clamp to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            pts[idx, 0] = x
            pts[idx, 1] = y
            pts[idx, 2] = r_est * 0.8 # Start with slightly smaller radius
            
            idx += 1
            
    return pts

def construct():
    start_time = time.time()
    time_limit = 110  # leave some margin
    
    # Initial guess using hexagonal lattice
    pts = _make_initial_guess_hex(N)
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
    max_outer_iters = 8
    
    # Penalty weight
    # We want to maximize sum(r), so minimize -sum(r) + lambda * penalty
    # If penalty is large, we prioritize feasibility.
    # If penalty is small, we prioritize size.
    # Start with high penalty to get a feasible solution, then reduce it?
    # Or just use a moderate penalty and rely on SLSQP constraints?
    # SLSQP handles constraints, but the penalty helps with local minima.
    # Let's use a combined approach: SLSQP with constraints AND a penalty in objective.
    
    for outer_iter in range(max_outer_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Re-compute edges based on current best solution
        centers_cur = best_x.reshape(N, 3)[:, :2]
        edges = _delaunay_constraints(centers_cur)
        
        # Build objective function with penalty
        # We minimize: -sum(r) + lambda * penalty
        # The penalty term ensures that if SLSQP struggles with hard constraints,
        # it still moves towards feasibility.
        lambda_penalty = 10.0
        
        def obj_func(x):
            pts_x = x.reshape(N, 3)
            sum_r = np.sum(pts_x[:, 2])
            pen = _compute_penalty(x, N, edges)
            return -sum_r + lambda_penalty * pen
        
        # Constraints for SLSQP
        constraints = []
        
        # Boundary constraints (hard)
        # g(x) >= 0
        # x_i - r_i >= 0
        # y_i - r_i >= 0
        # 1 - x_i - r_i >= 0
        # 1 - y_i - r_i >= 0
        
        for i in range(N):
            # Left
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
            # Bottom
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
            # Right
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
            # Top
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
        
        # Non-overlap constraints for Delaunay edges (hard)
        # dist(i,j) - r_i - r_j >= 0
        
        for (i, j) in edges:
            def _nonoverlap(x, i=i, j=j):
                xi, yi = x[3*i], x[3*i+1]
                xj, yj = x[3*j], x[3*j+1]
                ri = x[3*i+2]
                rj = x[3*j+2]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                return dist - ri - rj
            constraints.append({'type': 'ineq', 'fun': _nonoverlap})
        
        # Run SLSQP
        try:
            result = minimize(
                obj_func,
                best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            # Check if the result is better
            # We evaluate the true objective (sum of radii) and validity
            x_sol = result.x
            pts_sol = x_sol.reshape(N, 3)
            
            # Verify validity
            valid = True
            # Check boundaries
            for i in range(N):
                xi, yi, ri = pts_sol[i]
                if ri - xi > 1e-6 or ri - yi > 1e-6:
                    valid = False
                if xi + ri - 1 > 1e-6 or yi + ri - 1 > 1e-6:
                    valid = False
            
            # Check all pairwise non-overlap
            for i in range(N):
                for j in range(i + 1, N):
                    xi, yi, ri = pts_sol[i]
                    xj, yj, rj = pts_sol[j]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    if ri + rj - dist > 1e-5:
                        valid = False
            
            sum_r = np.sum(pts_sol[:, 2])
            
            if valid and sum_r > -best_obj: # Note: best_obj stores -sum_r
                best_obj = -sum_r
                best_x = x_sol.copy()
                
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
        pts_final[i, 2] = ri
    
    # Final pairwise check and shrink if overlapping
    # Iterate a few times to resolve overlaps
    for _ in range(10):
        modified = False
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = pts_final[i]
                xj, yj, rj = pts_final[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if ri + rj > dist - 1e-8:
                    # Shrink both radii slightly
                    overlap = ri + rj - dist
                    shrink = overlap / 2 + 1e-6
                    new_ri = max(ri - shrink, 1e-6)
                    new_rj = max(rj - shrink, 1e-6)
                    if new_ri != ri or new_rj != rj:
                        pts_final[i, 2] = new_ri
                        pts_final[j, 2] = new_rj
                        modified = True
        if not modified:
            break
            
    # Final boundary check
    for i in range(N):
        xi, yi, ri = pts_final[i]
        ri = min(ri, xi - 1e-6, yi - 1e-6, 1 - xi - 1e-6, 1 - yi - 1e-6)
        ri = max(ri, 1e-6)
        pts_final[i, 2] = ri
    
    return pts_final
# EVOLVE-BLOCK-END
