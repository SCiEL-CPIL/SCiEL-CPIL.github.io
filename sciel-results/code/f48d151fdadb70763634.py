"""
Circle packing in unit square using Contact Graph Optimization.

Strategy:
1. Represent the packing by a contact graph: edges between circles or between
   circles and walls (x=0, x=1, y=0, y=1) indicate tangency.
2. For a given graph, the positions (x_i, y_i) and radii (r_i) must satisfy:
   - For circle-circle edge (i, j): (x_i - x_j)^2 + (y_i - y_j)^2 = (r_i + r_j)^2
   - For circle-wall edge (i, w): distance from center to wall = r_i
     e.g., if touching left wall (x=0): x_i = r_i
3. We have 3*N variables. We need 3*N equations.
   - Wall contacts provide 1 equation each.
   - Circle-circle contacts provide 1 equation each.
4. We search for valid contact graphs (planar, feasible) and solve the system.
5. To ensure a unique solution (or at least a well-posed one), we often need
   specific structures. However, fsolve can handle under/over-determined systems
   if we use least squares or if the system is consistent.
   Actually, for a rigid packing, the number of constraints should match degrees of freedom.
   In 2D, a circle has 3 DOF (x, y, r).
   A contact graph with E edges and N circles.
   If all circles are in contact with the boundary or each other in a way that
   fixes the configuration, we can solve.
   
   A simpler robust approach for "maximize sum of radii":
   Use a heuristic to generate candidate contact graphs (e.g., Delaunay triangulation
   of a random set of points, or random planar graphs), then solve for the configuration
   that satisfies the tangencies.
   
   To make it deterministic and fast, we will:
   1. Generate random initial positions.
   2. Compute Delaunay triangulation to get a candidate contact graph.
   3. Add wall contacts for circles near boundaries.
   4. Solve the nonlinear system.
   5. Check validity (all inside, no overlaps).
   6. Keep the best.
"""

import numpy as np
import time
from scipy.optimize import fsolve, least_squares
from scipy.spatial import Delaunay
import random

N = 26

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _solve_contact_graph(edges, wall_contacts, initial_guess):
    """
    Solve for x, y, r given a set of circle-circle edges and wall contacts.
    variables: v = [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
    
    Equations:
    1. For each circle-circle edge (i, j):
       (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 = 0
    2. For each wall contact (i, wall_id):
       wall_id 0: x_i - r_i = 0
       wall_id 1: 1 - x_i - r_i = 0
       wall_id 2: y_i - r_i = 0
       wall_id 3: 1 - y_i - r_i = 0
    """
    n = N
    def residuals(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        res = []
        
        # Circle-circle contacts
        for i, j in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            r_sum = r[i] + r[j]
            res.append(dist_sq - r_sum*r_sum)
            
        # Wall contacts
        for i, w in wall_contacts:
            if w == 0: # Left
                res.append(x[i] - r[i])
            elif w == 1: # Right
                res.append(1.0 - x[i] - r[i])
            elif w == 2: # Bottom
                res.append(y[i] - r[i])
            elif w == 3: # Top
                res.append(1.0 - y[i] - r[i])
                
        return np.array(res)
    
    # We might have fewer equations than variables.
    # fsolve expects M equations for N variables. If M < N, it finds the minimum norm solution.
    # This is good for under-determined systems (flexible packings).
    # If M > N, it finds least squares solution.
    
    try:
        solution, info, ier, msg = fsolve(residuals, initial_guess, full_output=True)
        if ier != 1:
            return None, None
        x = solution[0:n]
        y = solution[n:2*n]
        r = solution[2*n:3*n]
        
        # Check if radii are positive
        if np.any(r < 0.001):
            return None, None
            
        # Check if solution is valid (inside square, no overlaps)
        # The solver minimizes residuals, but might still have overlaps if the graph was inconsistent
        # or if it converged to a local min of residuals that isn't zero.
        total, valid = _eval_packing(np.column_stack([x, y]), r)
        if not valid:
            return None, None
        return np.column_stack([x, y]), r
    except Exception:
        return None, None

def _generate_graph_and_solve(seed, deadline):
    """
    Generate a random initial configuration, derive a contact graph,
    and solve for the optimal radii/positions for that graph.
    """
    rng = np.random.RandomState(seed)
    
    # Random initial positions
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = rng.uniform(0.05, 0.15, N)
    
    points = np.column_stack([x, y])
    
    # Delaunay triangulation for circle-circle contacts
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                i, j = simplex[k], simplex[(k+1)%3]
                if i > j: i, j = j, i
                edges.add((i, j))
    except Exception:
        # Fallback: random edges
        for _ in range(N):
            i = rng.randint(0, N)
            j = rng.randint(0, N)
            if i != j:
                if i > j: i, j = j, i
                edges.add((i, j))
    
    # Wall contacts: assign each circle to the nearest wall
    wall_contacts = []
    for i in range(N):
        dists = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
        min_idx = np.argmin(dists)
        wall_contacts.append((i, int(min_idx)))
    
    # Initial guess for solver
    v0 = np.concatenate([x, y, r])
    
    # Solve
    sol_xy, sol_r = _solve_contact_graph(list(edges), wall_contacts, v0)
    
    if sol_xy is None:
        return None, None, 0.0
        
    total = float(np.sum(sol_r))
    return sol_xy, sol_r, total

def _refine_with_gradient(xy, r, deadline):
    """
    Use a gradient-based method to further refine the packing.
    Objective: maximize sum(r) subject to constraints.
    We use a penalty method.
    """
    n = N
    def objective(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        # Penalty for wall violations
        pen = 0.0
        for i in range(n):
            if x[i] < r[i]: pen += (r[i] - x[i])**2
            if 1.0 - x[i] < r[i]: pen += (r[i] - (1.0 - x[i]))**2
            if y[i] < r[i]: pen += (r[i] - y[i])**2
            if 1.0 - y[i] < r[i]: pen += (r[i] - (1.0 - y[i]))**2
            
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    pen += (r[i] + r[j] - dist)**2
                    
        return -np.sum(r) + 100.0 * pen

    from scipy.optimize import minimize
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    try:
        res = minimize(objective, np.concatenate([xy[:,0], xy[:,1], r]), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-12})
        v_opt = res.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            return new_xy, new_r
    except Exception:
        pass
        
    return xy, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Phase 1: Contact Graph Search
    # Try many random seeds to find a good contact graph
    num_trials = 200
    for i in range(num_trials):
        if time.time() > deadline - 20:
            break
            
        seed = i * 7 + 1
        sol_xy, sol_r, total = _generate_graph_and_solve(seed, deadline)
        
        if sol_xy is not None and total > best_total:
            best_total = total
            best_valid = True
            best_xy = sol_xy
            best_r = sol_r
            
            # If we found a good one, try to refine it immediately
            if time.time() < deadline - 10:
                refined_xy, refined_r = _refine_with_gradient(best_xy, best_r, deadline)
                total_ref, valid_ref = _eval_packing(refined_xy, refined_r)
                if valid_ref and total_ref > best_total:
                    best_total = total_ref
                    best_xy = refined_xy
                    best_r = refined_r

    # Fallback if no valid packing found (unlikely, but safe)
    if not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            xy[i, 0] = (gx + 0.5) / side
            xy[i, 1] = (gy + 0.5) / side
            r[i] = 0.4 / side
        best_xy = xy
        best_r = r
        best_total, best_valid = _eval_packing(best_xy, best_r)
        
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Extreme fallback
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
