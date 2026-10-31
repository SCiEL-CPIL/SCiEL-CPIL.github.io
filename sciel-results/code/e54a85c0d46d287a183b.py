"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import sys

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time


def _initial_packing(n):
    """
    Create an initial packing of n circles in the unit square.
    We use a grid-like arrangement with some randomization to avoid symmetry.
    """
    # Place circles on a grid with some jitter
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side + (np.random.rand() - 0.5) * 0.1 / side
            y = (j + 0.5) / side + (np.random.rand() - 0.5) * 0.1 / side
            # Initial radius: half the cell size minus a small margin
            r = 0.4 / side
            cells.append([x, y, r])
    
    # If we didn't fill enough, add more in random positions
    while len(cells) < n:
        x = np.random.rand()
        y = np.random.rand()
        r = 0.05
        cells.append([x, y, r])
    
    return np.array(cells[:n])


def _objective_and_constraints(x, n):
    """
    x is a vector of length 3*n: [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    We want to maximize sum of radii, so minimize -sum(r)
    
    Constraints:
    1. Each circle must be inside the unit square:
       r_i <= x_i <= 1 - r_i
       r_i <= y_i <= 1 - r_i
    2. Circles must be disjoint:
       (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
       => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    """
    pts = x.reshape(n, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    rs = pts[:, 2]
    
    # Objective: minimize -sum(r)
    obj = -np.sum(rs)
    
    # Inequality constraints (need to be >= 0 for scipy's 'ineq' form,
    # but we'll use LinearConstraint for the box constraints and handle
    # non-linear constraints via penalty or SLSQP's constraint dict)
    
    # For SLSQP, we define constraints as a list of dicts:
    # {'type': 'ineq', 'fun': lambda x: ...} where fun(x) >= 0
    
    constraints = []
    
    # Box constraints: r_i <= x_i <= 1 - r_i, etc.
    # These are non-linear because they involve r_i on both sides.
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    # We'll add these as separate inequality constraints
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i] - x[3*i+2]
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]
        })
        # r_i >= 0 (radius non-negative)
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+2]
        })
    
    # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
            })
    
    return obj, constraints


def _is_valid_packing(pts, tol=1e-6):
    """Check if a packing is valid."""
    n = len(pts)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Check bounds
    if np.any(xs < -tol) or np.any(xs > 1 + tol) or np.any(ys < -tol) or np.any(ys > 1 + tol):
        return False
    if np.any(rs < -tol):
        return False
    for i in range(n):
        if xs[i] - rs[i] < -tol or 1 - xs[i] - rs[i] < -tol:
            return False
        if ys[i] - rs[i] < -tol or 1 - ys[i] - rs[i] < -tol:
            return False
    
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _fix_packing(pts):
    """Adjust a packing to be strictly valid by shrinking radii slightly if needed."""
    pts = pts.copy()
    n = len(pts)
    
    # First, clip positions and radii to be within bounds
    for i in range(n):
        x, y, r = pts[i]
        # Ensure r is positive
        r = max(r, 1e-8)
        # Clip x to [r, 1-r]
        x = np.clip(x, r, 1 - r)
        # Clip y to [r, 1-r]
        y = np.clip(y, r, 1 - r)
        pts[i] = [x, y, r]
    
    # Now iteratively shrink radii to resolve overlaps
    for iteration in range(100):
        fixed = True
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = pts[i]
                x2, y2, r2 = pts[j]
                dx = x1 - x2
                dy = y1 - y2
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r1 + r2
                if dist < min_dist:
                    # Shrink both radii
                    overlap = min_dist - dist
                    shrink = overlap / 2 + 1e-10
                    r1 = max(r1 - shrink, 1e-8)
                    r2 = max(r2 - shrink, 1e-8)
                    pts[i, 2] = r1
                    pts[j, 2] = r2
                    fixed = False
        
        if fixed:
            break
    
    # Final clip
    for i in range(n):
        x, y, r = pts[i]
        r = max(r, 1e-8)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        pts[i] = [x, y, r]
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_score = 0
    best_pts = None
    
    # Try multiple random initializations
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + restart)
        init = _initial_packing(N)
        
        # Initial feasibility check and fix
        init = _fix_packing(init)
        if not _is_valid_packing(init):
            # Try a more conservative initial packing
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) >= N:
                        break
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.4 / side
                    cells.append([x, y, r])
            while len(cells) < N:
                x = 0.5
                y = 0.5
                r = 0.05
                cells.append([x, y, r])
            init = np.array(cells[:N])
            init = _fix_packing(init)
        
        if not _is_valid_packing(init):
            continue
        
        score_init = np.sum(init[:, 2])
        if score_init > best_score:
            best_score = score_init
            best_pts = init.copy()
        
        # Run SLSQP optimization
        x0 = init.flatten()
        
        # Build constraints
        constraints = []
        for i in range(N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3*i] - x[3*i+2]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3*i+2]
            })
        
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
                })
        
        def objective(x):
            pts = x.reshape(N, 3)
            return -np.sum(pts[:, 2])
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.success:
                pts_opt = result.x.reshape(N, 3)
                pts_opt = _fix_packing(pts_opt)
                
                if _is_valid_packing(pts_opt):
                    score_opt = np.sum(pts_opt[:, 2])
                    if score_opt > best_score:
                        best_score = score_opt
                        best_pts = pts_opt.copy()
                        
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    if best_pts is None:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.4 / side
                cells.append([x, y, r])
        while len(cells) < N:
            cells.append([0.5, 0.5, 0.05])
        best_pts = np.array(cells[:N])
        best_pts = _fix_packing(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
