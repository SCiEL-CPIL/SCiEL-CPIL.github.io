"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog
import scipy.spatial.distance as sdist

N = 26
_START_TIME = time.time()
TIME_LIMIT = 130.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _is_valid(circles):
    """Check if circles are valid: inside unit square, disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-7:
                return False
    return True

def _objective(circles):
    """Sum of radii, but only if valid."""
    if not _is_valid(circles):
        return -1e10
    return np.sum(circles[:, 2])

def _initial_grid(n):
    """Create an initial grid layout of n circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
            count += 1
    return np.array(cells)

def _refine_positions(circles, max_iter=2000, lr=0.05):
    """
    Gradient-based refinement: push circles apart if overlapping,
    pull towards boundaries if too close to edge.
    Uses a simple repulsion/attraction scheme.
    """
    n = len(circles)
    pos = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    
    # We'll keep radii fixed during position refinement, or adjust them slightly
    # Actually, let's do a joint optimization with a penalty method.
    
    best_obj = _objective(circles)
    best_circles = circles.copy()
    
    for it in range(max_iter):
        if _elapsed() > TIME_LIMIT:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    overlap = min_dist - dist
                    fx = dx / dist * overlap
                    fy = dy / dist * overlap
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
                elif dist < 1e-10:
                    # Random push
                    angle = np.random.uniform(0, 2 * np.pi)
                    forces[i] += [np.cos(angle), np.sin(angle)]
                    forces[j] -= [np.cos(angle), np.sin(angle)]
        
        # Boundary repulsion
        for i in range(n):
            r = radii[i]
            x, y = pos[i]
            if x < r:
                forces[i, 0] += (r - x)
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r))
            if y < r:
                forces[i, 1] += (r - y)
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r))
        
        # Update positions
        pos += lr * forces
        # Clamp to reasonable bounds
        pos = np.clip(pos, [0, 0], [1, 1])
        
        new_circles = np.column_stack([pos, radii])
        new_obj = _objective(new_circles)
        if new_obj > best_obj:
            best_obj = new_obj
            best_circles = new_circles.copy()
            # Maybe reduce learning rate
            lr = max(lr * 0.999, 0.001)
        else:
            lr = max(lr * 0.99, 0.001)
    
    return best_circles

def _lp_optimize_radii(circles):
    """
    Given positions, solve an LP to maximize sum of radii subject to
    non-overlap and boundary constraints.
    Variables: r_i for each circle.
    Constraints:
      r_i + r_j <= dist_ij for all i < j
      r_i <= x_i (left), r_i <= 1 - x_i (right), r_i <= y_i (bottom), r_i <= 1 - y_i (top)
      r_i >= 0
    """
    n = len(circles)
    pos = circles[:, :2]
    
    # Distances between pairs
    d = sdist.cdist(pos, pos)
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Pair constraints: r_i + r_j <= d_ij
    A_rows = []
    b_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_rows.append(row)
            b_rows.append(d[i, j])
    
    # Boundary constraints:
    # r_i <= x_i  ->  -r_i >= -x_i  ->  r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    for i in range(n):
        # r_i <= x_i
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_rows.append(pos[i, 0])
        
        # r_i <= 1 - x_i
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_rows.append(1 - pos[i, 0])
        
        # r_i <= y_i
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_rows.append(pos[i, 1])
        
        # r_i <= 1 - y_i
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_rows.append(1 - pos[i, 1])
    
    if A_rows:
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
    else:
        A_ub = None
        b_ub = None
    
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            new_radii = res.x
            new_circles = np.column_stack([pos, new_radii])
            if _is_valid(new_circles):
                return new_circles
    except:
        pass
    
    return circles

def _joint_optimize(circles, max_iter=500):
    """
    Jointly optimize positions and radii using a penalty/gradient approach,
    then polish with LP.
    """
    n = len(circles)
    pos = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    
    best_obj = _objective(circles)
    best_circles = circles.copy()
    
    for it in range(max_iter):
        if _elapsed() > TIME_LIMIT:
            break
        
        forces_pos = np.zeros_like(pos)
        forces_rad = np.zeros(n)
        
        # Repulsion between circles (position force)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    fx = dx / dist * overlap
                    fy = dy / dist * overlap
                    forces_pos[i] += [fx, fy]
                    forces_pos[j] -= [fx, fy]
                    # Also reduce radii slightly to help resolve overlap
                    forces_rad[i] -= overlap * 0.5
                    forces_rad[j] -= overlap * 0.5
                elif dist < 1e-10:
                    angle = np.random.uniform(0, 2 * np.pi)
                    forces_pos[i] += [np.cos(angle), np.sin(angle)]
                    forces_pos[j] -= [np.cos(angle), np.sin(angle)]
        
        # Boundary constraints
        for i in range(n):
            r = radii[i]
            x, y = pos[i]
            # Left
            if x < r:
                forces_pos[i, 0] += (r - x)
                forces_rad[i] -= (r - x)
            # Right
            if x > 1 - r:
                forces_pos[i, 0] -= (x - (1 - r))
                forces_rad[i] -= (x - (1 - r))
            # Bottom
            if y < r:
                forces_pos[i, 1] += (r - y)
                forces_rad[i] -= (r - y)
            # Top
            if y > 1 - r:
                forces_pos[i, 1] -= (y - (1 - r))
                forces_rad[i] -= (y - (1 - r))
        
        # Update
        pos += 0.05 * forces_pos
        radii += 0.05 * forces_rad
        radii = np.maximum(radii, 1e-4)
        pos = np.clip(pos, 0, 1)
        
        new_circles = np.column_stack([pos, radii])
        new_obj = _objective(new_circles)
        if new_obj > best_obj:
            best_obj = new_obj
            best_circles = new_circles.copy()
    
    # Polish with LP
    polished = _lp_optimize_radii(best_circles)
    if _objective(polished) > best_obj:
        best_circles = polished
    
    return best_circles

def _random_perturb(circles, strength=0.05):
    """Randomly perturb circles to escape local minima."""
    n = len(circles)
    new_circles = circles.copy()
    for i in range(n):
        new_circles[i, 0] += np.random.uniform(-strength, strength)
        new_circles[i, 1] += np.random.uniform(-strength, strength)
        # Keep inside square (roughly)
        new_circles[i, 0] = np.clip(new_circles[i, 0], 0.01, 0.99)
        new_circles[i, 1] = np.clip(new_circles[i, 1], 0.01, 0.99)
    return new_circles

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with a grid
    circles = _initial_grid(N)
    best_circles = circles.copy()
    best_obj = _objective(best_circles)
    
    # Refine
    circles = _refine_positions(circles, max_iter=500)
    obj = _objective(circles)
    if obj > best_obj:
        best_obj = obj
        best_circles = circles.copy()
    
    # Joint optimization
    circles = _joint_optimize(circles, max_iter=300)
    obj = _objective(circles)
    if obj > best_obj:
        best_obj = obj
        best_circles = circles.copy()
    
    # Try multiple random restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if _elapsed() > TIME_LIMIT - 10:
            break
        
        # Start from current best with perturbation
        start = _random_perturb(best_circles, strength=0.02 * (restart + 1))
        
        circles = _refine_positions(start, max_iter=300)
        obj = _objective(circles)
        if obj > best_obj:
            best_obj = obj
            best_circles = circles.copy()
        
        circles = _joint_optimize(circles, max_iter=200)
        obj = _objective(circles)
        if obj > best_obj:
            best_obj = obj
            best_circles = circles.copy()
    
    # Final LP polish
    final = _lp_optimize_radii(best_circles)
    if _objective(final) > best_obj:
        best_circles = final
    
    # Ensure validity
    if not _is_valid(best_circles):
        # Fallback to grid
        best_circles = _initial_grid(N)
    
    return best_circles
