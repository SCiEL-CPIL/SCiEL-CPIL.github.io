"""
SQP-like packer with explicit radius variable and active-set refinement.

Strategy:
1. Start with a grid initialization.
2. Formulate the problem as maximizing r subject to:
   - ||c_i - c_j|| >= 2r for all i < j
   - c_ik + r <= 0.5 and -c_ik + r <= 0.5 for all i, k
3. Use a penalty-based SQP approach:
   - At each step, solve a QP subproblem to find the step direction.
   - The QP minimizes 0.5*||delta||^2 - lambda*delta_r subject to linearized constraints.
   - Update the points and radius.
4. Use an active-set refinement to polish the solution.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize, linprog

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _grid_init():
    """Generate a 5x5x4 grid."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _sqp_step(pts, r, active_pairs, active_walls, max_iter=100, tol=1e-8):
    """
    Perform one SQP step.
    pts: (N, 3) array of current points
    r: current radius
    active_pairs: list of (i, j) pairs that are active (dist < 2r + tol)
    active_walls: list of (i, k) where k is coordinate, active if margin < r + tol
    Returns: (delta_pts, delta_r)
    """
    n_vars = N * 3 + 1  # 3N point coordinates + 1 radius
    
    # Objective: minimize 0.5*||delta||^2 - delta_r
    # This is equivalent to minimizing ||delta||^2 - 2*delta_r
    # In standard QP form: 0.5 * x^T H x + g^T x
    # H = I (identity), g = [0, ..., 0, -2] (last component for r)
    
    # We'll use scipy.optimize.minimize with L-BFGS-B on the QP subproblem
    # Actually, let's use a more direct approach with linprog or manual QP
    
    # Linearized constraints:
    # For each active pair (i, j):
    #   ||c_i - c_j + delta_i - delta_j|| >= 2(r + delta_r)
    #   Approximate: (c_i - c_j) . (delta_i - delta_j) / ||c_i - c_j|| >= 2*delta_r
    #   => (delta_i - delta_j) . (c_i - c_j)/||c_i - c_j|| - 2*delta_r >= 0
    
    # For each active wall (i, k):
    #   c_ik + delta_ik + (r + delta_r) <= 0.5
    #   => delta_ik + delta_r <= 0.5 - c_ik - r
    
    #   -c_ik - delta_ik + (r + delta_r) <= 0.5
    #   => -delta_ik + delta_r <= 0.5 + c_ik - r
    
    # Build constraint matrix A and vector b for A @ x <= b
    # x = [delta_pts (3N), delta_r]
    
    A_rows = []
    b_vals = []
    
    # Pair constraints: (delta_i - delta_j) . u_ij - 2*delta_r >= 0
    # => -((delta_i - delta_j) . u_ij - 2*delta_r) <= 0
    # => -u_ij . delta_i + u_ij . delta_j + 2*delta_r <= 0
    for (i, j) in active_pairs:
        d_ij = pts[i] - pts[j]
        norm = np.linalg.norm(d_ij)
        if norm < 1e-12:
            continue
        u_ij = d_ij / norm
        
        row = np.zeros(n_vars)
        row[3*i:3*i+3] = -u_ij
        row[3*j:3*j+3] = u_ij
        row[-1] = 2.0
        
        A_rows.append(row)
        b_vals.append(0.0)
    
    # Wall constraints (upper): delta_ik + delta_r <= 0.5 - c_ik - r
    for (i, k) in active_walls:
        if k == 0:  # x-coordinate
            row = np.zeros(n_vars)
            row[3*i] = 1.0
            row[-1] = 1.0
            A_rows.append(row)
            b_vals.append(0.5 - pts[i, 0] - r)
            
            row = np.zeros(n_vars)
            row[3*i] = -1.0
            row[-1] = 1.0
            A_rows.append(row)
            b_vals.append(0.5 + pts[i, 0] - r)
        elif k == 1:  # y-coordinate
            row = np.zeros(n_vars)
            row[3*i+1] = 1.0
            row[-1] = 1.0
            A_rows.append(row)
            b_vals.append(0.5 - pts[i, 1] - r)
            
            row = np.zeros(n_vars)
            row[3*i+1] = -1.0
            row[-1] = 1.0
            A_rows.append(row)
            b_vals.append(0.5 + pts[i, 1] - r)
        elif k == 2:  # z-coordinate
            row = np.zeros(n_vars)
            row[3*i+2] = 1.0
            row[-1] = 1.0
            A_rows.append(row)
            b_vals.append(0.5 - pts[i, 2] - r)
            
            row = np.zeros(n_vars)
            row[3*i+2] = -1.0
            row[-1] = 1.0
            A_rows.append(row)
            b_vals.append(0.5 + pts[i, 2] - r)
    
    if not A_rows:
        return np.zeros((N, 3)), 0.0
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    # Solve QP: min 0.5*||x||^2 - x[-1] subject to A @ x <= b
    # Use scipy.optimize.minimize with L-BFGS-B
    
    def qp_obj(x):
        return 0.5 * np.dot(x, x) - x[-1]
    
    def qp_grad(x):
        g = x.copy()
        g[-1] -= 1.0
        return g
    
    x0 = np.zeros(n_vars)
    bounds = [(-1.0, 1.0)] * (N * 3) + [(-1.0, 1.0)]
    
    try:
        res = minimize(qp_obj, x0, jac=qp_grad, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': max_iter, 'ftol': tol})
        
        if res.success or res.fun < qp_obj(x0):
            delta_pts = res.x[:3*N].reshape(N, 3)
            delta_r = res.x[-1]
            return delta_pts, delta_r
    except Exception:
        pass
    
    return np.zeros((N, 3)), 0.0

def _sqp_optimize(pts, r, max_iter=50):
    """
    Perform SQP optimization.
    """
    for _ in range(max_iter):
        # Find active constraints
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        margin = 0.5 - np.abs(pts).max(axis=1)
        
        # Active pairs: dist < 2r + 1e-6
        tol = 1e-6
        active_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                if dists[i, j] < 2*r + tol:
                    active_pairs.append((i, j))
        
        # Active walls: margin < r + tol
        active_walls = []
        for i in range(N):
            for k in range(3):
                if 0.5 - pts[i, k] < r + tol or 0.5 + pts[i, k] < r + tol:
                    active_walls.append((i, k))
        
        # Get SQP step
        delta_pts, delta_r = _sqp_step(pts, r, active_pairs, active_walls)
        
        # Update
        pts_new = pts + delta_pts
        r_new = r + delta_r
        
        # Ensure points stay in cube
        pts_new = np.clip(pts_new, -0.5, 0.5)
        
        # Check if radius increased
        if r_new > r:
            pts = pts_new
            r = r_new
        else:
            # If radius didn't increase, try a smaller step
            step_size = 0.5
            for _ in range(10):
                pts_try = pts + step_size * delta_pts
                r_try = r + step_size * delta_r
                pts_try = np.clip(pts_try, -0.5, 0.5)
                
                # Check if this is feasible
                diff_try = pts_try[:, np.newaxis, :] - pts_try[np.newaxis, :, :]
                dists_try = np.sqrt((diff_try ** 2).sum(axis=2))
                np.fill_diagonal(dists_try, np.inf)
                min_dist_try = dists_try.min()
                
                margin_try = 0.5 - np.abs(pts_try).max(axis=1)
                min_margin_try = margin_try.min()
                
                r_check = min(min_dist_try / 2.0, min_margin_try)
                
                if r_check >= r_try - 1e-10:
                    pts = pts_try
                    r = r_try
                    break
                step_size *= 0.5
    
    return pts, r

def _active_set_polish(pts, r, max_iter=20):
    """
    Polish using active set method.
    """
    for _ in range(max_iter):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        margin = 0.5 - np.abs(pts).max(axis=1)
        
        # Find the minimum constraint
        min_dist = dists.min()
        min_margin = margin.min()
        
        if min_dist / 2.0 < min_margin:
            # Pair constraint is active
            i, j = np.unravel_index(dists.argmin(), dists.shape)
            if i == j:
                break
            
            d_ij = pts[i] - pts[j]
            norm = np.linalg.norm(d_ij)
            if norm < 1e-12:
                break
            u_ij = d_ij / norm
            
            # Move i and j apart
            pts[i] -= 0.5 * u_ij * 1e-4
            pts[j] += 0.5 * u_ij * 1e-4
        else:
            # Wall constraint is active
            i = margin.argmin()
            k = np.abs(pts[i]).argmax()
            
            # Move i away from wall
            if pts[i, k] > 0:
                pts[i, k] -= 1e-4
            else:
                pts[i, k] += 1e-4
        
        # Update r
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        margin = 0.5 - np.abs(pts).max(axis=1)
        r = min(dists.min() / 2.0, margin.min())
        
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    random.seed(42)
    
    # Initial solution
    points = _grid_init()
    r = _score(points)
    best_points = points.copy()
    best_r = r
    
    # Phase 1: SQP optimization from grid
    if time.time() < deadline - 60:
        pts1, r1 = _sqp_optimize(points, r, max_iter=30)
        if r1 > best_r:
            best_points = pts1
            best_r = r1
        
        # Active set polish
        pts2, r2 = _active_set_polish(best_points, best_r, max_iter=20)
        if r2 > best_r:
            best_points = pts2
            best_r = r2

    # Phase 2: Multi-start
    while time.time() < deadline - 20:
        # Create a new start
        if random.random() < 0.7:
            new_start = _grid_init() + np.random.normal(0, 0.02, (N, 3))
            new_start = np.clip(new_start, -0.5, 0.5)
        else:
            new_start = np.random.uniform(-0.4, 0.4, (N, 3))
        
        r_new = _score(new_start)
        
        # SQP optimize
        if time.time() < deadline - 10:
            opt_pts, opt_r = _sqp_optimize(new_start, r_new, max_iter=20)
            if opt_r > best_r:
                best_points = opt_pts
                best_r = opt_r
            
            # Active set polish
            if time.time() < deadline - 5:
                polish_pts, polish_r = _active_set_polish(best_points, best_r, max_iter=10)
                if polish_r > best_r:
                    best_points = polish_pts
                    best_r = polish_r

    # Final Polish
    if time.time() < deadline - 5:
        polish_pts, polish_r = _active_set_polish(best_points, best_r, max_iter=20)
        if polish_r > best_r:
            best_points = polish_pts
            best_r = polish_r

    return best_points

# EVOLVE-BLOCK-END
