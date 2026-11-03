"""
Pack 200 equal circles in a unit square using a combined approach:
1. Soft-min gradient optimization (L-BFGS-B) for initial optimization
2. Interior-point constrained NLP (SLSQP) for final polish with explicit radius variable
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 145  # Leave margin for 150s limit

def _get_radius(pts):
    """
    Calculate the maximum possible radius for given centers.
    Returns the min of (wall distances) and (pairwise distances / 2).
    """
    if len(pts) == 0:
        return 0.0
    
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    min_wall = np.min(wall_dists)
    
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min(vals, beta):
    """
    Compute the soft minimum of a set of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta

def _objective_and_grad(x, beta):
    """
    Objective function for minimization.
    We want to MAXIMIZE the soft-min of constraints.
    So we MINIMIZE -soft_min.
    
    x: flattened array of shape (2*N,)
    beta: sharpness parameter for soft-min
    """
    pts = x.reshape((N, 2))
    
    # Indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # Compute all constraint values
    C = np.empty(4*N + num_pairs)
    
    # Wall values
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    
    # Pair values
    D = cdist(pts, pts)
    pair_dists = D[i_idx, j_idx]
    C[4*N:] = pair_dists / 2.0
    
    # Compute soft min and weights
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights for gradient
    weights = exp_shifted / sum_exp
    
    # Compute Gradient
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_x_left = weights[0:N] * 1.0
    grad_x_right = weights[N:2*N] * (-1.0)
    grad_y_bottom = weights[2*N:3*N] * 1.0
    grad_y_top = weights[3*N:4*N] * (-1.0)
    
    grad_pts[:, 0] += grad_x_left + grad_x_right
    grad_pts[:, 1] += grad_y_bottom + grad_y_top
    
    # Pair gradients
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    grad_pair_x_i = dx_safe / (2.0 * dists_safe)
    grad_pair_x_j = -dx_safe / (2.0 * dists_safe)
    grad_pair_y_i = dy_safe / (2.0 * dists_safe)
    grad_pair_y_j = -dy_safe / (2.0 * dists_safe)
    
    grad_pair_x_i = np.where(valid, grad_pair_x_i, 0.0)
    grad_pair_x_j = np.where(valid, grad_pair_x_j, 0.0)
    grad_pair_y_i = np.where(valid, grad_pair_y_i, 0.0)
    grad_pair_y_j = np.where(valid, grad_pair_y_j, 0.0)
    
    pair_weights = weights[4*N:]
    
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    grad_pts = -grad_pts / beta
    
    return obj_val, grad_pts.flatten()

def _generate_hex_initial():
    """
    Generate a centered hexagonal lattice packing as initial guess.
    """
    s = 0.0732
    points = []
    row_h = s * np.sqrt(3) / 2
    
    y = -0.6
    row_idx = 0
    while y < 0.6:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.6 + shift
        while x < 0.6:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
        
    pts = np.array(points)
    
    margin = 0.001
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]
    
    if len(pts) < N:
        s2 = 0.070
        points2 = []
        row_h2 = s2 * np.sqrt(3) / 2
        y = -0.6
        row_idx = 0
        while y < 0.6:
            shift = (s2 / 2.0) if row_idx % 2 == 1 else 0.0
            x = -0.6 + shift
            while x < 0.6:
                points2.append([x, y])
                x += s2
            y += row_h2
            row_idx += 1
        pts = np.array(points2)
        mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
               (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
        pts = pts[mask]
        if len(pts) < N:
            return np.random.uniform(-0.4, 0.4, (N, 2))
        
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]
    
    return pts

def _generate_random_initial():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.4, 0.4, (N, 2))

def _run_soft_min_optimization(pts, time_limit, betas=None):
    """Run L-BFGS-B optimization with soft-min objective."""
    if betas is None:
        betas = [10, 20, 50, 100, 200, 500, 1000]
        
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
            
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
            
        try:
            maxiter = 300 if beta >= 100 else 150
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
            
    return best_r, best_x.reshape((N, 2))

def _run_slsqp_polish(pts, time_limit):
    """
    Run SLSQP optimization with explicit radius variable.
    Maximize r subject to:
    - Circle non-overlap: ||c_i - c_j|| >= 2r
    - Boundary containment: |c_ik| <= 0.5 - r
    """
    t_start = time.time()
    
    # Initial radius estimate
    r0 = _get_radius(pts)
    
    # Variables: [x_1, y_1, ..., x_N, y_N, r]
    x0 = np.concatenate([pts.flatten(), [r0]])
    
    # Bounds: coordinates in [-0.5, 0.5], r > 0
    bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    def objective(x):
        """Minimize -r (i.e., maximize r)."""
        return -x[-1]
    
    def constraint_functions(x):
        """
        Generate constraint functions for SLSQP.
        Returns list of constraint dicts.
        
        Constraints:
        1. For each pair (i,j): ||c_i - c_j||^2 - (2r)^2 >= 0
        2. For each point i, each wall: (0.5 - r) - |c_ik| >= 0
        """
        pts = x[:2*N].reshape(N, 2)
        r = x[-1]
        
        constraints = []
        
        # Pair constraints: ||c_i - c_j||^2 >= (2r)^2
        # Equivalently: ||c_i - c_j||^2 - 4r^2 >= 0
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        dist_sq = dx**2 + dy**2
        pair_vals = dist_sq - 4.0 * r**2
        
        # We need to add these as constraints. But SLSQP expects individual constraints.
        # For efficiency, we'll add them in batches or use a single constraint that's the min.
        # Actually, SLSQP can handle many constraints. Let's add them individually.
        # But 19900 constraints might be slow. Let's try adding them.
        
        for k in range(num_pairs):
            def make_pair_constraint(k):
                def constraint(x):
                    pts = x[:2*N].reshape(N, 2)
                    r = x[-1]
                    i, j = i_idx[k], j_idx[k]
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    return dx**2 + dy**2 - 4.0 * r**2
                return constraint
            
            constraints.append({
                'type': 'ineq',
                'fun': make_pair_constraint(k)
            })
        
        # Wall constraints: 0.5 - r - |c_ik| >= 0
        # For each point and each wall
        
        for i in range(N):
            # Left wall: 0.5 - r - (-0.5 - pts[i,0]) = 1.0 - r + pts[i,0] >= 0
            # Actually: pts[i,0] >= -0.5 + r, so -0.5 + r - pts[i,0] <= 0
            # Or: pts[i,0] + 0.5 - r >= 0
            def make_wall_constraint(i, wall):
                def constraint(x):
                    pts = x[:2*N].reshape(N, 2)
                    r = x[-1]
                    if wall == 0:  # left
                        return pts[i, 0] + 0.5 - r
                    elif wall == 1:  # right
                        return 0.5 - pts[i, 0] - r
                    elif wall == 2:  # bottom
                        return pts[i, 1] + 0.5 - r
                    else:  # top
                        return 0.5 - pts[i, 1] - r
                return constraint
            
            for wall in range(4):
                constraints.append({
                    'type': 'ineq',
                    'fun': make_wall_constraint(i, wall)
                })
        
        return constraints
    
    # Due to the large number of constraints, SLSQP might be slow.
    # Let's use a simpler approach: minimize -r with constraints defined differently.
    
    # Alternative: Use SLSQP with a penalty-free approach but fewer constraints.
    # We'll only enforce the most binding constraints.
    
    # First, identify the most binding constraints from the current solution
    pts = pts.copy()
    r = r0
    
    # Compute all constraint values
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    pair_dists = D[i_idx, j_idx]
    pair_vals = pair_dists**2 - 4.0 * r**2
    
    # Find the most binding pair constraints (smallest values)
    num_bind_pairs = min(50, num_pairs)
    binding_pair_indices = np.argsort(pair_vals)[:num_bind_pairs]
    
    # Wall constraints
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5 - r,
        0.5 - pts[:, 0] - r,
        pts[:, 1] + 0.5 - r,
        0.5 - pts[:, 1] - r
    ])
    num_bind_walls = min(20, 4 * N)
    binding_wall_indices = np.argsort(wall_vals)[:num_bind_walls]
    
    # Build constraint list with only binding constraints
    constraints = []
    
    # Pair constraints
    for k in binding_pair_indices:
        i, j = i_idx[k], j_idx[k]
        def make_pair_constraint(i, j):
            def constraint(x):
                pts = x[:2*N].reshape(N, 2)
                r = x[-1]
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                return dx**2 + dy**2 - 4.0 * r**2
            return constraint
        constraints.append({
            'type': 'ineq',
            'fun': make_pair_constraint(i, j)
        })
    
    # Wall constraints
    for idx in binding_wall_indices:
        point_idx = idx // 4
        wall = idx % 4
        def make_wall_constraint(i, wall):
            def constraint(x):
                pts = x[:2*N].reshape(N, 2)
                r = x[-1]
                if wall == 0:
                    return pts[i, 0] + 0.5 - r
                elif wall == 1:
                    return 0.5 - pts[i, 0] - r
                elif wall == 2:
                    return pts[i, 1] + 0.5 - r
                else:
                    return 0.5 - pts[i, 1] - r
            return constraint
        constraints.append({
            'type': 'ineq',
            'fun': make_wall_constraint(point_idx, wall)
        })
    
    best_r = r0
    best_x = x0.copy()
    
    # Run SLSQP
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        
        if res.success:
            pts_curr = res.x[:2*N].reshape(N, 2)
            r_curr = res.x[-1]
            
            # Verify the solution
            r_actual = _get_radius(pts_curr)
            
            if r_actual > best_r:
                best_r = r_actual
                best_x = res.x.copy()
                
    except Exception:
        pass
    
    return best_r, best_x[:2*N].reshape(N, 2)

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Define multiple initializations
    inits = []
    
    # Init 1: Standard Hex
    pts_hex1 = _generate_hex_initial()
    if pts_hex1 is not None and len(pts_hex1) == N:
        inits.append(pts_hex1)
        
    # Init 2: Shifted Hex
    s = 0.0732
    points = []
    row_h = s * np.sqrt(3) / 2
    y = -0.6 + row_h / 2
    row_idx = 0
    while y < 0.6:
        shift = (s / 2.0) if row_idx % 2 == 0 else 0.0
        x = -0.6 + shift
        while x < 0.6:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
    pts_shifted = np.array(points)
    margin = 0.001
    mask = (pts_shifted[:, 0] >= -0.5 + margin) & (pts_shifted[:, 0] <= 0.5 - margin) & \
           (pts_shifted[:, 1] >= -0.5 + margin) & (pts_shifted[:, 1] <= 0.5 - margin)
    pts_shifted = pts_shifted[mask]
    if len(pts_shifted) >= N:
        dists_to_center = np.sum(pts_shifted**2, axis=1)
        indices = np.argsort(dists_to_center)
        pts_shifted = pts_shifted[indices[:N]]
        inits.append(pts_shifted)
        
    # Init 3: Slightly denser Hex
    s3 = 0.070
    points3 = []
    row_h3 = s3 * np.sqrt(3) / 2
    y3 = -0.6 + row_h3 / 2
    row_idx3 = 0
    while y3 < 0.6:
        shift3 = (s3 / 2.0) if row_idx3 % 2 == 1 else 0.0
        x3 = -0.6 + shift3
        while x3 < 0.6:
            points3.append([x3, y3])
            x3 += s3
        y3 += row_h3
    pts_dense = np.array(points3)
    mask3 = (pts_dense[:, 0] >= -0.5 + margin) & (pts_dense[:, 0] <= 0.5 - margin) & \
            (pts_dense[:, 1] >= -0.5 + margin) & (pts_dense[:, 1] <= 0.5 - margin)
    pts_dense = pts_dense[mask3]
    if len(pts_dense) >= N:
        dists_to_center3 = np.sum(pts_dense**2, axis=1)
        indices3 = np.argsort(dists_to_center3)
        pts_dense = pts_dense[indices3[:N]]
        inits.append(pts_dense)

    # Run optimization on each init
    num_inits = len(inits)
    if num_inits == 0:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        best_r = _get_radius(best_pts)
        inits = [best_pts]
        num_inits = 1
        
    # Allocate time: 70% for soft-min, 30% for SLSQP polish
    time_for_softmin = int(_TIME_LIMIT * 0.7)
    time_for_slsqp = _TIME_LIMIT - time_for_softmin - 5
    
    time_per_init = time_for_softmin / num_inits
    
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - t0
        remaining_for_phase = time_for_softmin - elapsed
        if remaining_for_phase <= 5:
            break
            
        t_limit = min(time_per_init, remaining_for_phase)
        
        r_opt, pts_opt = _run_soft_min_optimization(init_pts, time_limit=t_limit)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt
            
    # Phase: Polish with SLSQP
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - 5 - elapsed
        if remaining_time > 5:
            r_polish, pts_polish = _run_slsqp_polish(best_pts, time_limit=remaining_time)
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
                
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
