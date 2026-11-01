"""Combined program: Hexagonal init + Active-Constraint SLSQP Feasibility Polish."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try a range of spacings to find the best fit for N=200
    # Spacing d is the distance between centers.
    for d in np.linspace(0.055, 0.075, 40):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            # Take the first N points
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _get_active_pairs(c, r_target, margin=0.01):
    """
    Get indices of pairs that are close to the constraint boundary.
    We include pairs with distance < 2*r_target + margin.
    """
    n = c.shape[0]
    tree = cKDTree(c)
    # Query points within distance 2*r_target + margin
    dist_limit = 2 * r_target + margin
    
    pairs = set()
    for i in range(n):
        neighbors = tree.query_ball_point(c[i], dist_limit)
        for j in neighbors:
            if i < j:
                pairs.add((i, j))
    
    return list(pairs)


def _slsqp_feasibility(c0, r_target, max_iter=100):
    """
    Attempt to satisfy constraints for radius r_target starting from c0.
    Returns (success, new_c).
    """
    n = N
    pairs = _get_active_pairs(c0, r_target, margin=0.005)
    
    # If no pairs are active, the constraint is purely boundary.
    # SLSQP handles this, but we can shortcut if needed.
    # We always include boundary constraints.
    
    def constraints_func(x):
        c = x.reshape(-1, 2)
        # Boundary constraints: 0.5 - |c_ik| - r_target >= 0
        bd = 0.5 - np.abs(c) - r_target
        
        # Pairwise constraints: ||c_i - c_j|| - 2*r_target >= 0
        if pairs:
            pdist_vals = np.zeros(len(pairs))
            for idx, (i, j) in enumerate(pairs):
                diff = c[i] - c[j]
                pdist_vals[idx] = np.linalg.norm(diff) - 2 * r_target
            return np.concatenate([bd.ravel(), pdist_vals])
        else:
            return bd.ravel()
        
    def jac_func(x):
        c = x.reshape(-1, 2)
        jac_rows = []
        
        # Boundary Jacobian: d(0.5 - |c_ik| - r)/dx = -sign(c_ik)
        for i in range(n):
            for k in range(2):
                row = np.zeros(2 * n)
                row[2 * i + k] = -np.sign(c[i, k])
                jac_rows.append(row)
        
        # Pairwise Jacobian
        if pairs:
            for idx, (i, j) in enumerate(pairs):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    dist = 1e-12
                row = np.zeros(2 * n)
                row[2 * i:2 * i + 2] = diff / dist
                row[2 * j:2 * j + 2] = -diff / dist
                jac_rows.append(row)
        
        return np.array(jac_rows)
    
    try:
        result = minimize(
            lambda x: 0.0, # Constant objective
            c0.flatten(),
            method='SLSQP',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
            constraints=[
                {'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}
            ],
            options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Check if constraints are actually satisfied
        c_check = c_new
        bd = 0.5 - np.abs(c_check)
        if np.min(bd) < r_target - 1e-9:
            return False, c_new
            
        if pairs:
            for (i, j) in pairs:
                d = np.linalg.norm(c_check[i] - c_check[j])
                if d < 2 * r_target - 1e-9:
                    return False, c_new
        
        return True, c_new
    except Exception:
        return False, c0


def _polish(c0, max_time=100.0):
    """
    High-precision polish using bisection-style SLSQP feasibility loop.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Adaptive step size
    step = 1e-4
    
    # Try to increase radius
    while time.time() - start_time < max_time:
        r_target = best_r + step
        
        success, c_new = _slsqp_feasibility(best_c, r_target, max_iter=150)
        
        if success:
            # Verify actual radius
            r_actual = _compute_radius(c_new)
            if r_actual > best_r:
                best_c = c_new
                best_r = r_actual
                # Increase step slightly on success
                step = min(step * 1.2, 5e-3)
            else:
                # Feasibility claimed but radius didn't improve? 
                # Might be numerical noise. Keep best_c, but maybe reduce step.
                step = step * 0.8
        else:
            # Failed to find feasible configuration for r_target
            # Reduce step
            step = step * 0.5
            
            # If step becomes too small, we've likely hit the local limit
            if step < 1e-7:
                break
                
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # 1. Hexagonal Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Primary Polish on Hexagonal Start
    # Allocate most time here
    if time.time() < deadline - 30:
        remaining = deadline - time.time() - 30
        c_opt, r_opt = _polish(c_hex, max_time=remaining)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 3. Secondary Polish with Perturbation
    # If time remains, try a small perturbation of the best solution to escape local minima
    if best_c is not None and time.time() < deadline - 20:
        np.random.seed(42)
        c_pert = best_c + np.random.normal(0, 0.002, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        
        r_pert = _compute_radius(c_pert)
        # Only polish if perturbation didn't break constraints significantly
        if r_pert > best_radius * 0.99:
            remaining = deadline - time.time() - 20
            c_opt2, r_opt2 = _polish(c_pert, max_time=remaining)
            if r_opt2 > best_radius:
                best_radius = r_opt2
                best_c = c_opt2.copy()
                
    # 4. Final Polish on Best
    if best_c is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        c_final, r_final = _polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
