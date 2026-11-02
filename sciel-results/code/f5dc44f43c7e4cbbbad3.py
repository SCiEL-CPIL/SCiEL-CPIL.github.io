import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill

def _objective(params):
    """Maximize sum of radii, so minimize -sum(r)."""
    r = params[2::3]
    return -np.sum(r)

def _objective_grad(params):
    """Gradient of objective."""
    g = np.zeros_like(params)
    g[2::3] = -1.0
    return g

def _constraints(params):
    """
    Returns a list of constraint dictionaries for SLSQP.
    Inequality constraints: fun(x) >= 0
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    constraints = []
    
    # Wall constraints
    # x_i >= r_i  => x_i - r_i >= 0
    # 1 - x_i >= r_i => 1 - x_i - r_i >= 0
    # y_i >= r_i => y_i - r_i >= 0
    # 1 - y_i >= r_i => 1 - y_i - r_i >= 0
    
    # We can batch these or define them individually. 
    # Defining them individually is clearer for gradient checking but slower.
    # Let's define functions that return arrays.
    
    # Actually, SLSQP accepts a list of dicts, each with 'fun' and 'jac'.
    # It's more efficient to define fun and jac that return arrays for all constraints of a type.
    
    # Type 1: Walls
    def wall_fun(p):
        pp = p.reshape(N, 3)
        xx = pp[:, 0]
        yy = pp[:, 1]
        rr = pp[:, 2]
        c1 = xx - rr
        c2 = 1.0 - xx - rr
        c3 = yy - rr
        c4 = 1.0 - yy - rr
        return np.concatenate([c1, c2, c3, c4])
        
    def wall_jac(p):
        # Shape (4N, 3N)
        jac = np.zeros((4*N, 3*N))
        # c1: x_i - r_i
        # d/dx_i = 1, d/dr_i = -1
        for i in range(N):
            jac[i, 3*i] = 1.0
            jac[i, 3*i+2] = -1.0
        # c2: 1 - x_i - r_i
        # d/dx_i = -1, d/dr_i = -1
        for i in range(N):
            jac[N+i, 3*i] = -1.0
            jac[N+i, 3*i+2] = -1.0
        # c3: y_i - r_i
        # d/dy_i = 1, d/dr_i = -1
        for i in range(N):
            jac[2*N+i, 3*i+1] = 1.0
            jac[2*N+i, 3*i+2] = -1.0
        # c4: 1 - y_i - r_i
        # d/dy_i = -1, d/dr_i = -1
        for i in range(N):
            jac[3*N+i, 3*i+1] = -1.0
            jac[3*N+i, 3*i+2] = -1.0
        return jac

    constraints.append({'type': 'ineq', 'fun': wall_fun, 'jac': wall_jac})
    
    # Type 2: Pairwise non-overlap
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0 is NOT differentiable at 0 and can be negative.
    # Better: dist - (r_i + r_j) >= 0
    # This is non-smooth at dist=0. SLSQP might struggle.
    # Alternative: dist^2 - (r_i+r_j)^2 >= 0 is smooth but allows "tunneling" if not careful? No, it's convex in dist.
    # However, dist^2 - r^2 is not a standard constraint form for SLSQP if it becomes negative.
    # Let's stick to dist - (r_i + r_j) >= 0.
    # To handle the non-smoothness at dist=0, we can add a small epsilon to dist in the calculation, 
    # or rely on the fact that if they overlap significantly, the gradient pushes them apart.
    
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    def pair_fun(p):
        pp = p.reshape(N, 3)
        xx = pp[:, 0]
        yy = pp[:, 1]
        rr = pp[:, 2]
        
        dx = xx[iu[0]] - xx[iu[1]]
        dy = yy[iu[0]] - yy[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy)
        # Avoid division by zero in gradient if dist is 0
        dist_safe = np.maximum(dist, 1e-8)
        # Constraint: dist - (r_i + r_j) >= 0
        return dist_safe - (rr[iu[0]] + rr[iu[1]])
        
    def pair_jac(p):
        pp = p.reshape(N, 3)
        xx = pp[:, 0]
        yy = pp[:, 1]
        rr = pp[:, 2]
        
        dx = xx[iu[0]] - xx[iu[1]]
        dy = yy[iu[0]] - yy[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy)
        dist_safe = np.maximum(dist, 1e-8)
        
        jac = np.zeros((n_pairs, 3*N))
        
        idx_i = iu[0]
        idx_j = iu[1]
        
        # d/dx_i = dx/dist, d/dx_j = -dx/dist
        jac[:, 3*idx_i] = dx / dist_safe
        jac[:, 3*idx_j] = -dx / dist_safe
        
        # d/dy_i = dy/dist, d/dy_j = -dy/dist
        jac[:, 3*idx_i + 1] = dy / dist_safe
        jac[:, 3*idx_j + 1] = -dy / dist_safe
        
        # d/dr_i = -1, d/dr_j = -1
        jac[:, 3*idx_i + 2] = -1.0
        jac[:, 3*idx_j + 2] = -1.0
        
        return jac

    constraints.append({'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac})
    
    # Type 3: Minimum radius
    def min_r_fun(p):
        return p[2::3] - 1e-6
        
    def min_r_jac(p):
        jac = np.zeros((N, 3*N))
        for i in range(N):
            jac[i, 3*i+2] = 1.0
        return jac
        
    constraints.append({'type': 'ineq', 'fun': min_r_fun, 'jac': min_r_jac})
    
    return constraints

def _init_voronoi(seed):
    rng = np.random.RandomState(seed)
    # Random points
    points = rng.uniform(0.1, 0.9, size=(N, 2))
    
    # Calculate initial radii based on distance to neighbors and walls
    radii = np.zeros(N)
    for i in range(N):
        # Distance to walls
        dw = np.min([points[i,0], 1-points[i,0], points[i,1], 1-points[i,1]])
        # Distance to nearest neighbor
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        dn = np.min(dists)
        
        # Heuristic: half the minimum distance
        radii[i] = 0.5 * np.min([dw, dn]) * 0.9
        
    params = np.zeros(3*N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    # Add small jitter
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params

def _init_grid(seed):
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    # Create a grid of side x side points, take first N
    xs, ys = np.meshgrid(np.linspace(0.5/side, 1-0.5/side, side), 
                         np.linspace(0.5/side, 1-0.5/side, side))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    points = points[:N]
    
    # Estimate radius as half the spacing
    r = 0.5 / side * 0.95
    radii = np.full(N, r)
    
    params = np.zeros(3*N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    # Add jitter to break symmetry
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params

def _enforce_validity(params):
    """
    Post-process to ensure strict validity.
    If SLSQP leaves slight violations, we fix them by shrinking radii.
    """
    p = params.reshape(N, 3).copy()
    # Clip positions to [0, 1]
    p[:, 0] = np.clip(p[:, 0], 0, 1)
    p[:, 1] = np.clip(p[:, 1], 0, 1)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clamp to walls
    wall_dist = np.minimum(np.minimum(p[:, 0], 1-p[:, 0]), 
                           np.minimum(p[:, 1], 1-p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], wall_dist)
    
    # Resolve overlaps by shrinking smaller circle
    for _ in range(50):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx*dx + dy*dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-9
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller one, or split if equal
            if p[i, 2] < p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
    
    return p

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_params = None
    best_score = -np.inf
    
    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N
    cons = _constraints(np.zeros(3*N)) # Constraints are static in structure
    
    # List of initializers
    initializers = [
        lambda s: _init_voronoi(s),
        lambda s: _init_grid(s),
    ]
    
    # Multiple seeds to explore different local minima
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 11, 22, 33, 44, 55]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        for init_fn in initializers:
            if time.time() > deadline - 5:
                break
                
            x0 = init_fn(seed)
            x0 = _enforce_validity(x0).flatten()
            
            try:
                res = minimize(
                    _objective,
                    x0,
                    method='SLSQP',
                    jac=_objective_grad,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 200, 'ftol': 1e-9}
                )
                
                p = res.x.reshape(N, 3)
                p = _enforce_validity(p)
                s = np.sum(p[:, 2])
                
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
                    
            except Exception as e:
                continue
                
    if best_params is None:
        # Fallback
        best_params = _enforce_validity(_init_grid(42)).reshape(N, 3)
        best_score = np.sum(best_params[:, 2])
        
    # Final Refinement: Try to grow circles further
    # We can do a simple local search: try to increase each radius slightly and see if it's valid.
    # If not, try to move the circle to make space.
    
    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 5.0:
        p = best_params.copy()
        # Simple local improvement
        for _ in range(10): # Iterations
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(N):
                # Try to increase radius by 0.1%
                new_r = p[i, 2] * 1.001
                if new_r > 0.5: continue
                p[i, 2] = new_r
                
                # Check validity
                valid = True
                # Walls
                if p[i, 2] > p[i, 0] or p[i, 2] > 1-p[i, 0] or p[i, 2] > p[i, 1] or p[i, 2] > 1-p[i, 1]:
                    valid = False
                # Pairs
                if valid:
                    for j in range(N):
                        if i == j: continue
                        dx = p[i, 0] - p[j, 0]
                        dy = p[i, 1] - p[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < p[i, 2] + p[j, 2]:
                            valid = False
                            break
                            
                if valid:
                    improved = True
                else:
                    p[i, 2] /= 1.001 # Revert
            
            if not improved:
                break
                
        s = np.sum(p[:, 2])
        if s > best_score:
            best_score = s
            best_params = p
            
    return best_params
