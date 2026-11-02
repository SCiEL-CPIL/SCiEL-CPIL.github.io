import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4.0

# Precompute indices for all 165 triangles
_triplets = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _triplets])
_T_J = np.array([t[1] for t in _triplets])
_T_K = np.array([t[2] for t in _triplets])

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y).
    w = 1 - u - v.
    Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2).
    P = u*A + v*B + w*C
    x = v + 0.5*w
    y = 0.5*sqrt(3)*w
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = 0.5 * S3 * w
    return np.column_stack((x, y))

def _cart_to_bary(cart):
    """
    Convert Cartesian (x, y) to barycentric coordinates (u, v).
    """
    x = cart[:, 0]
    y = cart[:, 1]
    w = 2.0 * y / S3
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack((u, v))

def _compute_areas_cart(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _min_area_bary(bary, beta=50.0, hard_min=False):
    """
    Compute the soft-min or hard-min triangle area from barycentric coordinates.
    """
    cart = _bary_to_cart(bary)
    areas = _compute_areas_cart(cart)
    if hard_min:
        return np.min(areas)
    min_a = np.min(areas)
    # Log-sum-exp for numerical stability
    # shift = min_a
    # log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    # return min_a - log_sum / beta
    # Simpler soft min:
    return min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta

def _clip_bary(bary):
    """
    Clip barycentric coordinates to the valid simplex u>=0, v>=0, u+v<=1.
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip u and v to [0, 1]
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    # Enforce u + v <= 1
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        # Scale down u and v to make sum 1
        # This projects onto the line u+v=1
        # u_new = u / sum, v_new = v / sum
        u[mask] = u[mask] / sum_uv[mask]
        v[mask] = v[mask] / sum_uv[mask]
        
    return np.column_stack((u, v))

def _objective_slsqp(params, beta):
    """
    Objective for SLSQP: Minimize negative soft-min area.
    """
    bary = params.reshape(N, 2)
    # SLSQP handles bounds and constraints, but we can clip for safety if it drifts
    # However, SLSQP should respect the constraint u+v<=1.
    val = -_min_area_bary(bary, beta=beta)
    return val

def _slsqp_optimize(bary_init, beta, time_limit):
    """
    Optimize using SLSQP with constraints.
    """
    x0 = bary_init.flatten()
    
    # Bounds: u in [0, 1], v in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Constraints: u_i + v_i <= 1 for each i
    # This is a linear constraint: c_i(x) = u_i + v_i - 1 <= 0
    constraints = []
    for i in range(N):
        def _con(i=i):
            return -(x0[2*i] + x0[2*i+1] - 1.0) # Wait, SLSQP expects >= 0 for 'ineq'
            # Standard form: g(x) >= 0
            # We want u+v <= 1 => 1 - u - v >= 0
            return 1.0 - x0[2*i] - x0[2*i+1]
        constraints.append({'type': 'ineq', 'fun': _con})
        
    # Note: The constraint function above uses x0 which is captured in closure? 
    # No, minimize passes x to fun. We need to define fun(x) properly.
    
    def _con_fun(x, idx):
        u = x[2*idx]
        v = x[2*idx+1]
        return 1.0 - u - v
        
    constraints = [{'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]} for i in range(N)]

    def _obj(x):
        b = x.reshape(N, 2)
        return -_min_area_bary(b, beta=beta)
        
    try:
        res = minimize(
            _obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return bary_init

def _nelder_mead_bary(bary_init, beta, max_iter=1000):
    """
    Nelder-Mead on barycentric coordinates with clipping.
    """
    x0 = bary_init.flatten()
    
    def _obj(x):
        b = x.reshape(N, 2)
        b = _clip_bary(b)
        return -_min_area_bary(b, beta=beta)
        
    try:
        res = minimize(
            _obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        return _clip_bary(res.x.reshape(N, 2))
    except Exception:
        return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    rng = np.random.default_rng(12345)
    
    best_bary = None
    best_score = -np.inf
    
    def _update_best(bary):
        nonlocal best_bary, best_score
        score = _min_area_bary(bary, hard_min=True)
        if score > best_score:
            best_score = score
            best_bary = bary.copy()
            
    # Initial symmetric configuration in Cartesian
    init_cart = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    
    init_bary = _cart_to_bary(init_cart)
    init_bary = _clip_bary(init_bary)
    _update_best(init_bary)
    
    # Generate diverse starts
    starts = [init_bary]
    for _ in range(20):
        # Random points in simplex
        # Generate uniform random in square, reject if u+v>1
        # Or use Dirichlet(1,1,1)
        u, v, w = rng.dirichlet([1.0, 1.0, 1.0], size=(N, 3))
        bary = np.column_stack((u, v))
        starts.append(bary)
        
    # Add perturbed versions of init
    for _ in range(10):
        perturbed = init_bary + rng.normal(0, 0.1, size=(N, 2))
        perturbed = _clip_bary(perturbed)
        starts.append(perturbed)
        
    n_starts = 30
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break
            
        bary = starts[s % len(starts)].copy()
        # Add small noise
        bary = bary + rng.normal(0, 0.01, size=(N, 2))
        bary = _clip_bary(bary)
        
        _update_best(bary)
        
        # Annealed beta schedule for SLSQP
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 15:
                break
                
            # SLSQP step
            bary = _slsqp_optimize(bary, beta, time_limit=deadline - time.time() - 10)
            bary = _clip_bary(bary)
            _update_best(bary)
            
            # Nelder-Mead polish
            if time.time() > deadline - 10:
                break
            bary = _nelder_mead_bary(bary, beta, max_iter=200)
            _update_best(bary)
            
    # Final polish on the best found
    if best_bary is not None and time.time() < deadline - 5:
        # Use a high beta for Nelder-Mead
        bary = _nelder_mead_bary(best_bary, beta=500.0, max_iter=3000)
        _update_best(bary)
        
    # Convert best barycentric to Cartesian
    if best_bary is None:
        best_bary = init_bary
        
    best_cart = _bary_to_cart(best_bary)
    return best_cart
