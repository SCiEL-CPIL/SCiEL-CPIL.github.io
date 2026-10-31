import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            # Try different subsets
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -min(calc_areas_bary(bary))
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def nlp_objective(x):
    """
    Objective for NLP: minimize -t (i.e. maximize t).
    x = [u0, v0, ..., u10, v10, t]
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    return -t

def nlp_gradient(x):
    """
    Gradient for NLP.
    d(-t)/dt = -1
    d(-t)/du_i = 0
    """
    grad = np.zeros_like(x)
    grad[-1] = -1.0
    return grad

def nlp_constraints(x):
    """
    Constraints for NLP:
    1. Barycentric constraints: u_i >= 0, v_i >= 0, u_i + v_i <= 1
    2. Area constraints: area_k - t >= 0 for all k
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    
    cons = []
    
    # Barycentric constraints
    for i in range(N):
        u = bary[i, 0]
        v = bary[i, 1]
        cons.append({'type': 'ineq', 'fun': lambda u=u: u, 'jac': lambda u=u, i=i: np.zeros(N*2+1) * 0 + (lambda x: np.array([1.0 if x[0]==i else 0.0 for x in range(N*2+1)]) if False else None)})
        # Simplified: we will use bounds for barycentric coords mostly, but SLSQP handles bounds.
        # However, u+v <= 1 is a coupling constraint.
        
    # Let's use the standard way:
    # Bounds will handle u_i >= eps, v_i >= eps.
    # We need u_i + v_i <= 1 - eps.
    for i in range(N):
        def fun(i=i):
            return 1.0 - 1e-9 - (x[i*2] + x[i*2+1])
        def jac(i=i):
            g = np.zeros(N*2+1)
            g[i*2] = -1.0
            g[i*2+1] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': fun, 'jac': jac})
        
    # Area constraints: area_k - t >= 0
    # We need to compute areas and their gradients.
    # This is computationally heavy to do in a loop for every constraint if we want analytic jac.
    # However, SLSQP can handle numerical differentiation if we don't provide jac, but it's slower.
    # Given 165 area constraints, providing analytic jac is preferred if possible, or at least efficient.
    
    # Let's try to provide the area constraints.
    # Since computing analytic gradients for all 165 triangles is complex to implement correctly in the fly,
    # and we have a time limit, we might rely on SLSQP's numerical differentiation for the area constraints
    # but provide analytic for the simple barycentric ones.
    # Or, we can just use numerical for all.
    
    # Actually, let's just provide the area constraints without jac first to see if it works.
    # If it's too slow, we'll optimize.
    
    areas = calc_areas_bary(bary)
    for k in range(len(areas)):
        def fun(k=k):
            return areas[k] - x[N*2]
        cons.append({'type': 'ineq', 'fun': fun})
        
    return cons

def solve_nlp(bary_init, t_init, time_limit=5.0):
    """
    Solve the NLP problem starting from bary_init and t_init.
    """
    start_time = time.time()
    
    bary_init = clamp_bary(bary_init)
    t_init = max(1e-6, min(calc_areas_bary(bary_init)) * 0.9) # Start slightly below min area
    
    x0 = np.zeros(N*2 + 1)
    x0[:N*2] = bary_init.flatten()
    x0[N*2] = t_init
    
    # Bounds:
    # u_i, v_i in [1e-9, 1-1e-9]
    # t in [0, 1] (max area is 0.5 for unit triangle? No, max area of sub-triangle is smaller, but 1 is safe upper bound)
    bounds = [(1e-9, 1 - 1e-9)] * (N * 2)
    bounds.append((0, 1))
    
    try:
        result = minimize(
            nlp_objective,
            x0,
            jac=nlp_gradient,
            method='SLSQP',
            bounds=bounds,
            constraints=nlp_constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if result.success:
            bary_sol = clamp_bary(result.x[:N*2].reshape(N, 2))
            t_sol = result.x[N*2]
            return bary_sol, t_sol
        else:
            # If it fails, maybe return the best feasible point found?
            # Or just return the initial?
            # Let's check if result.x is feasible enough.
            bary_sol = clamp_bary(result.x[:N*2].reshape(N, 2))
            t_sol = result.x[N*2]
            # Verify feasibility roughly
            areas = calc_areas_bary(bary_sol)
            if np.all(areas >= t_sol - 1e-6):
                return bary_sol, t_sol
            else:
                return bary_init, min(calc_areas_bary(bary_init))
                
    except Exception:
        return bary_init, min(calc_areas_bary(bary_init))

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min(calc_areas_bary(bary))
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try Hexagonal Init with NLP
    hex_init = generate_hexagonal_init()
    bary_sol, t_sol = solve_nlp(hex_init, min(calc_areas_bary(hex_init)), time_limit=10.0)
    update_best(bary_sol)
    
    # 2. Multi-start with NLP
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.05, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
            
        # For random starts, NLP might struggle if t_init is too high.
        # We use a heuristic t_init.
        t_init = min(calc_areas_bary(init)) * 0.8
        
        bary_sol, t_sol = solve_nlp(init, t_init, time_limit=5.0)
        update_best(bary_sol)
        
    # 3. Final Polish with NLP
    if best_bary is not None and time.time() < deadline - 5:
        t_init = min(calc_areas_bary(best_bary)) * 0.95
        bary_sol, t_sol = solve_nlp(best_bary, t_init, time_limit=10.0)
        update_best(bary_sol)
        
    # 4. If NLP failed completely, fallback to simple local search? 
    # But the task asks for a different strategy. We stick to NLP.
    # If best_bary is still None, use hex_init
    if best_bary is None:
        best_bary = hex_init
        
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
