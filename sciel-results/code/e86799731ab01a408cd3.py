import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

# Triangle vertices
V0 = np.array([0, 0])
V1 = np.array([1, 0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    P = u*V0 + v*V1 + (1-u-v)*V2
    """
    w = 1.0 - u - v
    px = v + 0.5 * w
    py = H * w
    return np.column_stack((px, py))

def cart_to_bary(points):
    """
    Convert Cartesian (N,2) to barycentric (u, v) where P = u*V0 + v*V1 + w*V2, w=1-u-v.
    """
    x = points[:, 0]
    y = points[:, 1]
    v = x - 0.5 * (y / H)
    u = 1.0 - v - (y / H)
    return u, v

def calculate_areas(points):
    """
    points: (N, 2) array (Cartesian)
    Returns: array of areas for all triangles
    """
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    areas = 0.5 * np.abs(cross)
    return areas

def min_area_from_bary(params):
    """
    params: (2N,) array of barycentric coordinates [u0, v0, u1, v1, ...]
    Returns the minimum triangle area.
    """
    u = params[0::2]
    v = params[1::2]
    points = bary_to_cart(u, v)
    areas = calculate_areas(points)
    return np.min(areas)

def soft_min_neg(areas, beta):
    """
    Returns -SoftMin(areas) using stable log-sum-exp
    """
    min_area = np.min(areas)
    shifted = -beta * (areas - min_area)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    log_sum = -beta * min_area + log_sum_exp
    return log_sum / beta

def objective_soft(params, beta):
    """
    Soft minimum objective for global search.
    """
    u = params[0::2]
    v = params[1::2]
    points = bary_to_cart(u, v)
    areas = calculate_areas(points)
    return soft_min_neg(areas, beta)

def objective_hard(params):
    """
    Hard minimum objective for local refinement.
    """
    u = params[0::2]
    v = params[1::2]
    points = bary_to_cart(u, v)
    areas = calculate_areas(points)
    return -np.min(areas)

def make_bary_constraints():
    """
    Constraints for barycentric coordinates:
    u_i + v_i <= 1  =>  1 - u_i - v_i >= 0
    u_i >= 0, v_i >= 0 (handled by bounds)
    """
    cons = []
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[2*i] - p[2*i+1]
        })
    return cons

CONSTRAINTS = make_bary_constraints()

def clamp_point_to_triangle(x, y):
    """
    Clamp a single point (x, y) to be inside the equilateral triangle.
    """
    # Edge 1: y >= 0
    if y < 1e-12:
        y = 1e-12
    # Edge 2: y <= sqrt(3) * x
    if y > np.sqrt(3) * x + 1e-12:
        d = (np.sqrt(3) * x - y) / 4.0
        x = x - np.sqrt(3) * d
        y = y + d
    # Edge 3: y <= sqrt(3) * (1 - x)
    if y > np.sqrt(3) * (1 - x) + 1e-12:
        d = (np.sqrt(3) * x + y - np.sqrt(3)) / 4.0
        x = x - np.sqrt(3) * d
        y = y - d
        
    # Final safety check
    if y < 0 or y > np.sqrt(3) * x or y > np.sqrt(3) * (1 - x):
        cx, cy = 0.5, H / 3.0
        x = 0.9 * x + 0.1 * cx
        y = 0.9 * y + 0.1 * cy
        if y < 1e-12: y = 1e-12
        if y > np.sqrt(3) * x: y = np.sqrt(3) * x * 0.99
        if y > np.sqrt(3) * (1 - x): y = np.sqrt(3) * (1 - x) * 0.99
        
    return x, y

def clamp_points_to_triangle(points):
    """
    Clamp all points in (N, 2) array.
    """
    pts = points.copy()
    for i in range(N):
        pts[i, 0], pts[i, 1] = clamp_point_to_triangle(pts[i, 0], pts[i, 1])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_val = -1
    
    # Bounds for barycentric coords: u in [0, 1], v in [0, 1]
    bounds = [(0, 1), (0, 1)] * N
    
    # --- Phase 1: Global Search with Differential Evolution ---
    # Use a soft minimum objective to make the landscape smoother for DE
    beta_de = 50
    
    # DE can be slow, so limit iterations and population size
    # maxiter=100, popsize=15 (total 1500 evaluations)
    # Each evaluation is fast (vectorized area calc)
    
    try:
        if time.time() < deadline - 60:  # Reserve 60s for local refinement
            de_result = differential_evolution(
                lambda p: -objective_soft(p, beta_de),  # Minimize negative soft min
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=False,  # We will do our own local refinement
                init='sobol'
            )
            
            if de_result.success or de_result.fun < 0:
                best_params = de_result.x
                # Evaluate hard min
                val = min_area_from_bary(best_params)
                if val > best_val:
                    best_val = val
                    best_points = bary_to_cart(best_params[0::2], best_params[1::2])
                    best_points = clamp_points_to_triangle(best_points)
    except Exception as e:
        pass
    
    # --- Phase 2: Local Refinement with SLSQP ---
    # Start from the best point found by DE, and also from a few random good starts
    # to avoid getting stuck in a poor local minimum.
    
    starts = []
    if best_points is not None:
        u, v = cart_to_bary(best_points)
        # Clip barycentric coords to [0,1]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        # Normalize to ensure u+v <= 1
        s = u + v
        mask = s > 1
        if np.any(mask):
            u[mask] /= s[mask]
            v[mask] /= s[mask]
        params = np.zeros(2 * N)
        params[0::2] = u
        params[1::2] = v
        starts.append(params)
    
    # Add a few random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        params = np.zeros(2 * N)
        params[0::2] = u
        params[1::2] = v
        starts.append(params)
    
    for init_params in starts:
        if time.time() > deadline - 5:
            break
            
        current = init_params.copy()
        
        # Annealing betas for soft min, transitioning to hard min
        betas = [50, 100, 200, 500, 1000]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            try:
                # Use SLSQP with constraints
                # Objective: minimize -soft_min(areas, beta)
                # For very high beta, it approximates -min(areas)
                
                res = minimize(
                    objective_soft, 
                    current, 
                    args=(beta,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=CONSTRAINTS,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                
                if res.success:
                    current = res.x
            except Exception:
                continue
        
        # Final evaluation with hard min
        u_final = current[0::2]
        v_final = current[1::2]
        # Ensure feasibility
        if np.all(u_final >= -1e-8) and np.all(v_final >= -1e-8) and np.all(u_final + v_final <= 1 + 1e-8):
            u_final = np.clip(u_final, 0, 1)
            v_final = np.clip(v_final, 0, 1)
            final_points = bary_to_cart(u_final, v_final)
            final_points = clamp_points_to_triangle(final_points)
            
            areas = calculate_areas(final_points)
            val = np.min(areas)
            
            if val > best_val:
                best_val = val
                best_points = final_points
                
    # --- Phase 3: Nelder-Mead Polish (optional, if time remains) ---
    # Nelder-Mead is derivative-free and can sometimes escape SLSQP local minima
    # But it's slower. Only do it if we have time and haven't found a very good solution.
    
    if best_points is not None and time.time() < deadline - 10:
        # Convert best_points to barycentric
        u, v = cart_to_bary(best_points)
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        s = u + v
        mask = s > 1
        if np.any(mask):
            u[mask] /= s[mask]
            v[mask] /= s[mask]
        nm_params = np.zeros(2 * N)
        nm_params[0::2] = u
        nm_params[1::2] = v
        
        try:
            nm_result = minimize(
                objective_hard,
                nm_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if nm_result.fun < 0:  # Since objective is -min_area
                val = -nm_result.fun
                if val > best_val:
                    best_val = val
                    u_nm = nm_result.x[0::2]
                    v_nm = nm_result.x[1::2]
                    # Clamp barycentric
                    u_nm = np.clip(u_nm, 0, 1)
                    v_nm = np.clip(v_nm, 0, 1)
                    s_nm = u_nm + v_nm
                    mask_nm = s_nm > 1
                    if np.any(mask_nm):
                        u_nm[mask_nm] /= s_nm[mask_nm]
                        v_nm[mask_nm] /= s_nm[mask_nm]
                    best_points = bary_to_cart(u_nm, v_nm)
                    best_points = clamp_points_to_triangle(best_points)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        rng2 = np.random.default_rng(7)
        u = rng2.random(N)
        v = rng2.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        best_points = bary_to_cart(u, v)
        best_points = clamp_points_to_triangle(best_points)
        
    return best_points

