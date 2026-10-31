import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
# Vertices of equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])

# Pre-compute combinations for speed
COMBOS = np.array(list(itertools.combinations(range(N), 3)))

def bary_to_cart(u, v):
    # w = 1 - u - v
    return u * V[0] + v * V[1] + (1.0 - u - v) * V[2]

def get_points_from_uv(uv):
    """
    uv: (N, 2) array of (u, v) barycentric coords
    Returns (N, 2) cartesian points
    """
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
    return points

def calc_min_area(points):
    p1 = points[COMBOS[:, 0]]
    p2 = points[COMBOS[:, 1]]
    p3 = points[COMBOS[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def soft_min_area(areas, beta):
    """
    Soft minimum of areas using log-sum-exp
    """
    x_vals = -beta * areas
    m = np.max(x_vals)
    if m > 700:
        return 1e10
    log_sum = m + np.log(np.sum(np.exp(x_vals - m)))
    return -log_sum / beta

def objective_soft(x, beta, penalty_weight):
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Penalty for u+v > 1 (outside triangle)
    viol = np.maximum(0.0, u + v - 1.0)
    penalty = penalty_weight * np.sum(viol**2)
    
    # Also penalize negative u or v
    viol_u = np.maximum(0.0, -u)
    viol_v = np.maximum(0.0, -v)
    penalty += penalty_weight * (np.sum(viol_u**2) + np.sum(viol_v**2))
    
    # Calculate points
    w = 1.0 - u - v
    points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
    
    # Calculate areas
    p1 = points[COMBOS[:, 0]]
    p2 = points[COMBOS[:, 1]]
    p3 = points[COMBOS[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    soft_min_val = soft_min_area(areas, beta)
    
    return -soft_min_val + penalty

def objective_hard(x):
    """
    Hard objective: negative of min area
    """
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Check validity
    if np.any(u + v > 1.0) or np.any(u < 0) or np.any(v < 0):
        return 1e10
    
    w = 1.0 - u - v
    points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
    
    p1 = points[COMBOS[:, 0]]
    p2 = points[COMBOS[:, 1]]
    p3 = points[COMBOS[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal-like pattern of 11 points
    """
    # Center point
    center = np.array([1/3, 1/3])
    
    # 6 points around center in hexagonal pattern
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    radius = 0.15
    hex_points = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    
    # 4 more points near corners/edges
    corner1 = np.array([0.1, 0.1])
    corner2 = np.array([0.8, 0.1])
    corner3 = np.array([0.5, 0.6])
    edge1 = np.array([0.5, 0.05])
    
    points = np.vstack([center, hex_points, corner1, corner2, corner3, edge1])
    
    # Convert to uv coordinates (they're already in cartesian-like space for equilateral)
    # Actually, for equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # The barycentric coords are just u, v where point = u*V0 + v*V1 + (1-u-v)*V2
    # So if we have cartesian (x, y), we need to convert back to (u, v)
    
    # For now, let's just use the values as (u, v) directly and check validity
    uv = points
    # Clamp to valid region
    uv[:, 0] = np.clip(uv[:, 0], 0, 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, 1 - uv[:, 0])
    
    return uv.flatten()

def generate_random_init(rng):
    """
    Generate random valid init
    """
    u = rng.random(N)
    v = rng.random(N) * (1.0 - u)
    return np.concatenate([u, v])

def generate_perturbed_init(base_uv, rng, scale=0.05):
    """
    Generate perturbed version of a base init
    """
    uv = base_uv.reshape(N, 2).copy()
    noise = rng.normal(0, scale, (N, 2))
    uv += noise
    # Clamp
    uv[:, 0] = np.clip(uv[:, 0], 0, 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, 1 - uv[:, 0])
    return uv.flatten()

def refine_local(x_init, deadline):
    """
    Multi-stage local refinement
    """
    x_current = x_init.copy()
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Stages: (beta, penalty_weight, maxiter)
    stages = [
        (10.0, 100.0, 80),
        (50.0, 500.0, 100),
        (200.0, 2000.0, 150),
        (1000.0, 10000.0, 200),
        (5000.0, 50000.0, 300)
    ]
    
    for beta, pw, maxiter in stages:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                objective_soft,
                x_current,
                args=(beta, pw),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < objective_soft(x_current, beta, pw):
                x_current = res.x
        except Exception:
            pass
    
    return x_current

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_x = None
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    inits = []
    
    # 1. Hexagonal init
    inits.append(generate_hexagonal_init())
    
    # 2. Random inits
    for _ in range(20):
        inits.append(generate_random_init(rng))
    
    # 3. Perturbed versions of hexagonal
    for _ in range(10):
        inits.append(generate_perturbed_init(inits[0], rng, scale=0.03))
    
    # Try each init with local refinement
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        x_refined = refine_local(init, deadline)
        
        # Evaluate
        uv = x_refined.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        
        if np.any(u + v > 1.0) or np.any(u < -1e-6) or np.any(v < -1e-6):
            continue
        
        w = 1.0 - u - v
        points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
        
        min_area = calc_min_area(points)
        
        if min_area > best_score:
            best_score = min_area
            best_x = x_refined.copy()
    
    # If we have time, do a global search on a reduced parameter space
    # Use RBF-like representation: points defined by a few control parameters
    if best_x is not None and time.time() < deadline - 30.0:
        # Try differential evolution on the full 22D space with tight bounds
        try:
            def de_objective(x):
                uv = x.reshape(N, 2)
                u = uv[:, 0]
                v = uv[:, 1]
                
                # Penalize invalid points heavily
                viol = np.maximum(0.0, u + v - 1.0) + np.maximum(0.0, -u) + np.maximum(0.0, -v)
                if np.sum(viol) > 1e-4:
                    return 1e6 + 100 * np.sum(viol)
                
                w = 1.0 - u - v
                points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
                
                p1 = points[COMBOS[:, 0]]
                p2 = points[COMBOS[:, 1]]
                p3 = points[COMBOS[:, 2]]
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                
                return -np.min(areas)
            
            bounds_de = [(0.0, 1.0)] * (2 * N)
            
            de_start = time.time()
            de_deadline = min(deadline - 5.0, de_start + 60.0)
            
            res_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=False,
                callback=lambda xk, convergence: None
            )
            
            if time.time() < de_deadline:
                x_de = res_de.x
                uv = x_de.reshape(N, 2)
                u = uv[:, 0]
                v = uv[:, 1]
                
                if np.all(u + v <= 1.0 + 1e-6) and np.all(u >= -1e-6) and np.all(v >= -1e-6):
                    # Polish
                    x_polished = refine_local(x_de, deadline)
                    uv = x_polished.reshape(N, 2)
                    u = uv[:, 0]
                    v = uv[:, 1]
                    
                    if np.any(u + v > 1.0) or np.any(u < -1e-6) or np.any(v < -1e-6):
                        x_polished = x_de
                    
                    w = 1.0 - u - v
                    points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
                    min_area = calc_min_area(points)
                    
                    if min_area > best_score:
                        best_score = min_area
                        best_x = x_polished
        except Exception:
            pass
    
    # Final polish on best_x
    if best_x is not None and time.time() < deadline - 10.0:
        try:
            x_final_polish = refine_local(best_x, deadline)
            uv = x_final_polish.reshape(N, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            
            if np.any(u + v > 1.0) or np.any(u < -1e-6) or np.any(v < -1e-6):
                x_final_polish = best_x
            
            w = 1.0 - u - v
            points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
            min_area = calc_min_area(points)
            
            if min_area > best_score:
                best_score = min_area
                best_x = x_final_polish
        except Exception:
            pass
    
    # Return best points
    if best_x is None:
        # Fallback
        u = np.random.rand(N)
        v = np.random.rand(N) * (1.0 - u)
        x_final = np.concatenate([u, v])
    else:
        x_final = best_x
    
    uv = x_final.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
    
    return points
