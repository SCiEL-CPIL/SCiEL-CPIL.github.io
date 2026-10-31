import numpy as np
from itertools import combinations
import time
from scipy.optimize import differential_evolution, minimize

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def clamp_bary(bary, eps=1e-10):
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

def min_area(bary):
    """Compute the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def objective(x):
    """Objective function for minimization: negative of min area."""
    bary = x.reshape(N, 2)
    # Ensure validity
    bary = clamp_bary(bary)
    val = min_area(bary)
    return -val

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
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
            rng = np.random.default_rng(42)
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = min_area(bary)
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using high-precision local optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = min_area(bary_current)
    
    # Stage 1: Nelder-Mead for robust local improvement
    try:
        result1 = minimize(
            lambda x: -min_area(x.reshape(N, 2)),
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        candidate = clamp_bary(result1.x.reshape(N, 2))
        val = min_area(candidate)
        if val > current_val:
            bary_current = candidate
            current_val = val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.6:
        return bary_current, current_val
        
    # Stage 2: SLSQP for final polish with constraints
    try:
        result2 = minimize(
            lambda x: -min_area(x.reshape(N, 2)),
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        candidate = clamp_bary(result2.x.reshape(N, 2))
        val = min_area(candidate)
        if val > current_val:
            bary_current = candidate
            current_val = val
    except Exception:
        pass
        
    return bary_current, current_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate a good initial guess
    hex_init = generate_hexagonal_init()
    update_best(hex_init)

    # 2. Global Search using Differential Evolution
    # DE is effective for multimodal problems and doesn't require gradients.
    # We use a relatively large population and max iterations to explore the space.
    
    # Bounds for barycentric coordinates: u in [0,1], v in [0,1]
    # Note: The constraint u+v <= 1 is handled by clamp_bary inside the objective,
    # but it's better to provide tighter bounds if possible or rely on the penalty/clamp.
    # Since clamp_bary projects back to the valid region, DE will effectively search the surface.
    
    bounds = [(0, 1)] * (N * 2)
    
    # We will run DE for a fixed time budget
    de_time_budget = 60.0
    
    def de_callback(xk, convergence):
        # Update best found so far
        bary = xk.reshape(N, 2)
        val = min_area(bary)
        if val > best_val:
            nonlocal best_val, best_bary
            best_val = val
            best_bary = bary.copy()
            
        # Check time
        if time.time() - start_time > deadline - 20.0:
            # Signal to stop by raising an exception or using a flag
            # scipy's DE doesn't have a direct "stop" callback return, 
            # but we can check time in the main loop if we implement custom, 
            # or rely on maxiter. 
            # For simplicity, we'll just let it run or use a wrapper.
            # Actually, scipy.optimize.differential_evolution doesn't support early stopping via callback directly in a clean way without breaking.
            # We'll estimate maxiter based on time.
            pass

    # Estimate maxiter. Each function eval is fast (~0.1ms).
    # Population size 50, maxiter 200 -> 10000 evals.
    # 10000 * 0.1ms = 1s. 
    # So we can afford much more.
    # Let's try popsize=50, maxiter=500.
    
    try:
        result_de = differential_evolution(
            objective,
            bounds,
            maxiter=1000,
            popsize=50,
            tol=1e-8,
            seed=42,
            polish=False,  # We do our own polish
            init='latinhypercube',
            updating='immediate'
        )
        
        # Check time
        if time.time() < deadline - 10:
            candidate = clamp_bary(result_de.x.reshape(N, 2))
            update_best(candidate)
            
    except Exception as e:
        pass

    # 3. Final High-Precision Polish
    if best_bary is not None and time.time() < deadline - 5:
        refined, val = refine_local(best_bary, time_limit=10.0)
        update_best(refined)
        
    # 4. One last SLSQP pass if time permits
    if best_bary is not None and time.time() < deadline - 1:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                lambda x: -min_area(x.reshape(N, 2)),
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
