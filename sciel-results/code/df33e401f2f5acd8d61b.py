import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0

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

def min_area_and_grad(bary):
    """
    Compute the minimum area and its gradient with respect to barycentric coordinates.
    Returns (min_area, grad) where grad has shape (N, 2).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    areas = calc_areas_bary(bary)
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    
    # Compute gradient of the minimum area
    # The area for triangle (i,j,k) is |det| where det is a multilinear form in the barycentric coords
    # We need to handle the absolute value carefully
    
    # For the minimum triangle, compute gradient of |det|
    i, j, k = tri_idx[min_idx]
    
    ui, vi, wi = u[i], v[i], w[i]
    uj, vj, wj = u[j], v[j], w[j]
    uk, vk, wk = u[k], v[k], w[k]
    
    # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    # Gradient of det with respect to each point's barycentric coords
    # For point i:
    # d(det)/d(ui) = (vj*wk - vk*wj)
    # d(det)/d(vi) = -(uj*wk - uk*wj)
    # d(det)/d(wi) = (uj*vk - uk*vj)
    # But w_i = 1 - u_i - v_i, so we need to account for that
    
    # Gradient of det w.r.t. u_i, v_i (noting w_i = 1-u_i-v_i):
    # d(det)/d(u_i) = d(det)/d(ui) - d(det)/d(wi)  [since w_i depends on u_i]
    # Actually, let's be more careful. The barycentric coords are (u, v) with w = 1-u-v.
    # So the point is parameterized by (u, v).
    
    # Let me recompute. For point i, the contribution to det is:
    # ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + (1-ui-vi)*(uj*vk - uk*vj)
    # = ui*(vj*wk - vk*wj - uj*vk + uk*vj) - vi*(uj*wk - uk*wj + uj*vk - uk*vj) + (uj*vk - uk*vj)
    
    # d/d(ui) = (vj*wk - vk*wj - uj*vk + uk*vj)
    # d/d(vi) = -(uj*wk - uk*wj + uj*vk - uk*vj)
    
    d_det_d_ui = vj*wk - vk*wj - uj*vk + uk*vj
    d_det_d_vi = -(uj*wk - uk*wj + uj*vk - uk*vj)
    
    # For point j:
    # Contribution: -ui*(vj*wk - vk*wj) + (1-uj-vj)*... wait, let me redo this properly.
    
    # Actually, let me use a cleaner approach. The determinant is:
    # det = |ui vi wi; uj vj wj; uk vk wk|
    # where each row sums to 1.
    
    # The gradient w.r.t. the coordinates of point p (in terms of u_p, v_p) can be computed
    # by noting that changing u_p by du changes w_p by -du.
    
    # Let me compute the gradient numerically for safety, but use it in a gradient-based optimizer.
    
    # Actually, let's just compute the gradient numerically with central differences for robustness
    # since the analytical gradient has edge cases with the absolute value.
    
    grad = np.zeros_like(bary)
    eps = 1e-7
    
    for p in range(N):
        for c in range(2):
            bary_plus = bary.copy()
            bary_minus = bary.copy()
            bary_plus[p, c] += eps
            bary_minus[p, c] -= eps
            bary_plus = clamp_bary(bary_plus)
            bary_minus = clamp_bary(bary_minus)
            
            area_plus = np.min(calc_areas_bary(bary_plus))
            area_minus = np.min(calc_areas_bary(bary_minus))
            
            grad[p, c] = (area_plus - area_minus) / (2 * eps)
    
    # The objective is to maximize min_area, so we minimize -min_area
    # grad of -min_area is -grad of min_area
    return min_area, -grad

def objective_and_grad(bary_flat):
    """Objective function with gradient for SLSQP."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    min_area, grad = min_area_and_grad(bary)
    return -min_area, -grad

def simulated_annealing(bary_init, time_limit=30.0):
    """
    Simulated annealing for global search.
    """
    start_time = time.time()
    bary = bary_init.copy()
    best_bary = bary.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    T = 0.05  # Initial temperature
    cooling_rate = 0.9995
    min_T = 1e-6
    max_steps = 5000
    
    step_count = 0
    while time.time() - start_time < time_limit and T > min_T:
        # Generate a perturbation
        sigma = T * 0.5
        bary_new = bary + np.random.normal(0, sigma, bary.shape)
        bary_new = clamp_bary(bary_new)
        
        val_new = -hard_min_obj(bary_new.flatten())
        val_cur = -hard_min_obj(bary.flatten())
        
        delta = val_new - val_cur
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            bary = bary_new
            if val_new > best_val:
                best_val = val_new
                best_bary = bary.copy()
        
        T *= cooling_rate
        step_count += 1
        
        if step_count % 500 == 0:
            # Check time
            if time.time() - start_time > time_limit:
                break
    
    return best_bary

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_grid_init():
    """
    Generate a structured grid-based initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try various grid densities
    for n_rows in [2, 3, 4]:
        for n_cols in [3, 4, 5]:
            pts = []
            for r in range(n_rows + 1):
                for c in range(n_cols + 1):
                    # Barycentric coordinates for a grid
                    # Row r corresponds to w = r / n_rows
                    # Within row, u varies from 0 to 1-w
                    w_val = r / n_rows
                    if w_val > 1.0:
                        continue
                    u_max = 1.0 - w_val
                    for c_idx in range(n_cols + 1):
                        u_val = c_idx * u_max / n_cols
                        v_val = u_max - u_val
                        if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                            pts.append([u_val, v_val])
            
            if len(pts) >= N:
                # Select N points from the grid
                rng = np.random.default_rng(42)
                for _ in range(10):
                    if len(pts) == N:
                        bary = np.array(pts)
                    else:
                        idx = rng.choice(len(pts), N, replace=False)
                        bary = np.array(pts)[idx]
                    val = -hard_min_obj(bary.flatten())
                    if val > best_val:
                        best_val = val
                        best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_voronoi_init():
    """
    Generate a Voronoi-like initial configuration by placing points
    to maximize minimum distance, then using that as a starting point.
    """
    best_bary = None
    best_min_dist = -np.inf
    
    for seed in [42, 123, 456, 789, 1011]:
        rng = np.random.default_rng(seed)
        bary = np.column_stack([rng.random(N), rng.random(N)])
        bary = clamp_bary(bary)
        
        # Compute minimum pairwise distance in Cartesian
        cart = bary_to_cart(bary)
        dists = np.linalg.norm(cart[:, np.newaxis] - cart[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_bary = bary.copy()
    
    return best_bary

def refine_gradient(bary_init, time_limit=10.0):
    """
    Refine using gradient-based SLSQP with analytical gradient.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    
    # Stage 1: SLSQP with gradient
    try:
        def obj_grad(x):
            return objective_and_grad(x)
        
        bounds = [(0, 1)] * (N * 2)
        
        # Add inequality constraints: u >= 0, v >= 0, u+v <= 1
        # These are handled by bounds and clamping, but let's add explicit constraints
        # for the triangle containment
        
        result = minimize(
            lambda x: objective_and_grad(x)[0],
            bary_current.flatten(),
            jac=lambda x: objective_and_grad(x)[1],
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
        )
        
        if result.success or result.fun < hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 2: Another SLSQP pass with tighter tolerance
    try:
        result2 = minimize(
            lambda x: objective_and_grad(x)[0],
            bary_current.flatten(),
            jac=lambda x: objective_and_grad(x)[1],
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
    
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Grid-based initialization
    grid_init = generate_grid_init()
    refined_grid = refine_gradient(grid_init, time_limit=8.0)
    update_best(refined_grid)
    
    # 2. Voronoi-like initialization
    voronoi_init = generate_voronoi_init()
    refined_voronoi = refine_gradient(voronoi_init, time_limit=8.0)
    update_best(refined_voronoi)
    
    # 3. Simulated annealing from best so far
    if best_bary is not None and time.time() < deadline - 25:
        sa_result = simulated_annealing(best_bary, time_limit=15.0)
        update_best(sa_result)
        
        # Refine the SA result
        if time.time() < deadline - 10:
            refined_sa = refine_gradient(sa_result, time_limit=8.0)
            update_best(refined_sa)
    
    # 4. Multi-start with gradient refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best
            sigma = 0.03
            init = best_bary + rng.normal(0, sigma, best_bary.shape)
        else:
            # Fresh random
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init = np.column_stack([u, v])
        
        init = clamp_bary(init)
        refined = refine_gradient(init, time_limit=4.0)
        update_best(refined)
        
        # Occasionally do a short SA from the refined result
        if time.time() < deadline - 12 and restart_count % 5 == 0:
            sa_short = simulated_annealing(refined, time_limit=3.0)
            update_best(sa_short)
    
    # 5. Final high-precision gradient polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            result_final = minimize(
                lambda x: objective_and_grad(x)[0],
                best_bary.flatten(),
                jac=lambda x: objective_and_grad(x)[1],
                method='SLSQP',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 2000, 'ftol': 1e-18, 'disp': False}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One more pass
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_final2 = minimize(
                    lambda x: objective_and_grad(x)[0],
                    best_bary.flatten(),
                    jac=lambda x: objective_and_grad(x)[1],
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 3000, 'ftol': 1e-18, 'disp': False}
                )
                update_best(clamp_bary(result_final2.x.reshape(N, 2)))
            except Exception:
                pass
    
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
