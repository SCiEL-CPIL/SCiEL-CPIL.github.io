import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-based initial configuration.
    Hexagonal packing is optimal for maximizing minimum distance in 2D.
    """
    pts = []
    
    # Hexagonal grid points in barycentric coordinates
    # Use a triangular lattice pattern
    for i in range(4):
        for j in range(4):
            u = j / 3.0
            v = i / 3.0 + (j % 2) * 0.5 / 3.0
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
                pts.append([u, v])
    
    # Also try shifted hexagonal patterns
    offset_pts = []
    for i in range(4):
        for j in range(4):
            u = (j + 0.5) / 3.0
            v = (i + 0.5) / 3.0 + (j % 2) * 0.5 / 3.0
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
                offset_pts.append([u, v])
    
    # Select best 11 from combined set
    all_pts = pts + offset_pts
    if len(all_pts) < N:
        # Fill with random
        rng = np.random.default_rng(42)
        while len(all_pts) < N:
            u = rng.random()
            v = rng.random() * (1 - u)
            all_pts.append([u, v])
    
    # Score each subset and pick best
    best_val = -np.inf
    best_bary = np.array(all_pts[:N])
    
    rng = np.random.default_rng(42)
    for _ in range(50):
        idx = rng.choice(len(all_pts), N, replace=False)
        bary = np.array(all_pts)[idx]
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    return best_bary

def generate_vertex_edge_center_init():
    """
    Place points at vertices, edge midpoints, and center with slight perturbations.
    """
    # Vertices
    pts = [
        [0.0, 0.0],      # V0
        [1.0, 0.0],      # V1
        [0.0, 1.0],      # V2 (barycentric)
    ]
    
    # Edge midpoints
    pts.append([0.5, 0.0])  # Mid V0-V1
    pts.append([0.5, 0.5])  # Mid V1-V2
    pts.append([0.0, 0.5])  # Mid V0-V2
    
    # Center
    pts.append([1/3, 1/3])
    
    # Additional interior points - try symmetric positions
    extra_pts = [
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ]
    
    # Remove duplicates and ensure we have exactly N points
    unique_pts = []
    for p in pts + extra_pts:
        u, v = p
        if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
            # Check if already exists
            is_dup = False
            for existing in unique_pts:
                if abs(existing[0] - u) < 1e-6 and abs(existing[1] - v) < 1e-6:
                    is_dup = True
                    break
            if not is_dup:
                unique_pts.append([u, v])
    
    # If we have more than N, take first N; if fewer, add random
    if len(unique_pts) >= N:
        bary = np.array(unique_pts[:N])
    else:
        bary = np.array(unique_pts)
        while len(bary) < N:
            u = np.random.random()
            v = np.random.random() * (1 - u)
            bary = np.vstack([bary, [u, v]])
    
    return bary

def generate_grid_init():
    """Generate a grid-like initial configuration in barycentric coords."""
    best_bary = None
    best_val = -np.inf
    
    # Try different grid spacings
    for n_rows in [3, 4, 5]:
        for n_cols in [3, 4, 5]:
            pts = []
            for i in range(n_rows):
                for j in range(n_cols):
                    u = j / (n_cols - 1) if n_cols > 1 else 0
                    v = i / (n_rows - 1) if n_rows > 1 else 0
                    if u + v <= 1.0:
                        pts.append([u, v])
            
            if len(pts) >= N:
                # Try to select N points that maximize min area
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

def generate_symmetric_init():
    """
    Generate a symmetric initial configuration.
    3 points near vertices, 3 near edge midpoints, 3 in a ring, 1 at center.
    """
    pts = []
    
    # Near vertices (slightly inside)
    eps = 0.05
    pts.append([eps, eps])
    pts.append([1 - 2*eps, eps])
    pts.append([eps, 1 - 2*eps])
    
    # Near edge midpoints
    pts.append([0.5, 0.05])
    pts.append([0.5, 0.45])
    pts.append([0.05, 0.45])
    
    # Ring of 3 points
    r = 0.3
    center = [1/3, 1/3]
    for angle in [0, 120, 240]:
        theta = np.radians(angle)
        # Approximate circular motion in barycentric
        du = r * np.cos(theta)
        dv = r * np.sin(theta) * 0.866  # Adjust for aspect ratio
        u = center[0] + du
        v = center[1] + dv
        if u >= 0 and v >= 0 and u + v <= 1:
            pts.append([u, v])
    
    # Center
    pts.append([1/3, 1/3])
    
    # Ensure we have exactly N points
    while len(pts) < N:
        u = np.random.random()
        v = np.random.random() * (1 - u)
        pts.append([u, v])
    
    bary = np.array(pts[:N])
    return bary

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

def coordinate_descent(bary, max_iter=1000, step_size=0.01):
    """
    Coordinate descent refinement for the hard min objective.
    """
    bary = bary.copy()
    current_val = -hard_min_obj(bary.flatten())
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in range(2):
                original = bary[i, d]
                for step in [step_size, -step_size, step_size/2, -step_size/2, step_size/10, -step_size/10]:
                    new_val = bary[i, d] + step
                    if new_val < 0 or new_val > 1:
                        continue
                    bary[i, d] = new_val
                    new_bary = clamp_bary(bary)
                    val = -hard_min_obj(new_bary.flatten())
                    if val > current_val:
                        current_val = val
                        improved = True
                        break
                    else:
                        bary[i, d] = original
                # Reset if not improved in this direction
                bary[i, d] = original if not improved else bary[i, d]
        
        if not improved:
            break
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with adaptive beta
    for beta in [30.0, 80.0, 200.0]:
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
        
        if time.time() - start_time > time_limit * 0.25:
            break
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.6:
        return bary_current
    
    # Stage 3: Coordinate descent for fine-tuning
    try:
        bary_cd = coordinate_descent(bary_current, max_iter=500, step_size=0.005)
        val_cd = -hard_min_obj(bary_cd.flatten())
        val_curr = -hard_min_obj(bary_current.flatten())
        if val_cd > val_curr:
            bary_current = bary_cd
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 4: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-16}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
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

    # 1. Try multiple structured initializations
    inits = [
        generate_hexagonal_init(),
        generate_vertex_edge_center_init(),
        generate_grid_init(),
        generate_symmetric_init(),
    ]
    
    for init in inits:
        if time.time() > deadline - 25:
            break
        refined = refine_local(init, time_limit=5.0)
        update_best(refined)
    
    # 2. Global Search with Differential Evolution (multiple runs)
    if time.time() < deadline - 45:
        for de_seed in [42, 123, 456, 789]:
            if time.time() > deadline - 30:
                break
            try:
                def de_obj(x):
                    bary = x.reshape(N, 2)
                    bary = clamp_bary(bary)
                    areas = calc_areas_bary(bary)
                    return -np.min(areas)
                
                bounds = [(0, 1)] * (N * 2)
                result_de = differential_evolution(
                    de_obj,
                    bounds,
                    seed=de_seed,
                    maxiter=80,
                    popsize=18,
                    tol=1e-8,
                    polish=False
                )
                
                de_bary = clamp_bary(result_de.x.reshape(N, 2))
                refined_de = refine_local(de_bary, time_limit=5.0)
                update_best(refined_de)
            except Exception:
                pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 12:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution with adaptive sigma
            sigma = 0.02 + 0.015 * (restart_count % 4)
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=2.5)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 6:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # Coordinate descent final pass
        if best_bary is not None and time.time() < deadline - 1:
            try:
                bary_cd = coordinate_descent(best_bary, max_iter=1000, step_size=0.001)
                update_best(bary_cd)
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 5000, 'ftol': 1e-17}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
