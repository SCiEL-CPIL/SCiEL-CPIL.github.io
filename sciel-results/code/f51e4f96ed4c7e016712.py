import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 140.0  # Safety margin for 150s limit

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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
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

def energy_obj(bary_flat, p=2.0):
    """
    Energy-based objective: sum of (1/area)^p over all triangles.
    This is smooth and helps find configurations where all areas are reasonably large.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Add small epsilon to avoid division by zero
    areas = np.maximum(areas, 1e-12)
    energy = np.sum(areas**(-p))
    return energy

def epigraph_obj(params):
    """
    Epigraph formulation: optimize over (bary, t) where t is the minimum area.
    Minimize -t subject to all areas >= t (penalty-augmented).
    params: [u0, v0, u1, v1, ..., u10, v10, t]
    """
    bary_flat = params[:N*2]
    t = params[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for violating areas >= t
    violations = np.maximum(0, t - areas)
    penalty = 50.0 * np.sum(violations**2)
    
    # Objective: minimize -t + penalty
    return -t + penalty

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try a range of scales to find a good starting point
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

def coordinate_descent(bary_init, time_limit=3.0, max_iter=2000):
    """
    Coordinate-descent refinement: optimizes one coordinate at a time.
    This can escape local minima that gradient-based methods get stuck in.
    """
    start_time = time.time()
    bary = clamp_bary(bary_init.copy())
    current_val = -hard_min_obj(bary.flatten())
    
    # Pre-compute for each coordinate which triangles it participates in
    # For point i, coordinate c (0=u, 1=v), find all triangles containing i
    coord_tri_mask = []  # For each (i, c), mask of triangles containing point i
    for i in range(N):
        for c in range(2):
            mask = (tri_idx == i).any(axis=1)
            coord_tri_mask.append(mask)
    
    # Also pre-compute: for a given triangle (i,j,k) and point i, 
    # the area as a function of u_i (or v_i) is linear
    # det = u_i*(vj*wk - vk*wj) - v_i*(uj*wk - uk*wj) + w_i*(uj*vk - uk*vj)
    # where w_i = 1 - u_i - v_i
    # So if we vary u_i: det = u_i*(vj*wk - vk*wj) - v_i*(uj*wk - uk*wj) + (1-u_i-v_i)*(uj*vk - uk*vj)
    # = u_i * [(vj*wk - vk*wj) - (uj*vk - uk*vj)] + const
    # So area is linear in u_i (and in v_i)
    
    # For each triangle, pre-compute coefficients for each point
    # det = a_i * u_i + b_i * v_i + c_i for point i in triangle (i,j,k)
    # Actually, let's just do it numerically - for each coordinate, 
    # the min area as a function of that coordinate is piecewise linear
    # (since each triangle's area is linear in that coordinate)
    # So we can find the optimal value by checking breakpoints
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            for c in range(2):
                # Current value of this coordinate
                cur = bary[i, c]
                
                # Find the range of this coordinate
                if c == 0:  # u coordinate
                    lo = 1e-9
                    hi = 1 - bary[i, 1] - 1e-9
                else:  # v coordinate
                    lo = 1e-9
                    hi = 1 - bary[i, 0] - 1e-9
                
                if hi <= lo:
                    continue
                
                # The minimum area as a function of this coordinate is
                # the minimum of 165 linear functions (one per triangle)
                # We want to maximize this minimum.
                # The optimal point is either at a breakpoint (where two 
                # triangle areas are equal) or at a boundary.
                
                # Evaluate at several candidate points:
                # - current value
                # - boundaries
                # - breakpoints (where any two triangle areas are equal)
                
                candidates = [lo, hi, cur]
                
                # Find breakpoints: for each pair of triangles, 
                # where their areas are equal as functions of this coordinate
                # area_t(x) = a_t * x + b_t (linear in x)
                # breakpoint: a_t1 * x + b_t1 = a_t2 * x + b_t2
                # x = (b_t2 - b_t1) / (a_t1 - a_t2)
                
                # Compute area coefficients for all triangles w.r.t. this coordinate
                # For triangle (p, q, r), area = |det| where det is linear in x
                # Let's compute a_t and b_t for each triangle
                
                # Actually, let's just sample intelligently.
                # The piecewise-linear max-min function has at most 165*165/2 breakpoints,
                # but most are outside [lo, hi]. Let's use a smarter approach:
                # ternary search won't work (not unimodal), but we can do:
                # evaluate at many points and take the best, then refine.
                
                # Strategy: evaluate at 20 points in [lo, hi], take best,
                # then refine locally.
                n_samples = 20
                xs = np.linspace(lo, hi, n_samples)
                best_x = cur
                best_val_x = current_val
                
                for x in xs:
                    bary[i, c] = x
                    val = -hard_min_obj(bary.flatten())
                    if val > best_val_x:
                        best_val_x = val
                        best_x = x
                
                # Refine around best_x
                if best_val_x > current_val + 1e-12:
                    # Local refinement
                    span = (hi - lo) / n_samples
                    for _ in range(10):
                        xs_fine = np.linspace(max(lo, best_x - span), min(hi, best_x + span), 20)
                        improved_local = False
                        for x in xs_fine:
                            bary[i, c] = x
                            val = -hard_min_obj(bary.flatten())
                            if val > best_val_x:
                                best_val_x = val
                                best_x = x
                                improved_local = True
                        if not improved_local:
                            break
                    
                    bary[i, c] = best_x
                    bary = clamp_bary(bary)
                    current_val = -hard_min_obj(bary.flatten())
                    improved = True
                else:
                    bary[i, c] = cur
        
        if not improved:
            break
    
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

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    Uses energy-based, soft-min, epigraph, and hard-min formulations.
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    
    def try_update(bary_new):
        nonlocal bary_current, current_val
        bary_new = clamp_bary(bary_new)
        new_val = -hard_min_obj(bary_new.flatten())
        if new_val > current_val:
            bary_current = bary_new
            current_val = new_val
            return True
        return False
    
    # Stage 0: Energy-based refinement (smooth, helps escape bad local minima)
    # Use lower p for broader exploration, higher p for focused refinement
    if time.time() - start_time < time_limit * 0.30:
        for p in [1.0, 1.5, 2.0, 3.0]:
            if time.time() - start_time > time_limit * 0.30:
                break
            try:
                result = minimize(
                    energy_obj,
                    bary_current.flatten(),
                    args=(p,),
                    method='L-BFGS-B',
                    bounds=bounds_bary,
                    options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                try_update(result.x.reshape(N, 2))
            except Exception:
                pass
    
    # Stage 1: Graduated soft-min with increasing beta
    betas = [30.0, 80.0, 200.0]
    for beta in betas:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds_bary,
                options={'maxiter': 120, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            try_update(result.x.reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.65:
        return bary_current
    
    # Stage 2: Epigraph formulation
    if time.time() - start_time < time_limit * 0.8:
        try:
            areas = calc_areas_bary(bary_current)
            t_init = np.min(areas) * 0.95
            params_init = np.concatenate([bary_current.flatten(), [t_init]])
            
            bounds_epigraph = bounds_bary + [(0.0, 1.0)]
            
            result_ep = minimize(
                epigraph_obj,
                params_init,
                method='L-BFGS-B',
                bounds=bounds_epigraph,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            try_update(result_ep.x[:N*2].reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.85:
        return bary_current
    
    # Stage 3: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        try_update(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.90:
        return bary_current
    
    # Stage 4: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds_bary,
            options={'maxiter': 600, 'ftol': 1e-14}
        )
        try_update(result3.x.reshape(N, 2))
    except Exception:
        pass
    
    # Stage 5: Coordinate descent for escaping local minima
    if time.time() - start_time < time_limit * 0.98:
        try:
            cd_bary = coordinate_descent(bary_current, time_limit=min(2.0, time_limit - (time.time() - start_time)))
            try_update(cd_bary)
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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=6.0)
    update_best(refined_hex)
    
    # 2. Global Search: multi-start with diverse structured initializations
    # (faster than DE, more diverse coverage)
    if time.time() < deadline - 25:
        try:
            # Generate diverse starting points and refine the best few
            candidates = []
            for seed in range(10):
                if time.time() > deadline - 20:
                    break
                if seed % 3 == 0:
                    init = generate_random_init(seed=1000 + seed)
                elif seed % 3 == 1:
                    init = generate_hexagonal_init()
                else:
                    # Structured: grid-like with jitter
                    rng = np.random.default_rng(2000 + seed)
                    u = rng.random(N)
                    v = rng.random(N) * (1 - u)
                    init = np.column_stack([u, v])
                    # Add some structure: sort and spread
                    idx = np.argsort(u)
                    init = init[idx]
                
                candidates.append(init)
            
            # Refine top candidates
            for cand in candidates[:5]:
                if time.time() > deadline - 15:
                    break
                refined = refine_local(cand, time_limit=4.0)
                update_best(refined)
        except Exception:
            pass
    
    # 3. Multi-start: diverse initializations with local refinement
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        remaining = deadline - 10 - time.time()
        if remaining < 2.5:
            break
        
        # Cycle through different initialization strategies
        if restart_count % 3 == 1:
            # Random restart
            init = generate_random_init(seed=5000 + restart_count)
        elif restart_count % 3 == 2:
            # Perturbation of best solution
            if best_bary is not None:
                init = generate_perturbed_init(best_bary, sigma=0.03, seed=6000 + restart_count)
            else:
                init = generate_random_init(seed=5000 + restart_count)
        else:
            # Hexagonal with different scale
            if best_bary is not None:
                init = generate_perturbed_init(best_bary, sigma=0.06, seed=7000 + restart_count)
            else:
                init = generate_hexagonal_init()
        
        t_limit = min(4.0, remaining - 0.5)
        if t_limit < 1.0:
            break
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 4:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 1.5:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # Final coordinate descent pass
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                cd_final = coordinate_descent(best_bary, time_limit=max(0.5, deadline - 0.5 - time.time()))
                update_best(cd_final)
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
