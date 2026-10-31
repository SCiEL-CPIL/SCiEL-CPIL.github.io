"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, differential_evolution
import warnings

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
_I = tri_idx[:, 0]
_J = tri_idx[:, 1]
_K = tri_idx[:, 2]

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_I]
    vi = v[_I]
    wi = w[_I]
    
    uj = u[_J]
    vj = v[_J]
    wj = w[_J]
    
    uk = u[_K]
    vk = v[_K]
    wk = w[_K]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
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

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def _simulated_annealing_bary(start_bary, time_limit=10.0, seed=None):
    """Simulated annealing in barycentric coordinates."""
    rng = np.random.default_rng(seed)
    bary = clamp_bary(np.array(start_bary, dtype=float))
    current_score = -hard_min_obj(bary.flatten())
    best_bary = bary.copy()
    best_score = current_score

    T = 0.05
    T_min = 1e-7
    alpha = 0.9995
    step_size = 0.08

    start_time = time.time()
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
        idx = rng.integers(N)
        noise = rng.normal(0, step_size, 2)
        new_bary = bary.copy()
        new_bary[idx] += noise
        new_bary = clamp_bary(new_bary)
        old_row = bary[idx].copy()
        bary = new_bary
        new_score = -hard_min_obj(bary.flatten())
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_bary = bary.copy()
        else:
            bary = bary.copy()
            bary[idx] = old_row
        T *= alpha
        step_size *= 0.9999
    return best_bary, best_score

def _coord_descent_bary(bary, time_limit=5.0):
    """Coordinate descent polish in barycentric coordinates."""
    bary = clamp_bary(np.array(bary, dtype=float)).copy()
    best = bary.copy()
    best_score = -hard_min_obj(bary.flatten())
    start_time = time.time()
    step = 0.01
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            base = bary[i].copy()
            cur = best_score
            for du in (step, -step):
                for dv in (step, -step):
                    cand = bary.copy()
                    cand[i] = base + np.array([du, dv])
                    cand = clamp_bary(cand)
                    s = -hard_min_obj(cand.flatten())
                    if s > cur + 1e-12:
                        cur = s
                        bary = cand
                        improved = True
                        if cur > best_score:
                            best_score = cur
                            best = bary.copy()
        step *= 0.7
    return best, best_score

def _slsqp_polish(bary, time_limit=5.0):
    """SLSQP polish with explicit constraints."""
    bary = clamp_bary(np.array(bary, dtype=float)).copy()
    
    def obj(x):
        return hard_min_obj(x)
    
    # Constraints: each point must be inside the triangle
    # In barycentric: u >= 0, v >= 0, u+v <= 1
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*i+1]})
    
    bounds = [(0, 1)] * (N * 2)
    
    start_time = time.time()
    try:
        result = minimize(
            obj,
            bary.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if result.success or time.time() - start_time < time_limit:
            return clamp_bary(result.x.reshape(N, 2))
    except Exception:
        pass
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # 12s safety margin before 150s kill

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

    def try_optimize(bary_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        """
        Two-stage optimization:
        1. Smooth landscape (low beta) to find a good basin.
        2. Sharp landscape (high beta) to refine the minimum.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1: Exploration
            def obj1(x):
                return soft_min_obj(x, beta=beta1)
            
            bounds = [(0, 1)] * (N * 2)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Check time before stage 2
            if time.time() > deadline - 5:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Exploitation
            def obj2(x):
                return soft_min_obj(x, beta=beta2)
            
            result2 = minimize(
                obj2,
                result1.x,
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result2.x.reshape(N, 2))
                
        except Exception:
            pass

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 2. Try differential evolution for global exploration
    if time.time() < deadline - 40:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                de_result = differential_evolution(
                    hard_min_obj,
                    bounds=[(0, 1)] * (N * 2),
                    seed=42,
                    maxiter=30,
                    popsize=12,
                    tol=1e-6,
                    polish=False,
                    update_interval=10
                )
            if de_result.success or True:
                de_bary = clamp_bary(de_result.x.reshape(N, 2))
                refined_de = refine_local(de_bary, time_limit=8.0)
                update_best(refined_de)
        except Exception:
            pass
    
    # 3. Try multiple random initializations
    num_random_starts = 8
    for i in range(num_random_starts):
        if time.time() > deadline - 30:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        refined_rand = refine_local(rand_init, time_limit=5.0)
        update_best(refined_rand)
        
    # 4. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(5):
            if time.time() > deadline - 15:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            refined_pert = refine_local(perturbed, time_limit=5.0)
            update_best(refined_pert)
    
    # 5. Simulated annealing to escape local minima
    if best_bary is not None and time.time() < deadline - 20:
        remaining = deadline - time.time() - 20
        if remaining > 5.0:
            try:
                sa_bary, sa_score = _simulated_annealing_bary(best_bary, time_limit=remaining, seed=999)
                update_best(sa_bary)
            except Exception:
                pass
    
    # 6. SLSQP polish
    if best_bary is not None and time.time() < deadline - 10:
        try:
            slsqp_bary = _slsqp_polish(best_bary, time_limit=5.0)
            update_best(slsqp_bary)
        except Exception:
            pass
    
    # 7. Final polish with Nelder-Mead on hard min
    if best_bary is not None and time.time() < deadline - 8:
        try:
            nm_result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            update_best(nm_result.x.reshape(N, 2))
        except Exception:
            pass
    
    # 8. Coordinate descent polish
    if best_bary is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        if remaining > 2.0:
            try:
                cd_bary, cd_score = _coord_descent_bary(best_bary, time_limit=remaining)
                update_best(cd_bary)
            except Exception:
                pass
    
    # 9. Convert best barycentric to Cartesian
    if best_bary is not None:
        result = bary_to_cart(best_bary)
    else:
        # Fallback
        result = bary_to_cart(generate_random_init(seed=0))
    
    return result
