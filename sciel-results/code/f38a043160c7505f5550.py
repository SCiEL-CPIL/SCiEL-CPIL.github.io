import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
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
    
    # Area is proportional to the determinant of the barycentric coordinates
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-8):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Ensure u, v >= 0 and u+v <= 1
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-clip to be safe
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
    Returns negative soft-min (so minimizing = maximizing soft-min).
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Avoid log(0) or numerical issues
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 1.0  # Large positive value = bad
    
    # Use log-sum-exp for numerical stability
    # soft_min = -(1/beta) * log(sum(exp(-beta * areas)))
    # = min(areas) + (1/beta) * log(mean(exp(-beta*(areas - min(areas)))))
    shifted = -beta * (areas - min_area)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted))) / beta
    soft_min = min_area + log_sum
    
    # Return negative so that minimizing this function maximizes soft_min
    return -soft_min

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.flatten()
    bary = bary.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    This is a strong prior for Heilbronn triangles.
    """
    best_bary = None
    best_val = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Robust Hexagonal Generator
    candidates = []
    
    # Define lattice spacing
    for scale in [0.2, 0.25, 0.3, 0.35, 0.4]:
        pts = []
        # Iterate over "rows" parallel to the base (constant w)
        # w = 1 - u - v
        # Let w = k * scale
        max_k = int(1.0 / scale)
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            # u + v = 1 - w_val
            # Distribute points along this segment
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            bary = np.array(pts[:N])
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
                
    if best_bary is None:
        # Fallback to random
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

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using Basin Hopping.
    Basin Hopping combines global random steps with local minimization,
    which is effective for escaping local minima in max-min problems.
    """
    start_time = time.time()
    
    # Local minimizer function for basinhopping
    def local_min(x):
        # We use L-BFGS-B on the smooth objective
        res = minimize(
            soft_min_obj,
            x,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        # Clamp result to ensure validity
        bary = clamp_bary(res.x.reshape(N, 2))
        return {
            'x': bary.flatten(),
            'fun': -hard_min_obj(bary.flatten()), # Return hard min score for comparison
            'nfev': res.nfev
        }
    
    # Step function: random perturbation
    def step(x):
        rng = np.random.default_rng()
        # Gaussian step
        new_x = x + rng.normal(0, 0.1, x.shape)
        # Clamp to valid region
        bary = clamp_bary(new_x.reshape(N, 2))
        return bary.flatten()
    
    best_bary = bary_init.copy()
    best_val = -hard_min_obj(best_bary.flatten())
    
    try:
        # Run basin hopping
        # We limit the number of iterations based on time
        # Each iteration involves a step and a local minimization
        # Estimate ~0.1s per iteration? Let's be conservative.
        max_iters = max(10, int(time_limit / 0.2))
        
        res = basinhopping(
            local_min,
            bary_init.flatten(),
            stepfunc=step,
            niter=max_iters,
            T=1.0,
            stepsize=0.1,
            mininterval=2,
            maxinterval=5,
            disp=False
        )
        
        candidate = clamp_bary(res.xf.reshape(N, 2))
        val = -hard_min_obj(candidate.flatten())
        if val > best_val:
            best_bary = candidate
            best_val = val
            
    except Exception:
        pass
    
    # Final polish with Nelder-Mead on hard min
    if time.time() < start_time + time_limit - 0.5:
        try:
            result_nm = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
            )
            candidate = clamp_bary(result_nm.x.reshape(N, 2))
            val = -hard_min_obj(candidate.flatten())
            if val > best_val:
                best_bary = candidate
                best_val = val
        except Exception:
            pass
            
    return best_bary

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
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution (Quick)
    if time.time() < deadline - 40:
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
                seed=42,
                maxiter=20,
                popsize=10,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement (Basin Hopping)
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            sigma = max(0.02, 0.08 - 0.005 * restart_count)
            init = best_bary + rng.normal(0, sigma, best_bary.shape)
            init = clamp_bary(init)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        # Use shorter time limit for more restarts
        remaining = deadline - 15 - time.time()
        if remaining > 10:
            tl = min(8.0, remaining / 3)
        else:
            tl = min(4.0, remaining / 2)
            
        refined = refine_local(init, time_limit=tl)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 2:
        final_bary = refine_local(best_bary, time_limit=min(5.0, deadline - 2 - time.time()))
        update_best(final_bary)

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
