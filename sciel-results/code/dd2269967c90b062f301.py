"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Strategy: Multi-start optimization with soft-min objective, Delaunay-based refinement,
SLSQP polish, and coordinate descent. Uses barycentric coordinates for constraint handling.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute triangle indices
tri_idx = np.array(list(combinations(range(N), 3)))  # (165, 3)
_I = tri_idx[:, 0]
_J = tri_idx[:, 1]
_K = tri_idx[:, 2]

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_I]; vi = v[_I]; wi = w[_I]
    uj = u[_J]; vj = v[_J]; wj = w[_J]
    uk = u[_K]; vk = v[_K]; wk = w[_K]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.maximum(sums[invalid], 1e-15)
        u[invalid] *= scale
        v[invalid] *= scale
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective: negative of minimum triangle area."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """Smooth approximation of negative min area using log-sum-exp."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def _delaunay_smooth_obj(bary_flat, beta=50.0):
    """
    Objective that focuses on Delaunay triangles (which are likely to have small areas).
    Uses soft-min over Delaunay triangles plus a small penalty for non-Delaunay.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    try:
        tri = Delaunay(cart)
        delaunay_triangles = tri.simplices  # (M, 3)
    except Exception:
        # Fallback to all triangles
        delaunay_triangles = tri_idx
    
    areas_all = calc_areas_bary(bary)
    
    # Get areas for Delaunay triangles
    del_areas = np.array([areas_all[tri_idx == t].item() if False else 
                          np.abs(np.linalg.det(np.column_stack([
                              cart[t[0]] - cart[t[2]],
                              cart[t[1]] - cart[t[2]]
                          ]))) / (2 * 0.5 * H)  # normalize by big triangle area
                          for t in delaunay_triangles])
    
    # Actually, let's just use the barycentric areas for Delaunay triangles
    del_areas_bary = np.array([
        np.abs(calc_areas_bary(bary)[
            (tri_idx[:, 0] == t[0]) & (tri_idx[:, 1] == t[1]) & (tri_idx[:, 2] == t[2])
        ]) if len(calc_areas_bary(bary)[
            (tri_idx[:, 0] == t[0]) & (tri_idx[:, 1] == t[1]) & (tri_idx[:, 2] == t[2])
        ]) > 0 else 0.0
        for t in delaunay_triangles
    ])
    
    # Simpler: just use all areas but weight Delaunay ones more
    # For efficiency, just use soft-min over all areas
    scaled = -beta * areas_all
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas_all)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_hexagonal_init():
    """Generate hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.45, 30):
        points = []
        for i in range(8):
            for j in range(8):
                u = i * s
                v = j * s
                if u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
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

def generate_structured_init(seed=None):
    """Generate structured initial configuration based on symmetry."""
    rng = np.random.default_rng(seed)
    bary = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [1/3, 1/3],
    ])
    
    for _ in range(N - len(bary)):
        u = rng.uniform(0.1, 0.8)
        v = rng.uniform(0.1, 0.8 - u)
        bary = np.vstack([bary, [[u, v]]])
    
    return clamp_bary(bary)

def try_optimize(bary_init, beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400, deadline=None):
    """Run multi-stage optimization from an initial configuration."""
    if deadline is not None and time.time() > deadline - 10:
        return bary_init
    
    try:
        bounds = [(0, 1)] * (N * 2)
        
        # Stage 1: Exploration with smooth landscape
        result1 = minimize(
            soft_min_obj,
            bary_init.flatten(),
            args=(beta1,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if deadline is not None and time.time() > deadline - 5:
            return result1.x.reshape(N, 2)

        # Stage 2: Exploitation with sharper landscape
        result2 = minimize(
            soft_min_obj,
            result1.x,
            args=(beta2,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        return result2.x.reshape(N, 2)
        
    except Exception:
        return bary_init

def slsqp_polish(bary_init, deadline=None):
    """Polish using SLSQP which handles the box constraints more naturally."""
    if deadline is not None and time.time() > deadline - 3.0:
        return bary_init
    
    try:
        bounds = [(0, 1)] * (N * 2)
        res = minimize(
            soft_min_obj,
            bary_init.flatten(),
            args=(300.0,),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return bary_init

def coordinate_descent_polish(bary_init, deadline=None, n_sweeps=30):
    """Polish using coordinate descent."""
    if deadline is not None and time.time() > deadline - 2.0:
        return bary_init
    
    bary = bary_init.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    for sweep in range(n_sweeps):
        if deadline is not None and time.time() > deadline - 1.0:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 1.0:
                break
            
            orig_u, orig_v = bary[i, 0], bary[i, 1]
            best_local_u, best_local_v = orig_u, orig_v
            best_local_val = -hard_min_obj(bary.flatten())
            
            # Multi-resolution search
            for scale in [0.05, 0.02, 0.01, 0.005]:
                for du in np.linspace(-scale, scale, 5):
                    for dv in np.linspace(-scale, scale, 5):
                        new_u = orig_u + du
                        new_v = orig_v + dv
                        if new_u < 1e-9 or new_v < 1e-9 or new_u + new_v > 1 - 1e-9:
                            continue
                        bary[i, 0] = new_u
                        bary[i, 1] = new_v
                        val = -hard_min_obj(bary.flatten())
                        if val > best_local_val:
                            best_local_val = val
                            best_local_u, best_local_v = new_u, new_v
                            orig_u, orig_v = new_u, new_v
            
            if best_local_val > best_val + 1e-10:
                bary[i, 0] = best_local_u
                bary[i, 1] = best_local_v
                best_val = best_local_val
                improved = True
            else:
                bary[i, 0] = orig_u
                bary[i, 1] = orig_v
        
        if not improved:
            break
    
    return bary

def simulated_annealing(bary_init, deadline=None, n_iter=1500):
    """Simulated annealing to escape local minima."""
    if deadline is not None and time.time() > deadline - 5.0:
        return bary_init
    
    try:
        bary = bary_init.copy()
        best_bary = bary.copy()
        best_val = -hard_min_obj(bary.flatten())
        
        t0 = 0.04
        cooling = 0.997
        rng = np.random.default_rng(12345)
        
        for i in range(n_iter):
            if deadline is not None and time.time() > deadline - 3.0:
                break
            
            noise = rng.normal(0, t0, bary.shape)
            new_bary = clamp_bary(bary + noise)
            
            val = -hard_min_obj(bary.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            
            delta = new_val - val
            if delta > 0 or rng.random() < np.exp(delta / max(t0, 1e-10)):
                bary = new_bary
                if new_val > best_val:
                    best_val = new_val
                    best_bary = bary.copy()
            
            t0 *= cooling
        
        return best_bary
        
    except Exception:
        return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
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

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    result = try_optimize(hex_init, beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400, deadline=deadline)
    update_best(result)
    
    # 2. Try structured initialization
    if time.time() < deadline - 40:
        struct_init = generate_structured_init(seed=0)
        result = try_optimize(struct_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300, deadline=deadline)
        update_best(result)
    
    # 3. Try multiple random initializations
    num_random_starts = 10
    for i in range(num_random_starts):
        if time.time() > deadline - 50:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        result = try_optimize(rand_init, beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200, deadline=deadline)
        update_best(result)
        
    # 4. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(6):
            if time.time() > deadline - 35:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.03, seed=200 + i)
            result = try_optimize(perturbed, beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300, deadline=deadline)
            update_best(result)
    
    # 5. SLSQP polish
    if best_bary is not None and time.time() < deadline - 20:
        slsqp_result = slsqp_polish(best_bary, deadline=deadline)
        update_best(slsqp_result)
    
    # 6. Simulated annealing to escape local minima, then re-optimize
    if best_bary is not None and time.time() < deadline - 15:
        sa_result = simulated_annealing(best_bary, deadline=deadline, n_iter=1500)
        update_best(sa_result)
        if best_val > -np.inf and time.time() < deadline - 12:
            try:
                bounds = [(0, 1)] * (N * 2)
                res_sa = minimize(
                    soft_min_obj,
                    best_bary.flatten(),
                    args=(200.0,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                update_best(res_sa.x.reshape(N, 2))
            except Exception:
                pass
    
    # 7. Final polish: high-precision L-BFGS-B with very high beta
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            res_polish = minimize(
                soft_min_obj,
                best_bary.flatten(),
                args=(500.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            update_best(res_polish.x.reshape(N, 2))
        except Exception:
            pass
        
        # SLSQP final polish
        if time.time() < deadline - 5:
            slsqp_final = slsqp_polish(best_bary, deadline=deadline)
            update_best(slsqp_final)
        
        # Coordinate descent polish
        if time.time() < deadline - 2:
            cd_result = coordinate_descent_polish(best_bary, deadline=deadline, n_sweeps=40)
            update_best(cd_result)
    
    # Fallback if something went wrong
    if best_bary is None:
        best_bary = generate_random_init(seed=0)
    
    # Final safety clamp and convert to Cartesian
    best_bary = clamp_bary(best_bary)
    result_cart = bary_to_cart(best_bary)
    
    # Ensure all points are strictly inside
    result_cart = np.clip(result_cart, 1e-9, None)
    result_cart[:, 1] = np.clip(result_cart[:, 1], 1e-9, H - 1e-9)
    x_min = result_cart[:, 1] / np.sqrt(3)
    x_max = 1.0 - result_cart[:, 1] / np.sqrt(3)
    result_cart[:, 0] = np.clip(result_cart[:, 0], x_min + 1e-9, x_max - 1e-9)
    
    return result_cart
