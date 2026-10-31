"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-6):
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

def _delaunay_smooth_obj(bary_flat, target=None, lam=1.0):
    """
    Smooth surrogate: minimize sum of squared deviations of Delaunay triangle areas
    from a target value (default: current min area). This creates a differentiable
    landscape that encourages uniform small-triangle areas.
    """
    from scipy.spatial import Delaunay
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    try:
        tri = Delaunay(cart)
        simplices = tri.simplices  # (K, 3)
        
        if len(simplices) == 0:
            return hard_min_obj(bary_flat)
        
        # Compute areas of Delaunay triangles
        areas = np.zeros(len(simplices))
        for i, (a, b, c) in enumerate(simplices):
            # Area = 0.5 * |cross(b-a, c-a)|
            ab = cart[b] - cart[a]
            ac = cart[c] - cart[a]
            areas[i] = 0.5 * abs(ab[0]*ac[1] - ab[1]*ac[0])
        
        # Target: use the minimum area as reference
        if target is None:
            target = np.min(areas)
        
        # Objective: sum of squared deviations from target + penalty for min area
        deviations = areas - target
        obj = lam * np.sum(deviations**2) - np.min(areas)
        
        return obj
    except Exception:
        return hard_min_obj(bary_flat)

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
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
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

    def try_delaunay_optimize(bary_init, maxiter=300, lam=1.0):
        """
        Optimize using Delaunay-based smooth objective.
        This creates a differentiable landscape that encourages
        uniform small-triangle areas.
        """
        if time.time() > deadline - 15:
            return
        
        try:
            def obj_dl(x):
                return _delaunay_smooth_obj(x, target=None, lam=lam)
            
            bounds = [(0, 1)] * (N * 2)
            
            result = minimize(
                obj_dl,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result.x.reshape(N, 2))
            
        except Exception:
            pass

    def final_polish(bary, maxiter_nm=5000, maxiter_lbfgs=500):
        """
        High-precision final polish:
        1. Nelder-Mead on hard min with very tight tolerances
        2. L-BFGS-B on soft-min with high beta for final refinement
        """
        if time.time() > deadline - 5:
            return
        
        try:
            # Step 1: Nelder-Mead polish
            nm_result = minimize(
                hard_min_obj,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter_nm, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            update_best(nm_result.x.reshape(N, 2))
            
            if time.time() > deadline - 3:
                return
            
            # Step 2: L-BFGS-B polish with very high beta
            def obj_polish(x):
                return soft_min_obj(x, beta=500.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            lbfgs_result = minimize(
                obj_polish,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter_lbfgs, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            update_best(lbfgs_result.x.reshape(N, 2))
            
        except Exception:
            pass

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400)
    
    # 2. Try multiple random initializations
    num_random_starts = 8
    for i in range(num_random_starts):
        if time.time() > deadline - 40:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        try_optimize(rand_init, method='L-BFGS-B', beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200)
        
    # 3. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(5):
            if time.time() > deadline - 25:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300)
    
    # 4. Delaunay-based optimization from best configuration
    if best_bary is not None and time.time() < deadline - 20:
        try_delaunay_optimize(best_bary, maxiter=400, lam=1.0)
        
        # Try again with different lambda
        if time.time() < deadline - 15:
            try_delaunay_optimize(best_bary, maxiter=300, lam=0.5)
    
    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 10:
        final_polish(best_bary, maxiter_nm=5000, maxiter_lbfgs=500)
    
    # 6. Convert best barycentric to Cartesian and return
    if best_bary is not None:
        cart_pts = bary_to_cart(best_bary)
        # Final safety clamp to ensure points are strictly inside
        cart_pts = np.array([
            np.array([0.01, 0.01]) if p[0] < 0 else p,
        ] if False else [
            np.clip(p, [0, 0], [1, H]) for p in cart_pts
        ])
        return cart_pts
    else:
        # Fallback: return a simple grid
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        return bary_to_cart(clamp_bary(np.column_stack([u, v])))

# EVOLVE-BLOCK-END
