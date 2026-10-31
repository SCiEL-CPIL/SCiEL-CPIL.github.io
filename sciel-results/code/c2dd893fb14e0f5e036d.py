"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
NUM_TRIS = len(tri_indices)

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

def cart_to_bary(cart):
    """
    Convert Cartesian (x, y) to barycentric coords (u, v).
    w = 1 - u - v
    x = v + 0.5*w => x = v + 0.5*(1-u-v) = 0.5 + v - 0.5*u - 0.5*v = 0.5 + 0.5*v - 0.5*u
    y = w*H => w = y/H => u+v = 1 - y/H
    From x: 0.5*v - 0.5*u = x - 0.5 => v - u = 2*x - 1
    So: u + v = 1 - y/H, v - u = 2*x - 1
    => 2v = 1 - y/H + 2*x - 1 = 2*x - y/H => v = x - y/(2*H)
    => u = 1 - y/H - v = 1 - y/H - x + y/(2*H) = 1 - x - y/(2*H)
    """
    x = cart[:, 0]
    y = cart[:, 1]
    v = x - y / (2 * H)
    u = 1 - x - y / (2 * H)
    return np.column_stack([u, v])

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

def calc_areas_and_grads(bary):
    """
    Calculate areas and their gradients with respect to barycentric coordinates.
    Returns (areas, grads) where areas is (165,) and grads is (165, N, 2).
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
    
    # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    areas = np.abs(det)
    
    # For gradient, we need d(det)/d(u_i) etc.
    # Let's compute gradients numerically for stability
    grads = np.zeros((NUM_TRIS, N, 2))
    eps = 1e-7
    for i in range(N):
        for dim in range(2):
            bary_plus = bary.copy()
            bary_plus[i, dim] += eps
            bary_minus = bary.copy()
            bary_minus[i, dim] -= eps
            
            areas_plus = calc_areas_bary(bary_plus)
            areas_minus = calc_areas_bary(bary_minus)
            
            # Use central difference for abs: d|f|/dx = sign(f) * df/dx
            # But we can just use (|f+| - |f-|) / (2*eps)
            grads[:, i, dim] = (areas_plus - areas_minus) / (2 * eps)
    
    return areas, grads

def calc_pairwise_distances_cart(cart):
    """Calculate all pairwise Euclidean distances in Cartesian space."""
    # cart: (N, 2)
    # dist[i,j] = ||cart[i] - cart[j]||
    diff = cart[:, np.newaxis, :] - cart[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist

def repulsion_penalty(cart, gamma=0.1):
    """
    Convex quadratic penalty for sum of inverse squared distances.
    This encourages points to be well-separated.
    Returns the penalty value (positive, to be minimized).
    """
    dist = calc_pairwise_distances_cart(cart)
    # Set diagonal to 1 to avoid division by zero
    np.fill_diagonal(dist, 1.0)
    # Sum of 1/d^2 for all pairs (upper triangle only to avoid double counting)
    iu = np.triu_indices(N, k=1)
    inv_sq = 1.0 / (dist[iu] ** 2 + 1e-10)
    return gamma * np.sum(inv_sq)

def smooth_obj_with_repulsion(bary_flat, beta=50.0, gamma=0.1):
    """
    Combined objective: soft-min of areas + repulsion penalty.
    This creates a smoother landscape that guides toward well-distributed configs.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    areas = calc_areas_bary(bary)
    rep = repulsion_penalty(cart, gamma=gamma)
    
    # Soft min of areas
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        soft_min = -np.min(areas)
    else:
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
    
    # We want to maximize soft_min and minimize repulsion
    # So minimize: -soft_min + rep
    return -soft_min + rep

def repulsion_penalty(cart, gamma=0.1):
    """
    Convex quadratic penalty for sum of inverse squared distances.
    This encourages points to be well-separated.
    Returns the penalty value (positive, to be minimized).
    """
    dist = calc_pairwise_distances_cart(cart)
    # Set diagonal to 1 to avoid division by zero
    np.fill_diagonal(dist, 1.0)
    # Sum of 1/d^2 for all pairs (upper triangle only to avoid double counting)
    iu = np.triu_indices(N, k=1)
    inv_sq = 1.0 / (dist[iu] ** 2 + 1e-10)
    return gamma * np.sum(inv_sq)

def calc_pairwise_distances_cart(cart):
    """Calculate all pairwise Euclidean distances in Cartesian space."""
    # cart: (N, 2)
    # dist[i,j] = ||cart[i] - cart[j]||
    diff = cart[:, np.newaxis, :] - cart[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

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

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.5, 30):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s
                if u >= 0 and v >= 0 and u + v <= 1 and len(points) < N:
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

def generate_known_good_init():
    """
    Generate an initial configuration based on known good patterns for Heilbronn problem.
    For n=11, use a structured arrangement.
    """
    # Use a grid-like pattern that's known to work well
    # Arrange points in rows within the triangle
    points = []
    # Bottom row (y=0): 4 points
    for i in range(4):
        x = (i + 0.5) / 4.0
        points.append([x, 0.0])
    
    # Middle row (y=H/2): 3 points  
    mid_y = H / 2.0
    for i in range(3):
        x = 0.25 + i * 0.25
        points.append([x, mid_y])
    
    # Upper row (y=H*2/3): 2 points
    up_y = 2 * H / 3.0
    for i in range(2):
        x = 0.25 + i * 0.5
        points.append([x, up_y])
    
    # Top: 1 point
    points.append([0.5, H])
    
    # Convert to barycentric
    cart = np.array(points)
    bary = cart_to_bary(cart)
    return clamp_bary(bary)

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
    
    def try_epigraph_optimize(bary_init, t_init=None, beta=200.0, gamma=0.05, maxiter=500):
        """Optimize using epigraph formulation."""
        if time.time() > deadline - 5:
            return
        
        try:
            if t_init is None:
                areas = calc_areas_bary(bary_init)
                t_init = max(0.001, np.min(areas) * 0.5)
            
            x_init = np.concatenate([bary_init.flatten(), [t_init]])
            
            def obj(x):
                return epigraph_obj_smooth(x, beta=beta, gamma=gamma)
            
            # Bounds for barycentric coords and t
            bounds = [(0, 1)] * (N * 2) + [(0, 0.15)]
            
            result = minimize(
                obj,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            final_bary = clamp_bary(result.x[:N*2].reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    def try_smooth_optimize(bary_init, beta=50.0, gamma=0.1, maxiter=300):
        """Optimize using smooth objective with repulsion."""
        if time.time() > deadline - 5:
            return
        
        try:
            def obj(x):
                return smooth_obj_with_repulsion(x, beta=beta, gamma=gamma)
            
            bounds = [(0, 1)] * (N * 2)
            
            result = minimize(
                obj,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    def try_hard_polish(bary_init, maxiter=500):
        """Polish using hard min objective."""
        if time.time() > deadline - 5:
            return
        
        try:
            result = minimize(
                hard_min_obj,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    def try_nm_polish(bary_init, maxiter=2000):
        """Polish using Nelder-Mead on hard min."""
        if time.time() > deadline - 5:
            return
        
        try:
            bary_nm = bary_init.copy()
            bary_nm[:, 0] = np.clip(bary_nm[:, 0], 1e-6, 1 - 1e-6)
            bary_nm[:, 1] = np.clip(bary_nm[:, 1], 1e-6, 1 - bary_nm[:, 0] - 1e-6)
            
            result = minimize(
                hard_min_obj,
                bary_nm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-16}
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    # Phase 1: Global search with differential evolution on smooth objective
    if time.time() < deadline - 50:
        try:
            def de_smooth_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                cart = bary_to_cart(bary)
                areas = calc_areas_bary(bary)
                rep = repulsion_penalty(cart, gamma=0.05)
                return -np.min(areas) + rep
            
            bounds = [(0, 1)] * (N * 2)
            result = differential_evolution(
                de_smooth_obj,
                bounds,
                seed=42,
                maxiter=25,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    # Phase 2: Local optimization from known good init with epigraph
    known_init = generate_known_good_init()
    try_epigraph_optimize(known_init, beta=200.0, gamma=0.05, maxiter=400)
    
    # Phase 3: Hexagonal init with epigraph
    hex_init = generate_hexagonal_init()
    try_epigraph_optimize(hex_init, beta=200.0, gamma=0.05, maxiter=400)
    
    # Phase 4: Multiple random starts with epigraph
    for i in range(5):
        if time.time() > deadline - 35:
            break
        rand_init = generate_random_init(seed=100 + i)
        try_epigraph_optimize(rand_init, beta=150.0, gamma=0.05, maxiter=300)
    
    # Phase 5: Polish best with hard min using L-BFGS-B
    if best_bary is not None:
        if time.time() < deadline - 25:
            try_hard_polish(best_bary, maxiter=400)
        
        # Phase 6: Polish with Nelder-Mead
        if time.time() < deadline - 15:
            try_nm_polish(best_bary, maxiter=1500)
    
    # Phase 7: Perturb and re-optimize with epigraph
    if best_bary is not None:
        for i in range(4):
            if time.time() > deadline - 12:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            if time.time() < deadline - 8:
                try_epigraph_optimize(perturbed, beta=300.0, gamma=0.03, maxiter=250)
            if time.time() < deadline - 5:
                try_nm_polish(best_bary, maxiter=500)
    
    # Phase 8: Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        # Use very tight tolerances
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    # If all failed, generate a random valid set
    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    # Convert to Cartesian
    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    # Final safety check: ensure all points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
            
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
             cart[i, 1] = 1e-9
             
    return cart
