"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
NUM_TRIS = len(tri_indices)

# Triangle vertices in Cartesian
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    Point = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, H), w = 1-u-v
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = x - y / (2 * H)
    u = 1 - x - y / (2 * H)
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points in barycentric coords.
    Returns array of shape (165,) containing the area ratios.
    """
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

def clamp_bary(bary, eps=1e-6):
    """Clamp barycentric coordinates to the valid triangle region."""
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
    """Objective: negative of minimum triangle area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """Smooth approximation of the min using log-sum-exp."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum / beta

def generate_structured_init():
    """
    Generate a structured initial configuration based on known good patterns.
    For 11 points, use rows within the triangle.
    """
    points = []
    # Bottom row: 4 points
    for i in range(4):
        x = (i + 0.5) / 4.0
        points.append([x, 0.0])
    
    # Middle row: 3 points
    mid_y = H / 2.0
    for i in range(3):
        x = 0.25 + i * 0.25
        points.append([x, mid_y])
    
    # Upper row: 2 points
    up_y = 2 * H / 3.0
    for i in range(2):
        x = 0.25 + i * 0.5
        points.append([x, up_y])
    
    # Top: 1 point
    points.append([0.5, H])
    
    cart = np.array(points)
    bary = cart_to_bary(cart)
    return clamp_bary(bary)

def generate_hexagonal_init():
    """Generate hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.5, 25):
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
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
    
    def try_l_bfgs_smooth(bary_init, beta=100.0, maxiter=500):
        """Optimize using smooth soft-min objective with L-BFGS-B."""
        if time.time() > deadline - 5:
            return
        
        try:
            def obj(x):
                return soft_min_obj(x, beta=beta)
            
            bounds = [(0, 1)] * (N * 2)
            
            result = minimize(
                obj,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    def try_l_bfgs_hard(bary_init, maxiter=500):
        """Polish using hard min objective with L-BFGS-B."""
        if time.time() > deadline - 5:
            return
        
        try:
            result = minimize(
                hard_min_obj,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
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
    
    # Phase 1: Structured init with smooth optimization
    structured_init = generate_structured_init()
    try_l_bfgs_smooth(structured_init, beta=100.0, maxiter=500)
    
    # Phase 2: Hexagonal init with smooth optimization
    hex_init = generate_hexagonal_init()
    try_l_bfgs_smooth(hex_init, beta=100.0, maxiter=500)
    
    # Phase 3: Multiple random starts with smooth optimization
    for i in range(8):
        if time.time() > deadline - 40:
            break
        rand_init = generate_random_init(seed=100 + i)
        try_l_bfgs_smooth(rand_init, beta=80.0, maxiter=400)
    
    # Phase 4: Polish best with hard min using L-BFGS-B
    if best_bary is not None:
        if time.time() < deadline - 30:
            try_l_bfgs_hard(best_bary, maxiter=500)
        
        # Phase 5: Polish with Nelder-Mead
        if time.time() < deadline - 20:
            try_nm_polish(best_bary, maxiter=2000)
    
    # Phase 6: Perturb and re-optimize
    if best_bary is not None:
        for i in range(6):
            if time.time() > deadline - 15:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            if time.time() < deadline - 10:
                try_l_bfgs_smooth(perturbed, beta=150.0, maxiter=300)
            if time.time() < deadline - 5:
                try_l_bfgs_hard(best_bary, maxiter=300)
    
    # Phase 7: Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
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
