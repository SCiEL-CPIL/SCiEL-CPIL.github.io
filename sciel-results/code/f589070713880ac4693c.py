import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 140.0

# Pre-compute triangle indices for the full 11-point set
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_EQUI = 0.5 * H

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

def expand_symmetric_orbits(params):
    """
    Expand the reduced parameter vector into the full 11-point barycentric configuration.
    Structure:
    - 1 point at centroid (1/3, 1/3)
    - 3 orbits of size 3 (defined by u1,v1; u2,v2; u3,v3)
    - 1 point on axis u=v (defined by u4)
    
    params: [u1, v1, u2, v2, u3, v3, u4]
    """
    u1, v1, u2, v2, u3, v3, u4 = params
    
    # Orbit 1: (u1, v1), (v1, u1), (1-u1-v1, 1-v1-u1) -- wait, rotation is not just swap.
    # Rotation by 120 degrees in barycentric coords (u, v, w):
    # (u, v, w) -> (w, u, v) -> (v, w, u)
    # Let's define the 3 points for an orbit given (u, v):
    # p_a = (u, v)
    # p_b = (1-v-u, u)  -> (w, u)
    # p_c = (v, 1-u-v)  -> (v, w)
    
    # Orbit 1
    o1_1 = np.array([u1, v1])
    o1_2 = np.array([1.0 - v1 - u1, u1])
    o1_3 = np.array([v1, 1.0 - u1 - v1])
    
    # Orbit 2
    o2_1 = np.array([u2, v2])
    o2_2 = np.array([1.0 - v2 - u2, u2])
    o2_3 = np.array([v2, 1.0 - u2 - v2])
    
    # Orbit 3
    o3_1 = np.array([u3, v3])
    o3_2 = np.array([1.0 - v3 - u3, u3])
    o3_3 = np.array([v3, 1.0 - u3 - v3])
    
    # Point on axis u=v
    p_axis = np.array([u4, u4])
    
    # Centroid
    p_cent = np.array([1.0/3.0, 1.0/3.0])
    
    pts = np.array([
        p_cent,
        o1_1, o1_2, o1_3,
        o2_1, o2_2, o2_3,
        o3_1, o3_2, o3_3,
        p_axis
    ])
    
    return pts

def symmetric_obj(params):
    """
    Objective function for the symmetric parameter space.
    """
    pts = expand_symmetric_orbits(params)
    # Check validity: points must be inside triangle
    # Barycentric coords must be >= 0 and sum <= 1
    # Since we generate them from u,v, we need to clamp or penalize.
    # It's better to clamp to ensure validity, then calculate area.
    pts_clamped = clamp_bary(pts, eps=1e-9)
    
    # Check for coincident points (degenerate triangles)
    # If any two points are too close, area is 0, which is bad.
    # The min area will handle this naturally as it will be 0.
    
    areas = calc_areas_bary(pts_clamped)
    
    # Penalty if points are outside or too close
    # The clamp ensures they are inside.
    # We just return -min_area
    return -np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_params = None
    best_bary = None
    
    rng = np.random.default_rng(42)
    
    # Define bounds for the symmetric parameters
    # u, v must be such that u>=0, v>=0, u+v<=1
    # So u in [0, 1], v in [0, 1-u]
    # For simplicity, we use box bounds [0,1] for all and rely on clamp/penalty
    # But L-BFGS-B works better with tight bounds.
    # Let's use [0.01, 0.99] for u and v to stay away from edges initially.
    
    # Initial guess:
    # Centroid is fixed.
    # Orbit 1: small triangle around centroid. u=0.4, v=0.4? No, u+v<=1.
    # Let's try:
    # o1: (0.3, 0.3) -> (0.4, 0.3) -> (0.3, 0.4)
    # o2: (0.1, 0.1)
    # o3: (0.6, 0.2)
    # axis: u=0.5 (midpoint of side? no, (0.5, 0.5) is outside. u+v=1.0. On edge.)
    # Let's pick u4=0.2
    
    x0 = np.array([0.3, 0.3, 0.1, 0.1, 0.6, 0.2, 0.2])
    
    # Bounds
    bounds = [(0.01, 0.99)] * 7
    
    # Optimization
    try:
        res = minimize(
            symmetric_obj,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() < deadline - 10.0:
            best_params = res.x
            best_bary = expand_symmetric_orbits(best_params)
            best_bary = clamp_bary(best_bary)
            best_val = -symmetric_obj(best_params)
            
            # Try a few random starts in the symmetric space
            for i in range(10):
                if time.time() > deadline - 5.0:
                    break
                x_rand = rng.uniform(0.05, 0.95, size=7)
                # Ensure u+v < 1 for the orbits?
                # The expand function creates points that might be outside.
                # The clamp will fix it, but it might distort the symmetry.
                # It's better to constrain the random init.
                # For random init, we just try it.
                
                res_r = minimize(
                    symmetric_obj,
                    x_rand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                
                val_r = -res_r.fun
                if val_r > best_val:
                    best_val = val_r
                    best_params = res_r.x
                    best_bary = expand_symmetric_orbits(best_params)
                    best_bary = clamp_bary(best_bary)
                    
    except Exception as e:
        print(f"Symmetric optimization failed: {e}")
        
    # Fallback to standard optimization if symmetric fails or is not good enough
    if best_bary is None:
        # Use the old hexagonal init
        pts = []
        for i in range(4):
            for j in range(4 - i):
                pts.append([i/3.0, j/3.0])
        pts.append([1/3.0, 1/3.0])
        best_bary = clamp_bary(np.array(pts))
        
        # Refine it
        res = minimize(
            hard_min_obj,
            best_bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        best_bary = clamp_bary(res.x.reshape(N, 2))
        best_val = -hard_min_obj(best_bary.flatten())
    else:
        # Even with symmetric init, do a final full-space polish to break symmetry if beneficial
        # or just polish the symmetric solution in full space
        if time.time() < deadline - 2.0:
            res = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            polished_bary = clamp_bary(res.x.reshape(N, 2))
            polished_val = -hard_min_obj(polished_bary.flatten())
            if polished_val > best_val:
                best_val = polished_val
                best_bary = polished_bary

    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
