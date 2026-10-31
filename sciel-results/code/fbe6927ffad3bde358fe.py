"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
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
    # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    # Area is proportional to |det|. 
    # The area of the reference triangle is 1 (in barycentric units, the determinant of vertices is 1).
    # Actually, for vertices (1,0,0), (0,1,0), (0,0,1), det is 1.
    # So the ratio of areas is exactly |det|.
    return np.abs(det)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    
    # Project to valid barycentric region
    # If u < 0, v < 0, or u+v > 1, project back.
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Simple projection: clip and renormalize? 
    # Better: If invalid, return a large penalty.
    # But DE can handle bounds. We will use bounds in DE.
    # For SLSQP, we can use bounds too.
    
    # Check validity
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(u + v > 1 + 1e-9):
        return 1e6
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700: # exp overflow guard
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(u + v > 1 + 1e-9):
        return 1e6
        
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Bounds for DE: u in [0,1], v in [0,1]
    # Note: DE will explore the square [0,1]x[0,1]. Points with u+v>1 are invalid.
    # We penalize them in the objective.
    bounds = [(0, 1)] * (N * 2)
    
    best_bary = None
    best_val = -np.inf
    
    # 1. Differential Evolution Global Search
    # Use a smooth objective with moderate beta
    def de_obj(x):
        return soft_min_obj(x, beta=50.0)
        
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                de_obj,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            best_bary = de_result.x
            best_val = -de_result.fun
        except Exception:
            pass
    
    # 2. Local Polish with SLSQP
    # Use a sharper soft-min
    def slsqp_obj(x):
        return soft_min_obj(x, beta=200.0)
        
    if best_bary is not None and time.time() < deadline - 30:
        try:
            # SLSQP with bounds
            # Note: SLSQP might struggle with the non-smooth boundary u+v=1.
            # But soft-min is smooth everywhere.
            result = minimize(
                slsqp_obj,
                best_bary,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if -result.fun > best_val:
                best_val = -result.fun
                best_bary = result.x
        except Exception:
            pass
            
    # 3. Final Polish with Hard Min and Nelder-Mead
    # Nelder-Mead is derivative-free and can handle the non-differentiable min
    # It's good for final fine-tuning.
    if best_bary is not None and time.time() < deadline - 10:
        try:
            # We need to ensure best_bary is strictly inside to avoid boundary issues for NM
            # Clip slightly
            bary_nm = best_bary.reshape(N, 2).copy()
            bary_nm[:, 0] = np.clip(bary_nm[:, 0], 1e-6, 1 - 1e-6)
            bary_nm[:, 1] = np.clip(bary_nm[:, 1], 1e-6, 1 - bary_nm[:, 0] - 1e-6)
            
            result = minimize(
                hard_min_obj,
                bary_nm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            if -result.fun > best_val:
                best_val = -result.fun
                best_bary = result.x
        except Exception:
            pass

    # If all failed, generate a random valid set
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N)
        # Ensure u+v <= 1
        # Simple rejection or scaling
        # Scale v to fit
        v = v * (1 - u)
        best_bary = np.column_stack([u, v]).flatten()

    # Convert to Cartesian
    bary_final = best_bary.reshape(N, 2)
    # Final projection to ensure validity
    u = bary_final[:, 0]
    v = bary_final[:, 1]
    # Clip to valid region
    invalid = (u < 0) | (v < 0) | (u + v > 1)
    if np.any(invalid):
        # Project invalid points to centroid or boundary
        # Simple projection: if u+v>1, scale u,v by 1/(u+v)
        sums = u + v
        scale = np.where(sums > 1, 1.0/np.where(sums>0, sums, 1), 1.0)
        u = u * scale
        v = v * scale
        # Clip negatives
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        # Re-normalize if still bad (shouldn't happen)
        bary_final[:, 0] = u
        bary_final[:, 1] = v
        
    cart = bary_to_cart(bary_final)
    return cart
