import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 110.0  # Safety margin for 120s limit

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

def get_min_area(bary):
    """Get the minimum triangle area for a barycentric configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def generate_structured_inits():
    """
    Generate multiple diverse structured initializations.
    """
    inits = []
    
    # 1. Hexagonal-like grid (standard)
    # Vertices
    pts1 = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [1/3, 1/3],
        [1/3 + 0.15, 1/3],
        [1/3 - 0.15, 1/3],
        [1/3, 1/3 + 0.15],
        [1/3, 1/3 - 0.15],
    ]
    inits.append(clamp_bary(np.array(pts1)))
    
    # 2. Random Grid (Jittered Grid)
    rng = np.random.default_rng(1)
    # Create a 3x3 grid in barycentric space, pick 11 points
    # Barycentric grid points:
    # (0,0), (1/3,0), (2/3,0), (1,0)
    # (0,1/3), (1/3,1/3), (2/3,1/3)
    # (0,2/3), (1/3,2/3)
    # (0,1)
    grid_pts = [
        [0.0, 0.0], [1/3, 0.0], [2/3, 0.0], [1.0, 0.0],
        [0.0, 1/3], [1/3, 1/3], [2/3, 1/3],
        [0.0, 2/3], [1/3, 2/3],
        [0.0, 1.0]
    ]
    # We need 11 points. The grid has 10 valid interior/edge points if we exclude vertices?
    # Actually, let's just pick 11 random points from a fine grid.
    u_grid = np.linspace(0.05, 0.95, 10)
    v_grid = np.linspace(0.05, 0.95, 10)
    candidates = []
    for u in u_grid:
        for v in v_grid:
            if u + v <= 0.95:
                candidates.append([u, v])
    rng.shuffle(candidates)
    if len(candidates) >= 11:
        inits.append(clamp_bary(np.array(candidates[:11])))
        
    # 3. Boundary Focused
    # Place 5 on edges, 6 inside
    pts3 = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], # Vertices
        [0.25, 0.0], [0.75, 0.0], # Edge 1
        [0.0, 0.25], [0.0, 0.75], # Edge 2
        [0.25, 0.75], [0.75, 0.25], # Edge 3 (wait, u+v=1)
        # Interior
        [0.3, 0.3],
        [0.4, 0.2],
        [0.2, 0.4],
        [0.5, 0.3],
        [0.3, 0.5],
    ]
    inits.append(clamp_bary(np.array(pts3)))
    
    # 4. Pure Random
    rng2 = np.random.default_rng(2)
    u = rng2.random(N)
    v = rng2.random(N) * (1 - u)
    inits.append(clamp_bary(np.column_stack([u, v])))
    
    return inits

def coordinate_descent(bary_init, time_limit=5.0, seed=42):
    """
    Refine configuration using Coordinate Descent with local Nelder-Mead on single points.
    This is faster and often more robust than full Nelder-Mead or SA for this problem.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    current = bary_init.copy()
    current_val = get_min_area(current)
    
    best = current.copy()
    best_val = current_val
    
    # We will iterate through points and try to optimize each point's position
    # by running a quick local minimization (Nelder-Mead) on just that point's 2 coords.
    
    max_rounds = 100
    for _ in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle order to avoid bias
        order = rng.permutation(N)
        
        for idx in order:
            if time.time() - start_time > time_limit:
                break
                
            # Extract current position
            u0, v0 = current[idx]
            
            # Define objective for this single point
            def obj_single(params):
                u, v = params
                new_bary = current.copy()
                new_bary[idx, 0] = u
                new_bary[idx, 1] = v
                new_bary = clamp_bary(new_bary)
                return -get_min_area(new_bary)
            
            try:
                # Use Nelder-Mead for 2D optimization
                res = minimize(
                    obj_single, 
                    [u0, v0], 
                    method='Nelder-Mead',
                    options={'maxiter': 20, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
                )
                new_u, new_v = res.x
                new_bary = current.copy()
                new_bary[idx, 0] = new_u
                new_bary[idx, 1] = new_v
                new_bary = clamp_bary(new_bary)
                
                new_val = get_min_area(new_bary)
                
                if new_val > current_val:
                    current = new_bary
                    current_val = new_val
                    if current_val > best_val:
                        best_val = current_val
                        best = current.copy()
                        
            except Exception:
                pass
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate diverse structured inits
    inits = generate_structured_inits()
    
    # 2. Refine each init with Coordinate Descent
    # Allocate time evenly among inits, leaving some buffer
    num_inits = len(inits)
    time_per_init = (TIME_LIMIT - 10) / num_inits  # 10s buffer for final polish
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Update best with raw init first
        update_best(init)
        
        # Refine
        refined = coordinate_descent(init, time_limit=time_per_init, seed=100 + i)
        update_best(refined)
        
    # 3. Final Polish on the best solution found
    if best_bary is not None and time.time() < deadline - 5:
        # Run a few more rounds of CD on the best
        final_refined = coordinate_descent(best_bary, time_limit=5.0, seed=999)
        update_best(final_refined)

    # Fallback if something went wrong (shouldn't happen)
    if best_bary is None:
        best_bary = np.zeros((N, 2))
        
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
