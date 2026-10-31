import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0
H = np.sqrt(3) / 2

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle index combinations
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

def bary_to_cart(bary_pts):
    """
    Convert barycentric coordinates to Cartesian.
    bary_pts: (N, 2) array of (u, v). w = 1 - u - v.
    """
    u = bary_pts[:, 0]
    v = bary_pts[:, 1]
    w = 1.0 - u - v
    
    # Cartesian: x = v*1 + w*0.5, y = v*0 + w*H
    x = v + 0.5 * w
    y = H * w
    return np.column_stack((x, y))

def clamp_bary(bary_pts):
    """
    Clamp barycentric coordinates to the valid simplex.
    u >= 0, v >= 0, u+v <= 1
    """
    u = np.clip(bary_pts[:, 0], 0.0, 1.0)
    v = np.clip(bary_pts[:, 1], 0.0, 1.0)
    
    # Handle u + v > 1
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        # Scale down u and v proportionally to make sum 1
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    return np.column_stack((u, v))

def compute_min_area_bary(bary_pts):
    """
    Compute minimum triangle area for points given in barycentric coords.
    """
    cart_pts = bary_to_cart(bary_pts)
    
    pi = cart_pts[tri_idx[:, 0]]
    pj = cart_pts[tri_idx[:, 1]]
    pk = cart_pts[tri_idx[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective_bary(x):
    """
    Objective for optimization: minimize negative min area.
    x: (2N,) array of barycentric coords.
    """
    bary_pts = x.reshape(N, 2)
    bary_pts = clamp_bary(bary_pts)
    min_area = compute_min_area_bary(bary_pts)
    return -min_area

def simulated_annealing_bary(bary_init, time_limit=10.0):
    """
    Simulated Annealing in Barycentric space.
    """
    start_time = time.time()
    current_bary = bary_init.copy()
    current_val = -objective_bary(current_bary.flatten())
    best_bary = current_bary.copy()
    best_val = current_val
    
    T = 0.02  # Initial temperature
    cooling = 0.999
    min_T = 1e-9
    
    rng = np.random.default_rng()
    
    while time.time() - start_time < time_limit and T > min_T:
        # Perturb one point
        p_idx = rng.integers(N)
        
        # Random step in barycentric space
        # Step size should be scaled by T
        delta = rng.normal(0, T, 2)
        
        new_bary = current_bary.copy()
        new_bary[p_idx] += delta
        new_bary = clamp_bary(new_bary)
        
        new_val = -objective_bary(new_bary.flatten())
        delta_obj = new_val - current_val
        
        # Accept if better or with probability exp(delta_obj / T)
        if delta_obj > 0 or (T > 0 and rng.random() < np.exp(delta_obj / T)):
            current_bary = new_bary
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
        
        T *= cooling
        
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    # Phase 1: Multi-start Simulated Annealing
    # Use several restarts with different random seeds
    num_restarts = 5
    time_per_restart = (TIME_LIMIT - 10) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize random points in barycentric space
        # Uniform in triangle: sample u, v such that u>=0, v>=0, u+v<=1
        rng = np.random.default_rng(42 + i)
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1.0
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]
        
        init_bary = np.column_stack((u, v))
        
        # Run SA
        sa_time = time_per_restart
        candidate_bary = simulated_annealing_bary(init_bary, time_limit=sa_time)
        candidate_val = -objective_bary(candidate_bary.flatten())
        
        if candidate_val > best_val:
            best_val = candidate_val
            best_bary = candidate_bary
    
    # Phase 2: Final Polish with Nelder-Mead on the best solution
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_bary is not None:
        try:
            nm_result = minimize(
                objective_bary,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
            )
            nm_bary = clamp_bary(nm_result.x.reshape(N, 2))
            nm_val = -objective_bary(nm_bary.flatten())
            
            if nm_val > best_val:
                best_bary = nm_bary
                best_val = nm_val
        except Exception:
            pass
            
    # Convert best barycentric to Cartesian
    best_cart = bary_to_cart(best_bary)
    
    return best_cart

# EVOLVE-BLOCK-END
