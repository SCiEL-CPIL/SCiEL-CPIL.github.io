import numpy as np
import time
import sys
from scipy.optimize import minimize

# Grid definition
# x: -3 to 1.5 (exclusive of 1.5? np.arange(-3, 1.5, 0.01) -> 450 points)
# y: -1.5 to 1.5 (exclusive of 1.5? np.arange(-1.5, 1.5, 0.01) -> 300 points)
# Wait, np.arange(-1.5, 1.5, 0.01) gives 300 points.
# Total 135,000 points.

X, Y = np.meshgrid(np.arange(-3, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
GRID_POINTS = np.column_stack((X.ravel(), Y.ravel()))
N_POINTS = len(GRID_POINTS)

# Precompute for speed?
# We need to check if points are in the corridor.

def get_corridor_mask(x, y, a_deg):
    """
    Returns a boolean mask of shape (N_POINTS,) indicating which grid points
    are inside the corridor placement (x, y, a_deg).
    """
    a_rad = np.deg2rad(a_deg)
    cos_a = np.cos(-a_rad)
    sin_a = np.sin(-a_rad)
    
    # Translate
    dx = GRID_POINTS[:, 0] - x
    dy = GRID_POINTS[:, 1] - y
    
    # Rotate
    u = cos_a * dx - sin_a * dy
    v = sin_a * dx + cos_a * dy
    
    # Base corridor conditions
    # R1: 0 <= u <= 1, -3 <= v <= 1
    mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # R2: -3 <= u <= 1, 0 <= v <= 1
    mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    return mask1 | mask2

def compute_area(path):
    """
    path: np.array of shape (n, 3)
    Returns the area (count of points * 0.0001)
    """
    # Start
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    # Construct full sequence
    seq = np.vstack([start, path, end])
    
    # Initialize mask with Start
    mask = get_corridor_mask(0.0, 0.0, 0.0)
    
    for i in range(len(seq) - 1):
        p1 = seq[i]
        p2 = seq[i + 1]
        
        a1 = p1[2]
        a2 = p2[2]
        
        m = max(1, int(np.ceil(abs(a2 - a1))))
        
        # Interpolate
        for k in range(1, m + 1):
            t = k / m
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            a = a1 + t * (a2 - a1)
            
            step_mask = get_corridor_mask(x, y, a)
            mask &= step_mask
            
    return np.sum(mask) * 0.0001

def objective(params):
    """
    params: flat array of x, y for the intermediate points.
    Angles are fixed.
    """
    n_points = 8
    # Reconstruct path
    path = np.zeros((n_points, 3))
    # Fixed angles: 10, 20, ..., 80
    angles = np.arange(10, 90, 10.0)
    path[:, 2] = angles
    path[:, 0] = params[0:n_points]
    path[:, 1] = params[n_points:2*n_points]
    
    area = compute_area(path)
    return -area # Minimize negative area

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 500  # 500s limit
    
    # Initial guess
    # Try to keep the corner near the "center" of the L
    # x from 0 to -2, y from 0 to 0
    # Maybe an arc?
    n_points = 8
    angles = np.arange(10, 90, 10.0)
    
    # Initial x, y
    # Linear interpolation
    x_init = np.linspace(0, -2, n_points)
    y_init = np.linspace(0, 0, n_points)
    
    params_init = np.concatenate([x_init, y_init])
    
    best_params = params_init.copy()
    best_score = -objective(params_init)
    
    # Optimization
    # Use Nelder-Mead
    # We might need to restart or use multiple trials
    
    # To save time, we can do a few random restarts or just one long run
    # Nelder-Mead can be slow if not tuned.
    
    # Let's try Powell or Nelder-Mead with maxiter
    
    # Check time
    if time.time() < deadline:
        try:
            res = minimize(objective, params_init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception as e:
            pass # Keep initial if fails
            
    # Try a second optimization from a different start if time permits
    if time.time() < deadline - 100:
        # Perturb initial
        np.random.seed(42)
        params_init2 = params_init + np.random.normal(0, 0.5, len(params_init))
        try:
            res2 = minimize(objective, params_init2, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            if res2.fun < -best_score:
                best_score = -res2.fun
                best_params = res2.x
        except Exception as e:
            pass
            
    # Construct final path
    path = np.zeros((n_points, 3))
    angles = np.arange(10, 90, 10.0)
    path[:, 2] = angles
    path[:, 0] = best_params[0:n_points]
    path[:, 1] = best_params[n_points:2*n_points]
    
    return path

# EVOLVE-BLOCK-END
