import numpy as np
import time
from scipy.optimize import minimize

N = 130

def construct():
    # 1. Initial Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    pts = np.column_stack([x, y, z])
    
    # 2. Probe Points
    # Use a denser Fibonacci spiral for probes to ensure good coverage
    N_probe = 20000
    i_p = np.arange(N_probe) + 0.5
    z_p = 1.0 - 2.0 * i_p / N_probe
    phi_p = np.pi * (1.0 + np.sqrt(5.0)) * i_p
    r_p = np.sqrt(1.0 - z_p * z_p)
    probes = np.column_stack([r_p * np.cos(phi_p), r_p * np.sin(phi_p), z_p])
    
    # 3. Objective Function
    def objective(flat_pts):
        p = flat_pts.reshape(N, 3)
        # Normalize to project back to sphere
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        
        # Compute dot products: (N_probe, N)
        # probes: (N_probe, 3), p: (N, 3)
        # dot = probes @ p.T
        dots = probes @ p.T
        # Clip to [-1, 1] for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        # Angular distances
        # We want min distance for each probe, then max of those
        # min angle = arccos(max dot)
        max_dots = np.max(dots, axis=1)
        min_angles = np.arccos(max_dots)
        # Covering radius is the max of these min angles
        return np.max(min_angles)
        
    # 4. Optimization
    start_time = time.time()
    timeout = 120 # seconds
    
    # Flatten initial points
    x0 = pts.flatten()
    
    # Run Nelder-Mead
    # maxiter needs to be set. 
    # Let's try 500 iterations.
    # If it's too slow, we might need to reduce probes or iterations.
    # We can't easily interrupt minimize, so we estimate.
    # Each eval: 20k * 130 mults ~ 2.6M flops. Very fast in numpy.
    # 500 iterations * 391 evals = 195,500 evals.
    # If each eval is 1ms, that's 195 seconds. Too slow.
    # Let's reduce probes to 5000 and iterations to 200.
    # 5000 * 130 = 650k flops. ~0.5ms?
    # 200 * 391 = 78,200 evals * 0.5ms = 39 seconds. Safe.
    
    N_probe_opt = 5000
    i_p_opt = np.arange(N_probe_opt) + 0.5
    z_p_opt = 1.0 - 2.0 * i_p_opt / N_probe_opt
    phi_p_opt = np.pi * (1.0 + np.sqrt(5.0)) * i_p_opt
    r_p_opt = np.sqrt(1.0 - z_p_opt * z_p_opt)
    probes_opt = np.column_stack([r_p_opt * np.cos(phi_p_opt), r_p_opt * np.sin(phi_p_opt), z_p_opt])
    
    def objective_opt(flat_pts):
        p = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        dots = probes_opt @ p.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_angles = np.arccos(max_dots)
        return np.max(min_angles)
        
    # Try to run
    try:
        res = minimize(objective_opt, x0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
        best_pts = res.x.reshape(N, 3)
    except Exception:
        best_pts = pts
        
    # Final check and normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
