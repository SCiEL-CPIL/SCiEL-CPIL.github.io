"""Simulated annealing packing of 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import random
import time

N = 50

def compute_radius(pos):
    """Compute the actual radius of the packing."""
    pos = pos.reshape(-1, 2)
    dists = np.sqrt(np.sum(pos**2, axis=1))
    if np.any(dists > 1.0):
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    min_boundary_dist = np.min(1.0 - dists)
    return min(min_pair_dist / 2.0, min_boundary_dist)

def generate_hex_initial(seed, spacing):
    """Generate a hexagonal lattice initial configuration."""
    np.random.seed(seed)
    points = []
    sqrt3 = np.sqrt(3.0)
    
    # Generate a larger grid and filter
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing + (j % 2) * spacing / 2.0
            y = j * spacing * sqrt3 / 2.0
            # Add small random perturbation to break symmetry
            x += np.random.uniform(-0.005, 0.005)
            y += np.random.uniform(-0.005, 0.005)
            if x*x + y*y <= 0.95:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) > N:
        # Keep the N points closest to the center to ensure a compact start
        dists = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # If we don't have enough, add random points inside
        while len(points) < N:
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, 0.8)
            points = np.vstack([points, [[r*np.cos(angle), r*np.sin(angle)]]])
    
    return points

def sa_packing(seed, time_limit=10.0):
    """
    Run Simulated Annealing to find a good packing.
    """
    np.random.seed(seed)
    
    # 1. Initialization
    # Estimate optimal spacing. Area per circle ~ pi*r^2. Total area ~ pi*1^2.
    # r ~ 1/sqrt(N) * factor. For N=50, r~0.14. Spacing ~ 2*r ~ 0.28.
    spacing = 0.27 
    pos = generate_hex_initial(seed, spacing)
    
    # Ensure we start with a valid configuration (no overlaps)
    # If overlaps exist, we might need to adjust, but hex grid with correct spacing should be fine.
    # Let's verify and slightly perturb if necessary.
    
    # 2. Simulated Annealing Setup
    T = 0.1
    T_min = 1e-6
    cooling_rate = 0.9995
    max_iter = 200000
    start_time = time.time()
    
    best_pos = pos.copy()
    best_r = compute_radius(pos)
    current_r = best_r
    
    # We will optimize the radius directly by treating it as a parameter or by scaling?
    # Better approach: Optimize positions to maximize the minimum distance.
    # Energy function: E = sum( (2r - d_ij)^2 for d_ij < 2r ) + sum( (r + |c_i| - 1)^2 for |c_i| > 1-r )
    # But r is unknown. We can fix r and minimize violations, then increase r.
    # Or, we can use a soft potential that pushes circles apart.
    
    # Let's use a soft potential approach where we push circles apart if they are too close.
    # We want to maximize the minimum distance.
    # Let's define a function that computes the "energy" of a configuration for a given target radius r.
    # If we can achieve a configuration with no overlaps at radius r, it's valid.
    
    # Alternative: Directly optimize the objective function using a stochastic gradient-like method?
    # No, SA is better.
    
    # Let's try to maximize the radius by iteratively increasing a target radius and minimizing overlaps.
    
    # Initial radius estimate
    r_current = compute_radius(pos)
    if r_current < 0.1:
        r_current = 0.12
    
    # We will run SA for a fixed r, then increase r.
    # But standard SA on positions with a fixed r might get stuck.
    
    # Let's use a different energy: 
    # E = sum_{i<j} max(0, 2*r_target - d_ij)^2 + sum_i max(0, r_target + |c_i| - 1)^2
    # We start with a small r_target and increase it.
    
    r_target = r_current * 1.05
    
    def energy(pos_flat, r_t):
        p = pos_flat.reshape(-1, 2)
        dists_to_center = np.sqrt(np.sum(p**2, axis=1))
        
        # Boundary penalty
        boundary_viol = np.maximum(0, dists_to_center - (1.0 - r_t))
        boundary_pen = np.sum(boundary_viol**2)
        
        # Pairwise penalty
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, 1.0)
        
        close_pairs = np.maximum(0, 2.0 * r_t - dist_mat)
        pair_pen = np.sum(close_pairs**2)
        
        return boundary_pen + pair_pen

    # Run SA
    pos = pos.copy()
    current_energy = energy(pos, r_target)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive step size
        step_size = 0.05 * (1.0 + T)
        
        idx = np.random.randint(0, N)
        new_pos = pos.copy()
        new_pos[idx] += np.random.normal(0, step_size, 2)
        
        new_energy = energy(new_pos, r_target)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            pos = new_pos
            current_energy = new_energy
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Check if we have a valid configuration and update best
        # We only update best if energy is 0 (or very close)
        if current_energy < 1e-8:
            r_check = compute_radius(pos)
            if r_check > best_r:
                best_r = r_check
                best_pos = pos.copy()
                
    # After SA, try to increase r_target and re-optimize
    # This is a bit slow if done many times. Let's do a few iterations.
    
    # Local refinement
    pos = best_pos
    r_final = compute_radius(pos)
    
    return pos, r_final

def local_refinement(pos):
    """
    Perform local optimization to maximize the minimum radius.
    """
    pos = pos.reshape(-1, 2).copy()
    
    def objective(x):
        p = x.reshape(-1, 2)
        dists_to_center = np.sqrt(np.sum(p**2, axis=1))
        
        # Boundary constraint: |c_i| + r <= 1 => r <= 1 - |c_i|
        # Pairwise constraint: d_ij >= 2r => r <= d_ij / 2
        
        # We want to maximize r.
        # Equivalent to minimizing -r.
        # But r is min(min_pair/2, min_boundary).
        # This is a non-smooth function.
        
        # We can use a penalty method or just minimize the violations for a fixed r?
        # No, we want to find the max r.
        
        # Let's try to minimize the sum of squares of violations for a fixed r, and binary search on r?
        # Too slow.
        
        # Alternative: Use a smooth approximation of the min function.
        # Or, simply use scipy minimize on a function that returns -min_radius.
        # This works if the landscape is smooth enough.
        
        min_pair = np.inf
        min_boundary = np.inf
        
        # Vectorized min pair distance is expensive but N=50 is small.
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair = np.min(dist_mat)
        
        min_boundary = np.min(1.0 - dists_to_center)
        
        r = min(min_pair / 2.0, min_boundary)
        
        return -r

    # Use Powell or Nelder-Mead
    try:
        result = minimize(objective, pos.reshape(-1), method='Powell', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        new_pos = result.x.reshape(-1, 2)
        new_r = -result.fun
        if new_r > compute_radius(pos):
            pos = new_pos
    except Exception:
        pass
        
    return pos

def construct():
    np.random.seed(42)
    random.seed(42)
    
    best_pos = None
    best_r = 0.0
    start_time = time.time()
    
    # Try multiple restarts
    for restart in range(5):
        if time.time() - start_time > 110:
            break
            
        seed = 42 + restart * 101
        time_limit = 20.0
        
        pos, r = sa_packing(seed, time_limit)
        
        # Local refinement
        pos = local_refinement(pos)
        r = compute_radius(pos)
        
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
            
    if best_pos is None:
        # Fallback
        best_pos = generate_hex_initial(42, 0.27)
        
    return best_pos
