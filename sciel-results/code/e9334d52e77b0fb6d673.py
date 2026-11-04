# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 100

def _radius(c):
    """Compute the packing radius for a configuration c of shape (N, 3)."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    dists = pdist(c, metric='euclidean')
    if len(dists) > 0:
        min_pair = np.min(dists)
    else:
        min_pair = np.inf
        
    return min(min_wall, min_pair / 2.0)

def _objective(c_flat):
    """Objective function to minimize (negative radius)."""
    c = c_flat.reshape(N, 3)
    # Ensure feasibility
    c = np.clip(c, -0.5, 0.5)
    r = _radius(c)
    return -r

def _local_refine(c_flat, max_iter=20, tol=1e-8):
    """
    Perform local optimization using L-BFGS-B.
    Returns the refined flat array.
    """
    try:
        result = minimize(_objective, c_flat, method='L-BFGS-B', 
                          bounds=[(-0.5, 0.5)] * (N * 3), 
                          options={'maxiter': max_iter, 'ftol': tol})
        if result.fun < _objective(c_flat):
            return result.x
        else:
            return c_flat
    except Exception:
        return c_flat

def _generate_grid_init():
    """Generate a 4x5x5 grid initialization."""
    nx, ny, nz = 4, 5, 5
    points = []
    # Center the grid
    # Spacing a = 1 / max(nx, ny, nz) * factor
    # To fit 100 spheres, 4*5*5=100 is perfect.
    # We want to maximize the minimum distance.
    # A uniform grid is a good start.
    xs = np.linspace(-0.5 + 0.05, 0.5 - 0.05, nx)
    ys = np.linspace(-0.5 + 0.05, 0.5 - 0.05, ny)
    zs = np.linspace(-0.5 + 0.05, 0.5 - 0.05, nz)
    
    for x in xs:
        for y in ys:
            for z in zs:
                points.append([x, y, z])
                
    points = np.array(points[:N])
    return points.ravel()

def construct():
    np.random.seed(42)
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer for final polish and overhead
    
    swarm_size = 30
    max_iter = 100 # Reduced iterations to allow for time check
    
    w_start = 0.9
    w_end = 0.4
    c1 = 1.5
    c2 = 1.5
    v_max = 0.05 # Clamp velocity to prevent explosion
    
    particles = np.zeros((swarm_size, N * 3))
    velocities = np.zeros((swarm_size, N * 3))
    
    # Initialization
    base_grid = _generate_grid_init()
    
    for i in range(swarm_size):
        if i == 0:
            particles[i] = base_grid
        elif i < 5:
            # Perturbed grid
            particles[i] = base_grid + np.random.randn(N * 3) * 0.02
        else:
            # Random uniform
            particles[i] = np.random.uniform(-0.45, 0.45, N * 3)
            
    # Evaluate initial positions
    # We do a quick refinement on the initial grid-based particles to give them a head start
    personal_bests = np.zeros_like(particles)
    personal_best_vals = np.full(swarm_size, np.inf)
    
    for i in range(swarm_size):
        # Light refinement for initialization
        refined = _local_refine(particles[i], max_iter=10, tol=1e-6)
        particles[i] = refined
        val = _objective(particles[i])
        personal_best_vals[i] = val
        personal_bests[i] = particles[i].copy()
        
    global_best_idx = np.argmin(personal_best_vals)
    global_best = personal_bests[global_best_idx].copy()
    global_best_val = personal_best_vals[global_best_idx]
    
    for it in range(max_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        w = w_start - (w_start - w_end) * (it / max_iter)
        
        for i in range(swarm_size):
            r1 = np.random.rand(N * 3)
            r2 = np.random.rand(N * 3)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_bests[i] - particles[i]) + 
                           c2 * r2 * (global_best - particles[i]))
            
            # Clamp velocity
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            particles[i] += velocities[i]
            
            # Clip to bounds
            particles[i] = np.clip(particles[i], -0.5, 0.5)
            
            # Evaluate
            val = _objective(particles[i])
            
            if val < personal_best_vals[i]:
                # New personal best: Refine it locally
                refined = _local_refine(particles[i], max_iter=15, tol=1e-7)
                particles[i] = refined
                val = _objective(particles[i])
                
                if val < personal_best_vals[i]:
                    personal_best_vals[i] = val
                    personal_bests[i] = particles[i].copy()
                    
                    if val < global_best_val:
                        global_best_val = val
                        global_best = particles[i].copy()
                        
    # Final Polish
    # Use the best found so far
    c_best_flat = global_best
    # Run a more thorough local optimization
    # Use higher maxiter and tighter tolerance
    final_result = _local_refine(c_best_flat, max_iter=200, tol=1e-10)
    
    # One more try with Nelder-Mead if L-BFGS gets stuck? 
    # L-BFGS is usually good for smooth landscapes like this.
    
    c_final = np.clip(final_result.reshape(N, 3), -0.5, 0.5)
    
    return c_final
# EVOLVE-BLOCK-END
