"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _radius(c):
    """Compute the packing radius for a configuration c of shape (N, 3)."""
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    dists = pdist(c, metric='euclidean')
    if len(dists) > 0:
        min_pair = np.min(dists)
    else:
        min_pair = np.inf
    return min(min_wall, min_pair / 2.0)

def _objective(c_flat):
    """Objective function to minimize (negative radius)."""
    c = c_flat.reshape(N, 3)
    c = np.clip(c, -0.5, 0.5)
    r = _radius(c)
    return -r

def _local_refine(c_flat, max_iter=20, ftol=1e-8):
    """
    Perform local optimization step using L-BFGS-B.
    """
    try:
        bounds = [(-0.5, 0.5)] * (N * 3)
        result = minimize(_objective, c_flat, method='L-BFGS-B', 
                          bounds=bounds, 
                          options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-5})
        if result.fun < _objective(c_flat):
            return result.x
        else:
            return c_flat
    except Exception:
        return c_flat

def _init_random():
    """Initialize with random uniform points."""
    return np.random.uniform(-0.45, 0.45, (N, 3)).ravel()

def _init_lattice(type_id):
    """
    Initialize with a lattice structure.
    type_id selects the lattice type.
    """
    points = np.zeros((N, 3))
    if type_id == 0:
        # Simple Cubic 5x5x4
        nx, ny, nz = 5, 5, 4
        a = 1.0 / max(nx, ny, nz)
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if idx >= N: break
                    points[idx] = [(i - (nx-1)/2.0)*a, (j - (ny-1)/2.0)*a, (k - (nz-1)/2.0)*a]
                    idx += 1
    elif type_id == 1:
        # FCC-like (perturbed)
        nx, ny, nz = 5, 5, 4
        a = 1.0 / max(nx, ny, nz)
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if idx >= N: break
                    x = (i - (nx-1)/2.0)*a
                    y = (j - (ny-1)/2.0)*a
                    z = (k - (nz-1)/2.0)*a
                    points[idx] = [x, y, z]
                    idx += 1
    else:
        # BCC-like
        nx, ny, nz = 5, 5, 4
        a = 1.0 / max(nx, ny, nz)
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if idx >= N: break
                    points[idx] = [(i - (nx-1)/2.0)*a, (j - (ny-1)/2.0)*a, (k - (nz-1)/2.0)*a]
                    idx += 1
                    
    # Add noise to break symmetry
    points += np.random.randn(N, 3) * 0.01
    return np.clip(points, -0.49, 0.49).ravel()

def construct():
    np.random.seed(42)
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    swarm_size = 20
    w_start = 0.9
    w_end = 0.4
    c1 = 2.0
    c2 = 2.0
    
    # Initialize swarm
    particles = np.zeros((swarm_size, N * 3))
    
    for i in range(swarm_size):
        if i < swarm_size // 2:
            particles[i] = _init_random()
        else:
            particles[i] = _init_lattice(i % 3)
            
    # Evaluate initial positions and apply local refinement
    personal_bests = np.zeros_like(particles)
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        if time.time() - start_time > time_limit:
            break
        refined = _local_refine(particles[i], max_iter=10, ftol=1e-6)
        particles[i] = refined
        val = _objective(particles[i])
        personal_best_vals[i] = val
        personal_bests[i] = particles[i].copy()
        
    global_best_idx = np.argmin(personal_best_vals)
    global_best = personal_bests[global_best_idx].copy()
    global_best_val = personal_best_vals[global_best_idx]
    
    velocities = np.zeros_like(particles)
    
    # PSO Loop
    max_iter = 100
    stagnant_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Adaptive inertia
        w = w_start - (w_start - w_end) * (it / max_iter)
        
        improved = False
        
        for i in range(swarm_size):
            if time.time() - start_time > time_limit:
                break
                
            r1 = np.random.rand(N * 3)
            r2 = np.random.rand(N * 3)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_bests[i] - particles[i]) + 
                           c2 * r2 * (global_best - particles[i]))
            
            # Velocity clamping
            v_max = 0.1
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            particles[i] += velocities[i]
            
            # Clip to bounds
            particles[i] = np.clip(particles[i], -0.49, 0.49)
            
            # Local refinement
            refined = _local_refine(particles[i], max_iter=5, ftol=1e-5)
            particles[i] = refined
            
            # Evaluate
            val = _objective(particles[i])
            
            if val < personal_best_vals[i]:
                personal_best_vals[i] = val
                personal_bests[i] = particles[i].copy()
                
                if val < global_best_val:
                    global_best_val = val
                    global_best = particles[i].copy()
                    improved = True
                    
        if improved:
            stagnant_count = 0
        else:
            stagnant_count += 1
            
        # Stagnation handling
        if stagnant_count > 10:
            # Perturb global best and reset some particles
            global_best += np.random.randn(N * 3) * 0.05
            global_best = np.clip(global_best, -0.49, 0.49)
            # Reset half the swarm
            for i in range(swarm_size // 2):
                particles[i] = _init_random()
                velocities[i] = 0
                # Quick refine
                refined = _local_refine(particles[i], max_iter=5, ftol=1e-5)
                particles[i] = refined
                val = _objective(particles[i])
                if val < personal_best_vals[i]:
                    personal_best_vals[i] = val
                    personal_bests[i] = particles[i].copy()
            stagnant_count = 0
            
    # Final refinement on the best solution
    if time.time() - start_time < time_limit - 5:
        result = minimize(_objective, global_best, method='L-BFGS-B', 
                          bounds=[(-0.5, 0.5)] * (N * 3), 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        global_best = result.x
    
    c_final = np.clip(global_best.reshape(N, 3), -0.5, 0.5)
    
    return c_final
# EVOLVE-BLOCK-END
