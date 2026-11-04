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

def _local_refine(c_flat, max_iter=30, ftol=1e-9):
    """
    Perform local optimization step using L-BFGS-B with multiple restarts.
    """
    best_x = c_flat.copy()
    best_val = _objective(best_x)
    
    for restart in range(3):
        try:
            if restart == 0:
                x0 = best_x
            else:
                # Small perturbation for restart
                x0 = best_x + np.random.randn(N * 3) * 0.001 * (restart)
                x0 = np.clip(x0, -0.49, 0.49)
            
            bounds = [(-0.5, 0.5)] * (N * 3)
            result = minimize(_objective, x0, method='L-BFGS-B', 
                          bounds=bounds, 
                          options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-6})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    return best_x

def _generate_bcc_init():
    """
    Generate initial configuration based on Body-Centered Cubic lattice.
    This provides better packing density than simple cubic.
    """
    points = []
    # Use a 5x5x4 arrangement with BCC-like offsets
    nx, ny, nz = 5, 5, 4
    spacing = 1.0 / max(nx, ny, nz) * 0.95
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if len(points) >= N:
                    break
                x = (i - (nx - 1) / 2.0) * spacing
                y = (j - (ny - 1) / 2.0) * spacing
                z = (k - (nz - 1) / 2.0) * spacing
                points.append([x, y, z])
    
    points = np.array(points[:N])
    # Add small random perturbation
    points += np.random.randn(N, 3) * 0.003
    return points.ravel()

def _generate_fcc_init():
    """
    Generate initial configuration based on Face-Centered Cubic lattice.
    """
    points = []
    spacing = 1.0 / 5 * 0.98
    
    # FCC lattice points
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                if len(points) >= N:
                    break
                # FCC basis vectors
                x = i * spacing
                y = j * spacing
                z = k * spacing
                
                # Add face center offsets
                if (i + j + k) % 2 == 0:
                    points.append([x, y, z])
                else:
                    points.append([x + spacing/2, y + spacing/2, z + spacing/2])
    
    # Ensure we have exactly N points
    if len(points) < N:
        while len(points) < N:
            points.append(np.random.uniform(-0.4, 0.4, 3).tolist())
    
    points = np.array(points[:N])
    points += np.random.randn(N, 3) * 0.002
    return points.ravel()

def _generate_random_init():
    """Generate random initial configuration."""
    return np.random.uniform(-0.45, 0.45, N * 3)

def construct():
    np.random.seed(42)
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # PSO parameters
    swarm_size = 20
    w_start = 0.9
    w_end = 0.3
    c1_start = 2.0
    c1_end = 1.5
    c2_start = 2.0
    c2_end = 1.5
    
    # Initialize swarm with diverse starting points
    particles = np.zeros((swarm_size, N * 3))
    
    # Different initialization strategies
    bcc_init = _generate_bcc_init()
    fcc_init = _generate_fcc_init()
    
    for i in range(swarm_size):
        if i < 5:
            # BCC-based with varying noise
            noise_level = 0.01 * (i + 1)
            particles[i] = bcc_init + np.random.randn(N * 3) * noise_level
        elif i < 10:
            # FCC-based with varying noise
            noise_level = 0.01 * (i - 4 + 1)
            particles[i] = fcc_init + np.random.randn(N * 3) * noise_level
        elif i < 15:
            # Rotated/scaled BCC
            scale = 0.85 + 0.03 * (i - 9)
            pts = bcc_init.reshape(N, 3) * scale
            # Apply random rotation
            theta = np.random.uniform(0, np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])
            pts = pts @ rot.T
            particles[i] = pts.ravel()
        else:
            # Random initialization
            particles[i] = _generate_random_init()
        
        # Clip to valid range
        particles[i] = np.clip(particles[i], -0.49, 0.49)
    
    # Evaluate initial positions and apply local refinement
    personal_bests = np.zeros_like(particles)
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        if time.time() - start_time > time_limit:
            break
        refined = _local_refine(particles[i], max_iter=20, ftol=1e-8)
        particles[i] = refined
        val = _objective(particles[i])
        personal_best_vals[i] = val
        personal_bests[i] = particles[i].copy()
        
    global_best_idx = np.argmin(personal_best_vals)
    global_best = personal_bests[global_best_idx].copy()
    global_best_val = personal_best_vals[global_best_idx]
    
    velocities = np.zeros_like(particles)
    
    # Adaptive PSO loop
    max_iter = 40
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Adaptive parameters
        progress = it / max_iter
        w = w_start - (w_start - w_end) * progress
        c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start - (c2_start - c2_end) * progress
        
        for i in range(swarm_size):
            if time.time() - start_time > time_limit:
                break
                
            r1 = np.random.rand(N * 3)
            r2 = np.random.rand(N * 3)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_bests[i] - particles[i]) + 
                           c2 * r2 * (global_best - particles[i]))
            
            # Adaptive velocity clamping
            v_max = 0.1 * (1.0 - 0.5 * progress)
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            particles[i] += velocities[i]
            
            # Clip to bounds
            particles[i] = np.clip(particles[i], -0.49, 0.49)
            
            # Local refinement with adaptive iterations
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time > 30:
                iter_limit = 15
            elif remaining_time > 15:
                iter_limit = 8
            else:
                iter_limit = 4
            refined = _local_refine(particles[i], max_iter=iter_limit, ftol=1e-7)
            particles[i] = refined
            
            # Evaluate
            val = _objective(particles[i])
            
            if val < personal_best_vals[i]:
                personal_best_vals[i] = val
                personal_bests[i] = particles[i].copy()
                
                if val < global_best_val:
                    global_best_val = val
                    global_best = particles[i].copy()
                    
    # Final refinement on the best solution
    if time.time() - start_time < time_limit - 15:
        result = minimize(_objective, global_best, method='L-BFGS-B', 
                          bounds=[(-0.5, 0.5)] * (N * 3), 
                          options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-7})
        global_best = result.x
    
    c_final = np.clip(global_best.reshape(N, 3), -0.5, 0.5)
    
    return c_final
# EVOLVE-BLOCK-END
