"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _radius_fast(c):
    """Compute packing radius using vectorized operations for speed."""
    c = np.clip(c, -0.499, 0.499)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    dists = pdist(c, metric='euclidean')
    min_pair = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_wall, min_pair / 2.0)

def _objective(c_flat):
    """Objective function to minimize (negative radius)."""
    c = c_flat.reshape(N, 3)
    r = _radius_fast(c)
    return -r

def _local_refine(c_flat, max_iter=30, ftol=1e-9, gtol=1e-6):
    """
    Perform local optimization step using L-BFGS-B with adaptive settings.
    """
    try:
        bounds = [(-0.499, 0.499)] * (N * 3)
        result = minimize(_objective, c_flat, method='L-BFGS-B', 
                          bounds=bounds, 
                          options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
        if result.fun < _objective(c_flat):
            return result.x
        else:
            return c_flat
    except Exception:
        return c_flat

def _generate_bcc_init():
    """
    Generate initial configuration based on body-centered cubic lattice.
    BCC has better packing density than simple cubic.
    """
    points = []
    # Use a 4x4x4 grid of BCC unit cells (64 points) + additional layers
    nx, ny, nz = 4, 4, 4
    a = 1.0 / (nx * 1.5) * 0.95  # Scale to fit
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner points
                x = (i - (nx - 1) / 2.0) * a
                y = (j - (ny - 1) / 2.0) * a
                z = (k - (nz - 1) / 2.0) * a
                points.append([x, y, z])
                # Body center point
                points.append([x + a/2, y + a/2, z + a/2])
    
    points = np.array(points)
    # Take first N points or repeat with perturbation if needed
    if len(points) >= N:
        points = points[:N]
    else:
        while len(points) < N:
            points = np.vstack([points, points + np.random.randn(len(points), 3) * 0.01])
        points = points[:N]
    
    points += np.random.randn(N, 3) * 0.003
    return points.ravel()

def _generate_fcc_init():
    """
    Generate initial configuration based on face-centered cubic lattice.
    FCC has the highest packing density for equal spheres.
    """
    points = []
    nx, ny, nz = 3, 3, 4
    a = 1.0 / (nx * 1.4) * 0.92
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = (i - (nx - 1) / 2.0) * a
                y = (j - (ny - 1) / 2.0) * a
                z = (k - (nz - 1) / 2.0) * a
                # FCC: corner + face centers
                points.append([x, y, z])
                if i < nx - 1:
                    points.append([x + a/2, y, z])
                if j < ny - 1:
                    points.append([x, y + a/2, z])
                if k < nz - 1:
                    points.append([x, y, z + a/2])
    
    points = np.array(points)
    if len(points) >= N:
        points = points[:N]
    else:
        while len(points) < N:
            points = np.vstack([points, points + np.random.randn(len(points), 3) * 0.01])
        points = points[:N]
    
    points += np.random.randn(N, 3) * 0.003
    return points.ravel()

def _generate_hcp_init():
    """
    Generate initial configuration based on hexagonal close packing.
    """
    points = []
    nx, ny = 5, 5
    a = 1.0 / (nx * 0.85) * 0.9
    dz = a * np.sqrt(2/3)
    nz = int(1.0 / dz) + 1
    
    for k in range(nz):
        offset_x = (k % 2) * a / 2
        for i in range(nx):
            for j in range(ny):
                x = (i - (nx - 1) / 2.0) * a + offset_x
                y = (j - (ny - 1) / 2.0) * a
                z = (k - (nz - 1) / 2.0) * dz
                points.append([x, y, z])
    
    points = np.array(points)
    if len(points) >= N:
        points = points[:N]
    else:
        while len(points) < N:
            points = np.vstack([points, points + np.random.randn(len(points), 3) * 0.01])
        points = points[:N]
    
    points += np.random.randn(N, 3) * 0.003
    return points.ravel()

def construct():
    np.random.seed(42)
    start_time = time.time()
    time_limit = 112  # Leave 8s buffer
    
    swarm_size = 20
    w_start = 0.9
    w_end = 0.3
    c1 = 2.0
    c2 = 1.5
    
    # Initialize swarm with diverse high-quality starting points
    particles = np.zeros((swarm_size, N * 3))
    
    # Generate multiple high-quality initial configurations
    init_configs = [
        _generate_bcc_init(),
        _generate_fcc_init(),
        _generate_hcp_init(),
    ]
    
    for i in range(swarm_size):
        if i < len(init_configs):
            particles[i] = init_configs[i]
        elif i < 8:
            # Perturbed BCC
            base = _generate_bcc_init()
            noise = 0.015 * (i - len(init_configs) + 1)
            particles[i] = base + np.random.randn(N * 3) * noise
        elif i < 12:
            # Perturbed FCC
            base = _generate_fcc_init()
            noise = 0.015 * (i - 7)
            particles[i] = base + np.random.randn(N * 3) * noise
        elif i < 16:
            # Perturbed HCP
            base = _generate_hcp_init()
            noise = 0.015 * (i - 11)
            particles[i] = base + np.random.randn(N * 3) * noise
        else:
            # Random with slight structure
            particles[i] = np.random.uniform(-0.4, 0.4, N * 3)
            
    # Evaluate initial positions and apply local refinement
    personal_bests = np.zeros_like(particles)
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        if time.time() - start_time > time_limit:
            break
        refined = _local_refine(particles[i], max_iter=25, ftol=1e-8, gtol=1e-5)
        particles[i] = refined
        val = _objective(particles[i])
        personal_best_vals[i] = val
        personal_bests[i] = particles[i].copy()
        
    global_best_idx = np.argmin(personal_best_vals)
    global_best = personal_bests[global_best_idx].copy()
    global_best_val = personal_best_vals[global_best_idx]
    
    velocities = np.zeros_like(particles)
    
    # Adaptive PSO loop
    max_iter = 60
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Adaptive inertia with smoother decay
        progress = it / max_iter
        w = w_start - (w_start - w_end) * (progress ** 0.8)
        
        # Adaptive cognitive and social coefficients
        c1_eff = c1 - 0.5 * progress
        c2_eff = c2 + 0.3 * progress
        
        for i in range(swarm_size):
            if time.time() - start_time > time_limit:
                break
                
            r1 = np.random.rand(N * 3)
            r2 = np.random.rand(N * 3)
            
            velocities[i] = (w * velocities[i] + 
                           c1_eff * r1 * (personal_bests[i] - particles[i]) + 
                           c2_eff * r2 * (global_best - particles[i]))
            
            # Adaptive velocity clamping
            v_max = 0.12 * (1 - 0.3 * progress)
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            particles[i] += velocities[i]
            
            # Clip to bounds
            particles[i] = np.clip(particles[i], -0.499, 0.499)
            
            # Local refinement step with adaptive iterations
            iter_limit = 15 if it < max_iter // 3 else 10 if it < 2 * max_iter // 3 else 5
            refined = _local_refine(particles[i], max_iter=iter_limit, ftol=1e-8, gtol=1e-5)
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
    if time.time() - start_time < time_limit - 5:
        result = minimize(_objective, global_best, method='L-BFGS-B', 
                          bounds=[(-0.499, 0.499)] * (N * 3), 
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
        global_best = result.x
    
    c_final = np.clip(global_best.reshape(N, 3), -0.5, 0.5)
    
    return c_final
# EVOLVE-BLOCK-END
