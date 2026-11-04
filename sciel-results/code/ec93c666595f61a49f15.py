"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


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
    # Clip to valid range to ensure feasibility during optimization
    c = np.clip(c, -0.5, 0.5)
    r = _radius(c)
    return -r


def _local_refine(c_flat, max_iter=10):
    """
    Perform a quick local optimization step using L-BFGS-B.
    This helps PSO escape local minima and converge faster.
    """
    try:
        result = minimize(_objective, c_flat, method='L-BFGS-B', 
                          bounds=[(-0.5, 0.5)] * (N * 3), 
                          options={'maxiter': max_iter, 'ftol': 1e-8})
        if result.success or result.fun < _objective(c_flat):
            return result.x
        else:
            return c_flat
    except Exception:
        return c_flat


def _generate_lattice_init():
    """
    Generate an initial configuration based on a BCC (Body-Centered Cubic) lattice.
    BCC is denser than simple cubic and provides a good starting point.
    """
    # We need to fit 100 spheres.
    # A 5x5x5 simple cubic grid has 125 points.
    # Let's use a 5x5x5 grid of BCC unit cells? No, that's too many.
    # Let's try to pack 100 spheres in a 5x5x4 arrangement or similar.
    # Actually, a simpler approach: start with a 5x5x5 grid (125 points), 
    # remove 25 points randomly or keep the first 100? 
    # Better: Use a 4x5x5 grid (100 points exactly).
    
    nx, ny, nz = 4, 5, 5
    total = nx * ny * nz
    if total < N:
        # Fallback to 5x5x4 if needed, but 4*5*5=100 is exact.
        pass
        
    points = []
    # Scale the lattice to fit inside [-0.5, 0.5] with some margin
    # Lattice constant a. Side length = (n-1)*a. We want side length < 1.
    # Let's set a = 1.0 / max(nx, ny, nz) * 0.9 to leave room for expansion.
    a = 1.0 / max(nx, ny, nz) * 0.95
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = (i - (nx - 1) / 2.0) * a
                y = (j - (ny - 1) / 2.0) * a
                z = (k - (nz - 1) / 2.0) * a
                points.append([x, y, z])
                
    points = np.array(points[:N])
    # Add small random noise to break symmetry
    points += np.random.randn(N, 3) * 0.01
    return points.ravel()


def construct():
    np.random.seed(42)
    
    swarm_size = 20
    max_iter = 200
    w_start = 0.9
    w_end = 0.4
    c1 = 1.5
    c2 = 1.5
    
    # Initialize swarm
    # Particle 0: Lattice-based
    # Particles 1-4: Perturbed Lattice
    # Particles 5+: Random
    
    particles = np.zeros((swarm_size, N * 3))
    
    # Base lattice
    base_lattice = _generate_lattice_init()
    
    for i in range(swarm_size):
        if i == 0:
            particles[i] = base_lattice
        elif i < 5:
            # Perturbed lattice
            particles[i] = base_lattice + np.random.randn(N * 3) * 0.05
        else:
            # Random
            particles[i] = np.random.uniform(-0.4, 0.4, N * 3)
            
    # Evaluate initial positions and apply local refinement
    personal_bests = np.zeros_like(particles)
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        # Refine initial position
        refined = _local_refine(particles[i], max_iter=5)
        particles[i] = refined
        val = _objective(particles[i])
        personal_best_vals[i] = val
        personal_bests[i] = particles[i].copy()
        
    global_best_idx = np.argmin(personal_best_vals)
    global_best = personal_bests[global_best_idx].copy()
    global_best_val = personal_best_vals[global_best_idx]
    
    velocities = np.zeros_like(particles)
    
    for it in range(max_iter):
        # Adaptive inertia
        w = w_start - (w_start - w_end) * (it / max_iter)
        
        for i in range(swarm_size):
            r1 = np.random.rand(N * 3)
            r2 = np.random.rand(N * 3)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_bests[i] - particles[i]) + 
                           c2 * r2 * (global_best - particles[i]))
            
            particles[i] += velocities[i]
            
            # Clip to bounds
            particles[i] = np.clip(particles[i], -0.5, 0.5)
            
            # Local refinement step (crucial for performance in high D)
            # We reduce iterations as we get closer to the end to save time
            iter_limit = 5 if it < max_iter // 2 else 3
            refined = _local_refine(particles[i], max_iter=iter_limit)
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
    c_best = global_best.reshape(N, 3)
    # Use more iterations for the final polish
    result = minimize(_objective, global_best, method='L-BFGS-B', 
                      bounds=[(-0.5, 0.5)] * (N * 3), 
                      options={'maxiter': 100, 'ftol': 1e-10})
    
    c_final = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
    
    return c_final
# EVOLVE-BLOCK-END
