# EVOLVE-BLOCK-START
import numpy as np
import math
import time

N = 150

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    # Vectorized distance calculation
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    
    # We only need upper triangle
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = np.sqrt(d2[mask])
    
    # Filter out zeros to avoid log(0)
    d = d[d > 1e-15]
    if len(d) == 0:
        return 1e10
        
    terms = np.log(1.0 / d)
    return math.fsum(terms)

def _project_to_sphere(points):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _fibonacci_sphere(n):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    # Time limit management
    start_time = time.time()
    time_limit = 115.0 # Leave buffer
    
    best_points = None
    best_energy = float('inf')
    
    # Initial guess
    init_points = _fibonacci_sphere(N)
    
    np.random.seed(42)
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 1e-4
    cooling_rate = 0.999 # Adjust based on iterations
    
    # We will run multiple SA runs
    # Estimate iterations per run based on time
    # One SA step: ~ O(N^2) energy calc + O(N) update.
    # Let's test how fast _log_energy is.
    
    # Quick benchmark
    _ = _log_energy(init_points)
    bench_time = time.time() - start_time
    if bench_time > 0:
        steps_per_sec = 1.0 / bench_time
    else:
        steps_per_sec = 1000.0
        
    # We want to use ~90% of time for optimization
    # Let's do 3 restarts
    num_restarts = 3
    time_per_restart = (time_limit - start_time) / num_restarts
    
    for r in range(num_restarts):
        if r == 0:
            current = init_points.copy()
        else:
            current = init_points + np.random.normal(0, 0.1, size=(N, 3))
            current = _project_to_sphere(current)
            
        current_energy = _log_energy(current)
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = current.copy()
        
        T = T_start
        step_count = 0
        restart_start = time.time()
        
        while time.time() - restart_start < time_per_restart:
            # Pick a random point to move
            i = np.random.randint(N)
            
            # Generate a random tangent vector
            # Random vector in R3
            rand_vec = np.random.normal(0, 1, size=3)
            # Project to tangent plane at current[i]
            tangent_vec = rand_vec - np.dot(rand_vec, current[i]) * current[i]
            norm_t = np.linalg.norm(tangent_vec)
            if norm_t < 1e-10:
                continue
            tangent_vec = tangent_vec / norm_t
            
            # Step size proportional to temperature
            step_size = 0.1 * T ** 0.5
            new_point = current[i] + step_size * tangent_vec
            
            # Project to sphere
            new_point = new_point / np.linalg.norm(new_point)
            
            # Calculate new energy
            # Optimization: Only calculate change in energy involving point i
            # E_new = E_old - sum_{j!=i} log(1/|x_i - x_j|) + sum_{j!=i} log(1/|x_new - x_j|)
            
            # Distances from old point i to all others
            diff_old = current - current[i]
            d_old = np.linalg.norm(diff_old, axis=1)
            # Exclude self
            d_old[i] = 1.0 # Avoid log(0) or division by zero, will be excluded from sum anyway if we mask
            mask_old = np.ones(N, dtype=bool)
            mask_old[i] = False
            d_old = d_old[mask_old]
            d_old = d_old[d_old > 1e-15]
            old_terms = np.log(1.0 / d_old) if len(d_old) > 0 else np.array([])
            
            # Distances from new point to all others
            diff_new = current - new_point
            d_new = np.linalg.norm(diff_new, axis=1)
            d_new[i] = 1.0
            mask_new = np.ones(N, dtype=bool)
            mask_new[i] = False
            d_new = d_new[mask_new]
            d_new = d_new[d_new > 1e-15]
            new_terms = np.log(1.0 / d_new) if len(d_new) > 0 else np.array([])
            
            delta_e = math.fsum(new_terms) - math.fsum(old_terms)
            
            # Metropolis criterion
            if delta_e < 0 or np.random.rand() < math.exp(-delta_e / T):
                current[i] = new_point
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = current.copy()
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
            
            step_count += 1
            
    return best_points
# EVOLVE-BLOCK-END
