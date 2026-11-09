"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    min_dist_to_wall = np.min(0.5 - np.abs(centers), axis=(0, 1))
    
    # Minimum pairwise distance / 2
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0)  # query pairs within distance 1
    if len(pairs) == 0:
        # Fallback: compute all pairs
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                min_pair_dist = min(min_pair_dist, d)
        min_pair_dist = min_pair_dist
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            min_pair_dist = min(min_pair_dist, d)
    
    return min(min_dist_to_wall, min_pair_dist / 2.0)


def _initial_cubic_grid():
    """Create initial cubic grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_initial():
    """Create random initial placement."""
    return np.random.uniform(-0.5, 0.5, (N, 3))


def _local_optimization(centers, max_iters=2000, lr=0.01, cooling=0.999):
    """Local optimization using gradient-free approach with repulsive forces."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Use a simple force-based approach
    current_radius = best_radius
    step_size = 0.05
    
    for it in range(max_iters):
        if it % 100 == 0 and time.time() > 140:
            break
            
        # Compute forces: repulsion between spheres and attraction to center
        forces = np.zeros_like(centers)
        
        # Repulsive forces between pairs
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.uniform(-0.01, 0.01, 3)
                    dist = np.linalg.norm(diff)
                # Force proportional to 1/dist^2 for close spheres
                force_mag = 1.0 / (dist * dist + 1e-6)
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
        
        # Wall repulsion
        for i in range(3):
            # Attract away from walls
            forces[:, i] += (0.5 - np.abs(centers[:, i])) * 0.1 * np.sign(centers[:, i])
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        # Update positions
        new_centers = centers + step_size * forces
        # Clamp to cube
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_radius:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Reduce step size on failure
            step_size *= 0.95
        
        if step_size < 1e-6:
            break
    
    return best_centers, best_radius


def _simulated_annealing(centers, max_iters=5000, T0=0.05, T_end=0.001):
    """Simulated annealing optimization."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    
    T = T0
    cooling_rate = (T_end / T0) ** (1.0 / max_iters)
    
    for it in range(max_iters):
        if it % 50 == 0 and time.time() > 140:
            break
            
        # Random perturbation
        perturbation = np.random.normal(0, 0.02, (N, 3))
        new_centers = centers + perturbation
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        delta = new_radius - current_radius
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        
        T *= cooling_rate
    
    return best_centers, best_radius


def construct():
    """Construct 50 sphere centers in unit cube to maximize minimum radius."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Start with cubic grid
    best_centers = _initial_cubic_grid()
    best_radius = _compute_radius(best_centers)
    
    # Try local optimization from grid
    opt_centers, opt_radius = _local_optimization(best_centers, max_iters=3000)
    if opt_radius > best_radius:
        best_centers = opt_centers
        best_radius = opt_radius
    
    # Try simulated annealing from optimized solution
    sa_centers, sa_radius = _simulated_annealing(best_centers, max_iters=10000, T0=0.03, T_end=0.0005)
    if sa_radius > best_radius:
        best_centers = sa_centers
        best_radius = sa_radius
    
    # If time remains, try additional random restarts
    elapsed = time.time() - start_time
    if elapsed < 100:
        for restart in range(5):
            if time.time() > deadline:
                break
            random_centers = _random_initial()
            # Quick local optimization
            opt_c, opt_r = _local_optimization(random_centers, max_iters=1000)
            if opt_r > best_radius:
                best_centers = opt_c
                best_radius = opt_r
                # Refine with SA
                if time.time() < deadline - 10:
                    sa_c, sa_r = _simulated_annealing(best_centers, max_iters=5000, T0=0.02, T_end=0.0003)
                    if sa_r > best_radius:
                        best_centers = sa_c
                        best_radius = sa_r
    
    # Final verification
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        # Fallback to grid
        return _initial_cubic_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
