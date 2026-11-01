"""Program for packing 200 equal circles in a unit square using Simulated Annealing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Initialize with a jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    # Simulated Annealing Parameters
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.999
    max_iter = 10000
    step_size = 0.05
    
    def compute_energy(c, power=4):
        """Compute potential energy based on inverse distances."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise repulsion
        # Avoid division by zero
        dists = np.maximum(dists, 1e-8)
        pair_energy = np.sum(1.0 / (dists ** power))
        
        # Boundary repulsion
        # Distance to boundaries
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        d_bound = np.minimum(dx, dy)
        d_bound = np.maximum(d_bound, 1e-8)
        bound_energy = np.sum(1.0 / (d_bound ** power))
        
        return pair_energy + 1.0 * bound_energy

    current_energy = compute_energy(centers)
    best_centers = centers.copy()
    best_energy = current_energy
    
    for i in range(max_iter):
        # Propose a move: jitter all points
        noise = rng.normal(0, step_size, size=centers.shape)
        new_centers = centers + noise
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_energy = compute_energy(new_centers)
        
        # Acceptance criterion
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / T):
            centers = new_centers
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_centers = centers.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Refine with a local search (gradient-free) to maximize min distance
    # Use a simple jitter-and-optimize loop on the minimum distance
    r = 0.0
    for i in range(1000):
        c = best_centers
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        min_bound = np.min(np.minimum(dx, dy))
        
        current_r = 0.5 * min(min_pair, min_bound)
        if current_r > r:
            r = current_r
            best_centers = c
        
        # Small jitter
        best_centers += rng.normal(0, 1e-4, size=best_centers.shape)
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
    return best_centers
# EVOLVE-BLOCK-END
