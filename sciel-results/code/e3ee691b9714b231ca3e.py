"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def construct():
    """Construct a good packing of 200 equal circles in the unit square [-1/2, 1/2]^2."""
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    def compute_radius(centers):
        """Compute the maximum common radius for a given set of centers."""
        n = len(centers)
        if n < 2:
            return 0.5
        # Wall distances
        wall = np.min(np.stack([
            0.5 - centers[:, 0],
            centers[:, 0] + 0.5,
            0.5 - centers[:, 1],
            centers[:, 1] + 0.5
        ]), axis=0)
        wall_r = np.min(wall)
        
        # Pairwise distances using pdist
        if n > 1:
            dists = pdist(centers)
            min_pair = np.min(dists) / 2.0
        else:
            min_pair = wall_r
            
        return min(wall_r, min_pair)
    
    def generate_hex_initial(n):
        """Generate a good hexagonal initial configuration for n circles."""
        best_centers = None
        best_r = 0
        
        for rows in range(10, 25):
            base_cols = n // rows
            extra = n % rows
            
            centers = []
            for i in range(rows):
                cols = base_cols + (1 if i < extra else 0)
                if cols == 0:
                    continue
                for j in range(cols):
                    x = (j - (cols - 1) / 2) * 0.1
                    if i % 2 == 1:
                        x += 0.05
                    y = (i - (rows - 1) / 2) * 0.15
                    centers.append([x, y])
            
            centers = np.array(centers)
            if len(centers) != n:
                continue
            
            r = compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        
        return best_centers if best_centers is not None else None
    
    def objective(x):
        """Objective function: negative of the minimum radius (to minimize)."""
        centers = x.reshape(-1, 2)
        
        # Wall distances
        wall = np.min(np.stack([
            0.5 - centers[:, 0],
            centers[:, 0] + 0.5,
            0.5 - centers[:, 1],
            centers[:, 1] + 0.5
        ]), axis=0)
        wall_r = np.min(wall)
        
        # Pairwise distances
        if len(centers) > 1:
            dists = pdist(centers)
            min_pair = np.min(dists) / 2.0
        else:
            min_pair = wall_r
            
        r = min(wall_r, min_pair)
        return -r
    
    def objective_with_grad(x):
        """Objective function with gradient approximation."""
        centers = x.reshape(-1, 2)
        
        # Wall distances
        wall = np.min(np.stack([
            0.5 - centers[:, 0],
            centers[:, 0] + 0.5,
            0.5 - centers[:, 1],
            centers[:, 1] + 0.5
        ]), axis=0)
        wall_r = np.min(wall)
        
        # Pairwise distances
        if len(centers) > 1:
            dists = pdist(centers)
            min_pair = np.min(dists) / 2.0
        else:
            min_pair = wall_r
            
        r = min(wall_r, min_pair)
        
        # Compute gradient numerically (sparse)
        grad = np.zeros_like(x)
        eps = 1e-5
        
        # Only compute gradient for the closest pairs to save time
        # Find the minimum distance pair
        if len(centers) > 1:
            dists_full = pdist(centers)
            min_idx = np.argmin(dists_full)
            
            # Convert from condensed index to (i, j)
            n = len(centers)
            # Find i, j from condensed index
            i = 0
            remaining = min_idx
            while remaining >= n - 1 - i:
                remaining -= (n - 1 - i)
                i += 1
            j = i + 1 + remaining
            
            # Gradient for pair (i, j)
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1e-10:
                grad[2*i] = -dx / dist
                grad[2*i+1] = -dy / dist
                grad[2*j] = dx / dist
                grad[2*j+1] = dy / dist
        
        # Wall gradient for the closest wall
        min_wall_idx = np.argmin(wall)
        # wall[0] = 0.5 - x_i for some i, wall[1] = x_i + 0.5, etc.
        # Determine which center and which wall
        # This is complex; skip wall gradient for simplicity
        
        return -r, grad
    
    def l_bfgs_optimize(centers, max_iter=500, deadline=None):
        """Optimize using L-BFGS-B."""
        x0 = centers.flatten()
        bounds = [(-0.499, 0.499)] * 2 * len(centers)
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 2}
        )
        
        if result.success or result.fun < objective(x0):
            return result.x.reshape(-1, 2), compute_radius(result.x.reshape(-1, 2))
        
        return centers, compute_radius(centers)
    
    def multi_start_l_bfgs(initial, n_restarts=5, deadline=None):
        """Multiple restarts with L-BFGS-B."""
        best_centers = initial.copy()
        best_r = compute_radius(best_centers)
        
        for restart in range(n_restarts):
            if deadline is not None and time.time() > deadline:
                break
            
            # Add small random perturbation
            perturbation = np.random.normal(0, 0.005, size=initial.shape)
            perturbed = np.clip(initial + perturbation, -0.499, 0.499)
            
            centers, r = l_bfgs_optimize(perturbed, max_iter=100, deadline=deadline)
            
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
            
            # Also try optimizing the current best
            centers, r = l_bfgs_optimize(best_centers, max_iter=100, deadline=deadline)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        
        return best_centers, best_r
    
    # Main algorithm
    # Generate initial hexagonal configuration
    initial = generate_hex_initial(N)
    if initial is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        initial = grid[:N]
    
    best_centers = initial.copy()
    best_r = compute_radius(best_centers)
    
    # Phase 1: Single L-BFGS optimization from initial
    if time.time() < deadline - 20:
        centers, r = l_bfgs_optimize(initial, max_iter=200, deadline=deadline)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 2: Multi-start L-BFGS
    if time.time() < deadline - 10:
        centers, r = multi_start_l_bfgs(best_centers, n_restarts=3, deadline=deadline)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 3: Final refinement
    if time.time() < deadline - 5:
        centers, r = l_bfgs_optimize(best_centers, max_iter=200, deadline=deadline)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
