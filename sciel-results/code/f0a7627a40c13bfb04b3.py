"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin for 120s limit
    
    # Initial hexagonal lattice with random perturbation to break symmetry
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            x0 = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    x0 = x0 + np.random.randn(*x0.shape) * 0.005
    
    def score(x):
        """Compute the minimum radius for given centers."""
        x = x.reshape(-1, 2)
        # Distance to boundary
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        return min(min_pairwise / 2.0, np.min(dist_boundary))
    
    best_x = x0.copy()
    best_score = score(best_x)
    
    # Physics-based simulation: repulsive forces between circles and from container wall
    def physics_refine(x, max_iters=2000, dt=0.001, cooling_factor=0.999):
        """Refine packing using physics-based simulation with repulsive forces."""
        x = x.copy().reshape(-1, 2)
        current_dt = dt
        current_cooling = cooling_factor
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            n = len(x)
            forces = np.zeros_like(x)
            
            # Compute pairwise repulsive forces
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Only consider close pairs for efficiency
            close_mask = dists < 0.3
            for i in range(n):
                for j in range(i + 1, n):
                    if close_mask[i, j]:
                        d = dists[i, j]
                        if d < 1e-10:
                            # Random small force if coincident
                            forces[i] += np.random.randn(2) * 0.01
                            forces[j] -= np.random.randn(2) * 0.01
                        else:
                            # Repulsive force: stronger when closer
                            force_magnitude = 1.0 / (d * d + 1e-6)
                            direction = diff[i, j] / d
                            forces[i] += force_magnitude * direction
                            forces[j] -= force_magnitude * direction
            
            # Container wall repulsion
            norms = np.linalg.norm(x, axis=1)
            wall_dist = 1.0 - norms
            for i in range(n):
                if wall_dist[i] < 0.1:
                    if norms[i] > 1e-10:
                        direction = x[i] / norms[i]
                        # Push inward if too close to wall
                        forces[i] -= 5.0 * direction * (1.0 - wall_dist[i] / 0.1)
            
            # Apply forces with velocity damping
            x += forces * current_dt
            
            # Ensure points stay inside container
            norms = np.linalg.norm(x, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                x[mask] *= (0.999 / norms[mask])[:, np.newaxis]
            
            # Cool down
            current_dt *= current_cooling
            if current_dt < 1e-5:
                break
        
        return x
    
    # First pass: physics simulation to settle into a good configuration
    x_physics = physics_refine(x0, max_iters=3000, dt=0.002, cooling_factor=0.9995)
    if score(x_physics) > best_score:
        best_x = x_physics
        best_score = score(best_x)
    
    # Second pass: local optimization with smooth surrogate
    def soft_min_score(x, beta=100.0):
        """Smooth surrogate for the min radius using log-sum-exp."""
        x = x.reshape(-1, 2)
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pairwise_radii = dists / 2.0
        
        # Combine all constraints
        all_radii = np.concatenate([pairwise_radii.ravel(), dist_boundary])
        # Use soft-min: -log-sum-exp(-beta * radii) / beta
        max_r = np.max(all_radii)
        soft_min_val = max_r - np.log(np.sum(np.exp(-beta * (all_radii - max_r)))) / beta
        return soft_min_val
    
    def neg_soft_score(x, beta=100.0):
        return -soft_min_score(x, beta)
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Multiple optimization passes with increasing beta
    betas = [50.0, 100.0, 200.0, 500.0]
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        try:
            result = minimize(lambda x: neg_soft_score(x, beta), best_x, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if score(result.x) > best_score:
                best_x = result.x.copy()
                best_score = score(best_x)
        except Exception:
            continue
    
    # Final polish with exact score
    if time.time() < deadline - 5:
        def neg_score(x):
            return -score(x)
        
        try:
            result = minimize(neg_score, best_x, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < -best_score:
                best_x = result.x.copy()
                best_score = -result.fun
        except Exception:
            pass
    
    # Final validation: ensure all points are inside the unit circle
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_x[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
    
    return best_x
# EVOLVE-BLOCK-END
