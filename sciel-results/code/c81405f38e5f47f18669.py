"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _repulsive_energy(points, power=2.0):
    """
    Calculate the repulsive energy of the configuration.
    We want to maximize distances, so we minimize the sum of inverse powers of distances.
    """
    points = _normalize(points)
    # Compute pairwise distances squared
    # dist_sq[i, j] = |p_i - p_j|^2 = 2 - 2 * p_i . p_j
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero on diagonal
    
    # Inverse power law
    # We sum over i < j
    energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = dist_sq[i, j]
            if d > 1e-10:
                energy += 1.0 / (d ** (power / 2.0))
    return energy


def _sa_optimize(points, T_start=1.0, T_end=1e-4, n_iter=20000):
    """
    Simulated Annealing to minimize repulsive energy.
    """
    points = _normalize(points).copy()
    current_energy = _repulsive_energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Cooling schedule
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        T = T_start * (cooling ** i)
        
        # Select a random point to move
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Generate a new position by adding random noise and normalizing
        # Step size decreases as T decreases
        step_size = 0.1 * T / T_start
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        # Update points
        points[idx] = new_point
        
        new_energy = _repulsive_energy(points)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            # Reject move
            points[idx] = original_point
            
    return best_points


def _local_optimize(points):
    """
    Final local optimization using L-BFGS-B on the repulsive energy.
    We optimize the 3N coordinates, but we must ensure they stay on the sphere.
    A common trick is to optimize in R^3 and normalize inside the objective, 
    but this creates a non-smooth landscape at the origin.
    Alternatively, we can use the fact that the gradient of the normalized energy 
    can be computed.
    
    Simpler: Use L-BFGS-B on the flat array, and inside the objective, normalize.
    This works reasonably well for local polishing.
    """
    def objective(params):
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _repulsive_energy(pts)
        
    def gradient(params):
        # Numerical gradient is slow, but L-BFGS-B can estimate it.
        # For better performance, we'd implement analytic gradient.
        # Given time constraints, let's let scipy estimate or use a small number of iterations.
        # Actually, let's just use the objective and let L-BFGS-B estimate gradients.
        # This is slower but robust enough for a final polish on 150 variables.
        pass
        
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
    
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # 1. Initialize with Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Simulated Annealing
    # Set random seed for reproducibility? No, task doesn't specify, but it's good practice.
    # However, the evaluator might run multiple times. We'll just use system randomness.
    points_sa = _sa_optimize(x0, T_start=0.5, T_end=1e-5, n_iter=10000)
    
    # 3. Local Polish
    points_final = _local_optimize(points_sa)
    
    # 4. Final check and normalization
    points_final = _normalize(points_final)
    
    return points_final
# EVOLVE-BLOCK-END
