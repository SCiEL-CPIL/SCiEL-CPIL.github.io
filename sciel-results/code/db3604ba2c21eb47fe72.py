# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Simulated Annealing to explore the configuration space globally.
    """
    n = 14
    d = 3
    
    def normalize(points):
        """Center and normalize points to unit Frobenius norm."""
        points = points - np.mean(points, axis=0, keepdims=True)
        norm = np.linalg.norm(points)
        if norm > 1e-10:
            points = points / norm
        else:
            points = np.random.randn(n, d)
            points = points - np.mean(points, axis=0, keepdims=True)
            points = points / np.linalg.norm(points)
        return points

    def get_score(points):
        """Calculate the negative log ratio energy (lower is better)."""
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10 or d_min < 1e-10:
            return 1e10
        # We want to maximize d_min / d_max
        # Minimize -log(d_min/d_max) = log(d_max) - log(d_min)
        return np.log(d_max) - np.log(d_min)

    best_score = 1e10
    best_points = None
    n_restarts = 20
    n_iterations = 5000
    
    # Simulated Annealing parameters
    T_start = 0.5
    T_end = 1e-4
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    step_size = 0.1 # Initial step size for perturbation
    
    for r in range(n_restarts):
        np.random.seed(42 + r)
        
        # Initialize points randomly in a cube
        points = np.random.randn(n, d)
        points = normalize(points)
        
        current_energy = get_score(points)
        T = T_start
        
        for i in range(n_iterations):
            # Perturb a random point
            idx = np.random.randint(n)
            # Adaptive step size: scale with current temperature and typical distance
            # Typical distance is roughly 1/sqrt(n) for unit norm points? 
            # Let's just use a fixed base step size scaled by T
            move = np.random.randn(d) * (step_size * (T / T_start + 0.1))
            
            new_points = points.copy()
            new_points[idx] += move
            
            # Normalize the new configuration
            new_points = normalize(new_points)
            
            new_energy = get_score(new_points)
            
            # Acceptance criterion
            delta_E = new_energy - current_energy
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                points = new_points
                current_energy = new_energy
                
                if current_energy < best_score:
                    best_score = current_energy
                    best_points = points.copy()
            
            T *= cooling_rate

    if best_points is None:
        # Fallback
        np.random.seed(0)
        best_points = np.random.randn(n, d)
        best_points = normalize(best_points)

    return best_points

# EVOLVE-BLOCK-END
