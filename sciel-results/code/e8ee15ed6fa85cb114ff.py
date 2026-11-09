import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    def get_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def random_sphere_point():
        # Uniform random point on unit sphere
        v = np.random.normal(size=3)
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.array([1.0, 0.0, 0.0])
        return v / norm

    def perturb(points, idx, step_size):
        # Perturb point idx by adding a random vector and renormalizing
        # This moves the point on the sphere
        p = points[idx]
        noise = np.random.normal(size=3) * step_size
        new_p = p + noise
        norm = np.linalg.norm(new_p)
        if norm == 0:
            return points
        new_p = new_p / norm
        new_points = points.copy()
        new_points[idx] = new_p
        return new_points

    best_points = None
    best_score = -1.0
    
    # Number of restarts
    n_restarts = 10
    
    for r in range(n_restarts):
        # Initialize with Fibonacci sphere or random
        # Fibonacci sphere is a good start
        phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta = np.pi * (1 + 5**0.5) * np.arange(n)
        points = np.column_stack((
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ))
        
        # Add some randomness to break symmetry if needed, 
        # but Fibonacci is usually distinct.
        # Let's add a small random rotation or noise to avoid identical starts
        if r > 0:
            # Random rotation
            # Or just add small noise and renormalize
            points = points + np.random.normal(0, 0.1, size=(n, 3))
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
        current_score = get_score(points)
        
        # Simulated Annealing parameters
        T_start = 0.01
        T_end = 1e-6
        steps = 2000
        step_size = 0.1
        
        for i in range(steps):
            # Cool down
            T = T_start * (T_end / T_start) ** (i / steps)
            
            # Pick a random point to perturb
            idx = np.random.randint(n)
            
            # Perturb
            new_points = perturb(points, idx, step_size)
            new_score = get_score(new_points)
            
            # Calculate delta in energy (we want to maximize score, so energy = -score)
            delta_e = -new_score - (-current_score) # = current_score - new_score
            # If new_score > current_score, delta_e < 0 (improvement)
            
            accept = False
            if delta_e < 0:
                accept = True
            else:
                # Boltzmann probability
                if T > 0:
                    prob = np.exp(-delta_e / T)
                    if np.random.random() < prob:
                        accept = True
            
            if accept:
                points = new_points
                current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
    return best_points
