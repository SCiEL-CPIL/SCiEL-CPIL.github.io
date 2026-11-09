import numpy as np

def _generate_es_points(m):
    """
    Recursively generate 2^(m-1) points in the Erdos-Szekeres construction.
    """
    if m == 1:
        return np.array([[0.5, 0.5]])
    
    points_prev = _generate_es_points(m - 1)
    
    # Set 1: Scale to [0, 0.4]x[0, 0.4]
    set1 = points_prev * 0.4
    
    # Set 2: Scale to [0.6, 1.0]x[0.6, 1.0] and reflect y to make a "cap"
    set2 = points_prev * 0.4 + np.array([0.6, 0.6])
    set2[:, 1] = 1.0 - set2[:, 1]
    
    points = np.vstack([set1, set2])
    return points

def erdos_szekeres_problem7() -> np.ndarray:
    # Generate 32 points (S_6)
    points = _generate_es_points(6)
    
    # Add 33rd point at the center
    center = np.array([0.5, 0.5])
    points = np.vstack([points, center])
    
    # Add small perturbations to ensure general position
    np.random.seed(42)
    perturbations = np.random.rand(33, 2) * 1e-6
    points += perturbations
    
    # Clip to [0, 1]
    points = np.clip(points, 0, 1)
    
    return points
