import numpy as np

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Perturb angles to break symmetry at close range, preserve at far range
    delta = 0.03
    eps = np.zeros(n)
    for i in range(n):
        eps[i] = delta * np.sin(2 * np.pi * i / n)
            
    angles = angles + eps
    
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Ensure strictly convex (should be with small delta)
    return polygon
