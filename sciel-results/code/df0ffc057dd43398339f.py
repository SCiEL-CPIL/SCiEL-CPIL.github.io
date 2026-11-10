import numpy as np
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    # Precompute indices for all unique triples (i, j, k) with i < j < k
    idx1, idx2, idx3 = np.indices((n, n, n)).reshape(3, -1)
    mask = (idx1 < idx2) & (idx2 < idx3)
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    idx3 = idx3[mask]
    
    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
        except Exception:
            return 0.0
        if a_hull < 1e-10:
            return 0.0
        
        # Vectorized area calculation
        p1 = pts[idx1]
        p2 = pts[idx2]
        p3 = pts[idx3]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        min_area = np.min(areas)
        if min_area < 1e-10:
            return 0.0
        return min_area / a_hull

    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(seed=42)
    
    # Multiple restarts with Simulated Annealing
    num_restarts = 10
    max_iter = 2000
    T_start = 0.5
    T_end = 0.001
    cooling = (T_end / T_start) ** (1.0 / max_iter)
    
    for restart in range(num_restarts):
        if restart == 0:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        elif restart == 1:
            # Slightly perturbed regular polygon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.05, 0.05, n)
            radii = 1.0 + rng.uniform(-0.05, 0.05, n)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Random points on a circle with some noise
            angles = rng.uniform(0, 2 * np.pi, n)
            radii = rng.uniform(0.5, 1.0, n)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
        current_score = get_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        T = T_start
        for _ in range(max_iter):
            T *= cooling
            
            # Pick a random point to move
            idx = rng.integers(n)
            orig = points[idx].copy()
            
            # Gaussian perturbation, scale with temperature
            step = rng.normal(0, T, 2)
            points[idx] += step
            
            new_score = get_score(points)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # Accept with probability exp((new - current) / T)
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = points.copy()
                    else:
                        points[idx] = orig
                else:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
                        
    return best_points
