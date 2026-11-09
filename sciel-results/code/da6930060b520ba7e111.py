import numpy as np
import scipy.spatial
import time

N = 14

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    # Pre-compute indices for triangles
    idx_i, idx_j, idx_k = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    tri_i = idx_i[mask]
    tri_j = idx_j[mask]
    tri_k = idx_k[mask]
    
    def get_score(pts):
        # Vectorized triangle areas
        v1 = pts[tri_j] - pts[tri_i]
        v2 = pts[tri_k] - pts[tri_i]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        # Hull area
        try:
            hull = scipy.spatial.ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 0.0
        if hull_area <= 1e-10:
            return 0.0
            
        return min_area / hull_area

    def simulate_annealing(init_points, t_start, t_end, steps):
        current_points = init_points.copy()
        current_score = get_score(current_points)
        best_points = current_points.copy()
        best_score = current_score
        
        # Adaptive step size? Or fixed?
        # Let's use a step size that scales with temperature to explore broadly at first.
        base_step = 0.2
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            # Linear cooling or exponential? Exponential is standard.
            t = t_start * (t_end / t_start) ** (step / steps)
            
            # Perturb a random point
            idx = np.random.randint(N)
            step_size = base_step * (t / t_start) + 0.01
            perturbation = np.random.normal(0, step_size, 2)
            new_points = current_points.copy()
            new_points[idx] += perturbation
            
            new_score = get_score(new_points)
            
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / max(t, 1e-9)):
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    
        return best_points, best_score

    # Initialize with Regular Polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_points = np.vstack([np.cos(angles), np.sin(angles)]).T
    
    # Run SA
    # 20000 iterations should be fast enough
    best_points, best_score = simulate_annealing(init_points, 1.0, 0.001, 20000)
    
    # Try a second run with a different seed/perturbation if time remains
    if time.time() - start_time < time_limit - 20:
        np.random.seed(42) # Reproducibility for second run? Or just new random
        perturbed_init = init_points + np.random.normal(0, 0.2, (N, 2))
        p2, s2 = simulate_annealing(perturbed_init, 1.0, 0.001, 10000)
        if s2 > best_score:
            best_points = p2
            best_score = s2

    return best_points
