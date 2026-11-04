import numpy as np
import time
from scipy.spatial import ConvexHull

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_score(points):
        # points: (N, 2)
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # In 2D, volume is area
            if hull_area <= 0:
                return 0.0
        except Exception:
            return 0.0
        
        # Calculate all triangle areas
        # Pre-compute indices for efficiency
        # We can just loop or vectorize. Vectorizing is faster.
        # N=14, C(14,3)=364.
        
        # Generate indices
        # i, j, k with i < j < k
        idx = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
        
        p0 = points[idx[:, 0]]
        p1 = points[idx[:, 1]]
        p2 = points[idx[:, 2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area

    # Initialize with regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = get_score(best_points)
    
    # Local optimization
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Perturb a random point
        idx = np.random.randint(0, N)
        original = best_points[idx].copy()
        
        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        best_points[idx][0] += dx
        best_points[idx][1] += dy
        
        new_score = get_score(best_points)
        
        if new_score > best_score:
            best_score = new_score
            no_improve_count = 0
        else:
            # Revert
            best_points[idx] = original
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Reset step size and try a larger jump
                step_size = 0.2
                no_improve_count = 0
                # Perform a large jump
                idx = np.random.randint(0, N)
                original = best_points[idx].copy()
                angle = np.random.uniform(0, 2 * np.pi)
                dx = 1.0 * np.cos(angle)
                dy = 1.0 * np.sin(angle)
                best_points[idx][0] += dx
                best_points[idx][1] += dy
                new_score = get_score(best_points)
                if new_score > best_score:
                    best_score = new_score
                    step_size = 0.05
                else:
                    best_points[idx] = original
                    step_size = 0.05
                continue
        
        # Decay step size slowly
        step_size *= 0.995
        if step_size < 1e-6:
            step_size = 1e-6

    return best_points
