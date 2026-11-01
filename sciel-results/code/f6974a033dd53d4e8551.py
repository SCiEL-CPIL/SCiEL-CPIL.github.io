import numpy as np
import time
from scipy.spatial import cKDTree

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    def get_hex_initialization():
        # Generate a hexagonal lattice
        rows_counts = [8, 7, 8, 7, 8, 7, 8]
        y_step = np.sqrt(3) / 2.0
        x_step = 1.0
        
        points = []
        for i, count in enumerate(rows_counts):
            y = i * y_step
            start_x = -(count - 1) * x_step / 2.0
            for j in range(count):
                x = start_x + j * x_step
                points.append([x, y])
        
        pts = np.array(points)
        # We have 53 points. Keep the 50 closest to the center.
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
        
        # Scale to fit in [-0.5, 0.5]
        max_x = np.max(np.abs(pts[:, 0]))
        max_y = np.max(np.abs(pts[:, 1]))
        scale = 0.5 / max(max_x, max_y)
        pts *= scale
        
        # Center the points
        # The hex grid is already centered around x=0, but y starts at 0.
        # Shift y to center it.
        y_center = np.mean(pts[:, 1])
        pts[:, 1] -= y_center
        
        return pts

    def calculate_score(pts):
        # pts shape (N, 2)
        d_left = 0.5 + pts[:, 0]
        d_right = 0.5 - pts[:, 0]
        d_bottom = 0.5 + pts[:, 1]
        d_top = 0.5 - pts[:, 1]
        
        min_boundary = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
        
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        r_boundary = min_boundary
        r_pair = min_pair_dist / 2.0
        
        return min(r_boundary, r_pair)

    best_pts = get_hex_initialization()
    best_score = calculate_score(best_pts)
    current_pts = best_pts.copy()
    current_score = best_score
    
    T = 0.01
    T_min = 0.00001
    cooling_rate = 0.99995
    step_size = 0.01
    
    # Track best score
    while time.time() < deadline:
        # Perturb
        i = np.random.randint(N)
        old_x, old_y = current_pts[i]
        
        new_x = old_x + np.random.normal(0, step_size)
        new_y = old_y + np.random.normal(0, step_size)
        
        current_pts[i] = [new_x, new_y]
        
        new_score = calculate_score(current_pts)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
        else:
            prob = np.exp((new_score - current_score) / T)
            if np.random.random() < prob:
                current_score = new_score
            else:
                current_pts[i] = [old_x, old_y]
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            if step_size > 0.0005:
                step_size *= 0.98
            else:
                # Restart
                if np.random.random() < 0.05:
                    T = 0.005
                    step_size = 0.005
                    current_pts = best_pts.copy()
                    current_score = best_score
    
    return best_pts
