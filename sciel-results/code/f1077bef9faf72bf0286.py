"""Optimized: Lattice section + target-area guided local search."""
import time
import numpy as np
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin: return by 55s, limit is 90s.

    # Precompute indices for all triples (i, j, k) with i < j < k
    indices = list(combinations(range(30), 3))
    idx_arr = np.array(indices, dtype=np.int64)
    I, J, K = idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]

    def fast_evaluate(points):
        """Compute the max multiplicity of any positive doubled area."""
        if len(points) != 30:
            return 0
        pts = np.array(points, dtype=np.int64)
        if len(set(points)) < 30:
            return 0
        x, y = pts[:, 0], pts[:, 1]
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        positive_areas = areas[areas > 0]
        if len(positive_areas) == 0:
            return 0
        max_area = int(positive_areas.max())
        if max_area > 10000:
            _, counts = np.unique(positive_areas, return_counts=True)
            return int(counts.max())
        counts = np.bincount(positive_areas.astype(np.int64))
        return int(counts.max())

    def count_target_area(points, target_A):
        """Count triples with |determinant| == target_A. Fast for a specific target."""
        if len(points) != 30:
            return 0
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        return int(np.sum(areas == target_A))

    # --- Phase 1: Lattice Section Initialization ---
    # Generate candidate sets by selecting 30 points from dense sublattices
    # and optimizing for a target area A.
    
    best_points = None
    best_score = 0
    best_target_A = 2  # Default target area
    
    # Dense sublattice bases and dimensions
    bases = [
        ((1, 0), (0, 1)),   # Square
        ((1, 0), (0, 2)),   # Stretched vertical
        ((2, 0), (0, 1)),   # Stretched horizontal
        ((1, 0), (1, 1)),   # Sheared
        ((1, 0), (2, 1)),   # Sheared more
        ((1, 1), (1, -1)),  # Diagonal
        ((1, 0), (1, 2)),   # Steep shear
        ((2, 1), (1, 0)),   # Another shear
        ((1, 2), (1, -1)),  # Mixed
        ((2, 0), (0, 2)),   # Coarse square
    ]
    
    dims = [(5, 6), (6, 5), (3, 10), (10, 3), (5, 5), (4, 8), (8, 4), (3, 10), (2, 15), (15, 2)]
    
    # Target areas to try (small even numbers are common on integer grids)
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    for u, v in bases:
        for M, N in dims:
            if time.time() > deadline - 15:
                break
                
            # Generate raw lattice points
            pts = []
            for i in range(M):
                for j in range(N):
                    px = i * u[0] + j * v[0]
                    py = i * u[1] + j * v[1]
                    pts.append((px, py))
            
            if len(pts) != 30 or len(set(pts)) < 30:
                continue
                
            # Center the points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            centered_pts = [(p[0] - cx, p[1] - cy) for p in pts]
            
            # Check bounds
            if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in centered_pts):
                continue
            
            # Evaluate full score
            score = fast_evaluate(centered_pts)
            if score > best_score:
                best_score = score
                best_points = list(centered_pts)
                # Determine the dominant area for this configuration
                pts_arr = np.array(centered_pts, dtype=np.int64)
                x, y = pts_arr[:, 0], pts_arr[:, 1]
                xi, yi = x[I], y[I]
                xj, yj = x[J], y[J]
                xk, yk = x[K], y[K]
                areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                positive_areas = areas[areas > 0]
                if len(positive_areas) > 0:
                    _, counts = np.unique(positive_areas, return_counts=True)
                    best_target_A = int(positive_areas[np.argmax(counts)])
    
    # Fallback if nothing found
    if best_points is None:
        pts = [(i, j) for i in range(5) for j in range(6)]
        cx, cy = 2, 2
        best_points = [(p[0]-cx, p[1]-cy) for p in pts]
        best_score = fast_evaluate(best_points)
        best_target_A = 2

    # --- Phase 2: Target-Area Guided Local Search ---
    # Use the dominant area as a guide for moves.
    current_points = list(best_points)
    current_score = best_score
    current_target_A = best_target_A
    random.seed(42)
    
    # Neighborhood definitions
    small_neighborhood = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]
    medium_neighborhood = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]
    large_neighborhood = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)]
    
    best_overall_points = list(current_points)
    best_overall_score = current_score
    
    max_restarts = 5
    restart = 0
    
    while time.time() < deadline - 3 and restart < max_restarts:
        no_improve_count = 0
        max_no_improve = 1500
        use_large = False
        
        while time.time() < deadline - 3:
            if no_improve_count > max_no_improve:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]
            
            # Adaptive neighborhood
            if no_improve_count > 500:
                use_large = True
            neighborhood = large_neighborhood if use_large else medium_neighborhood
            
            # Pick a random move
            dx, dy = random.choice(neighborhood)
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Check bounds
            if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                no_improve_count += 1
                continue
            
            new_point = (new_x, new_y)
            new_points = list(current_points)
            new_points[idx] = new_point
            
            if len(set(new_points)) < 30:
                no_improve_count += 1
                continue
            
            # Quick check: count target-area triangles first (cheaper than full histogram)
            target_count = count_target_area(new_points, current_target_A)
            
            # Only do full evaluation if target count is promising
            # Heuristic: if target count is high, likely the full score is good
            if target_count >= current_score - 5:
                score = fast_evaluate(new_points)
                if score > current_score:
                    current_score = score
                    current_points = new_points
                    no_improve_count = 0
                    # Update best overall
                    if current_score > best_overall_score:
                        best_overall_score = current_score
                        best_overall_points = list(current_points)
                        # Update target area to the new dominant area
                        pts_arr = np.array(current_points, dtype=np.int64)
                        x, y = pts_arr[:, 0], pts_arr[:, 1]
                        xi, yi = x[I], y[I]
                        xj, yj = x[J], y[J]
                        xk, yk = x[K], y[K]
                        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                        positive_areas = areas[areas > 0]
                        if len(positive_areas) > 0:
                            _, counts = np.unique(positive_areas, return_counts=True)
                            current_target_A = int(positive_areas[np.argmax(counts)])
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Restart with perturbation if stuck
        if no_improve_count > max_no_improve and restart < max_restarts - 1:
            restart += 1
            current_points = list(best_overall_points)
            current_score = best_overall_score
            # Apply random perturbations
            for _ in range(5):
                idx = random.randint(0, 29)
                old_x, old_y = current_points[idx]
                dx, dy = random.choice(large_neighborhood)
                new_x = old_x + dx
                new_y = old_y + dy
                if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                    new_point = (new_x, new_y)
                    new_pts = list(current_points)
                    new_pts[idx] = new_point
                    if len(set(new_pts)) == 30:
                        current_points = new_pts

    return best_overall_points
