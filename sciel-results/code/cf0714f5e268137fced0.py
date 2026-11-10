import time
import random
import numpy as np
from collections import Counter

_START = time.time()
_DEADLINE = 78.0  # safety margin before 90s kill

def _det(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def _score_np(pts_array):
    """Vectorized score: largest multiplicity of a single positive doubled area."""
    n = pts_array.shape[0]
    if n < 3:
        return 0
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx = pts_array[j, 0] - pts_array[i, 0]
            dy = pts_array[j, 1] - pts_array[i, 1]
            dets = np.abs(dx * (pts_array[j+1:, 1] - pts_array[i, 1]) - 
                          dy * (pts_array[j+1:, 0] - pts_array[i, 0]))
            pos = dets[dets > 0]
            if len(pos) > 0:
                areas.append(pos)
    if not areas:
        return 0
    all_areas = np.concatenate(areas)
    if len(all_areas) == 0:
        return 0
    vals, counts = np.unique(all_areas, return_counts=True)
    return int(counts.max())

def _score(points):
    arr = np.array(points, dtype=np.int32)
    return _score_np(arr)

def _generate_grid(w, h, offset_x=0, offset_y=0):
    """Generate a w x h grid of points."""
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append((x + offset_x, y + offset_y))
    return pts

def construct():
    global _START
    start = time.time()
    deadline = start + 78.0
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Structured Grids
    # The 2x15 grid is expected to be the best performer.
    # We try various orientations and shifts.
    
    grid_configs = [
        (2, 15), (15, 2),
        (3, 10), (10, 3),
        (5, 6), (6, 5),
        (1, 30), (30, 1)
    ]
    
    for w, h in grid_configs:
        if time.time() > deadline - 5:
            break
        # Try a few shifts to ensure we are within bounds and centered
        for ox in range(-30, 31, 5):
            for oy in range(-30, 31, 5):
                pts = _generate_grid(w, h, ox, oy)
                # Check bounds
                if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                    if len(set(pts)) == 30:
                        s = _score(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts
    
    # Strategy 2: Local Search from Best Grid
    if best_points and time.time() < deadline - 10:
        current = list(best_points)
        current_score = best_score
        selected_set = set(current)
        
        # Define a pool of candidate points for replacement
        all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
        
        max_local_iters = 20
        for it in range(max_local_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(30):
                if time.time() > deadline - 5:
                    break
                # Try a few random replacements
                for _ in range(10):
                    new_pt = random.choice(all_points)
                    if new_pt in selected_set:
                        continue
                    new_pts = current[:]
                    new_pts[i] = new_pt
                    new_score = _score(new_pts)
                    if new_score > current_score:
                        current = new_pts
                        current_score = new_score
                        selected_set = set(current)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_points = current
    
    if best_points is None:
        # Fallback: 2x15 grid centered at 0
        best_points = _generate_grid(2, 15, -14, -7)
        best_points = [(x, y) for x, y in best_points if -30 <= x <= 30 and -30 <= y <= 30]
        # Ensure 30 distinct points
        if len(best_points) < 30:
             best_points = _generate_grid(2, 15, 0, 0)

    return best_points
