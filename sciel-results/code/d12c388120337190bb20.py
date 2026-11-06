"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s wall-clock

def _doubled_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def _compute_max_multiplicity(points):
    """Compute the maximum multiplicity of a single positive doubled area among all triples."""
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d = _doubled_area(points[i], points[j], points[k])
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _fast_score(points):
    """Vectorized score computation using numpy."""
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    if n < 3:
        return 0
    # Compute all doubled areas
    # For each triple (i,j,k), doubled area = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # Use vectorized approach: for each pair (i,j), compute areas with all k
    max_mult = 0
    area_counter = Counter()
    for i in range(n):
        for j in range(i+1, n):
            # Vectorized computation of areas for all k > j
            xi, yi = pts[i]
            xj, yj = pts[j]
            # For k from j+1 to n-1
            if j + 1 >= n:
                continue
            ks = pts[j+1:n]
            # doubled area = |xi*(yj - yk) + xj*(yk - yi) + xk*(yi - yj)|
            # = |xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj|
            # = |(xi*yj - xj*yi) + xk*(yi - yj) + yk*(xj - xi)|
            const = xi * yj - xj * yi
            dx = yi - yj
            dy = xj - xi
            areas = np.abs(const + ks[:, 0] * dx + ks[:, 1] * dy)
            # Only count positive areas
            mask = areas > 0
            if np.any(mask):
                for a in areas[mask]:
                    area_counter[int(a)] += 1
    if area_counter:
        max_mult = max(area_counter.values())
    return max_mult

def _try_grid_config(cx, cy, w, h):
    """Try a rectangular grid of w x h points centered at (cx, cy)."""
    if w * h != 30:
        return None, 0
    x_start = cx - (w - 1) // 2
    y_start = cy - (h - 1) // 2
    points = []
    valid = True
    for dx in range(w):
        for dy in range(h):
            x = x_start + dx
            y = y_start + dy
            if x < -30 or x > 30 or y < -30 or y > 30:
                valid = False
                break
            points.append((x, y))
        if not valid:
            break
    if not valid or len(points) != 30:
        return None, 0
    score = _fast_score(points)
    return points, score

def _try_rectangular_grid():
    """Try all rectangular grid configurations."""
    best_points = None
    best_score = 0
    # Factor pairs of 30: (1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)
    for w in [1, 2, 3, 5, 6, 10, 15, 30]:
        h = 30 // w
        if w * h != 30:
            continue
        # Try various centers
        for cx in range(-25, 26, 1):
            for cy in range(-25, 26, 1):
                pts, score = _try_grid_config(cx, cy, w, h)
                if pts is not None and score > best_score:
                    best_score = score
                    best_points = pts
    return best_points, best_score

def _local_search(points, time_limit):
    """Local search: try swapping points to improve score."""
    points = list(points)
    current_score = _fast_score(points)
    start_time = time.time()
    best_points = list(points)
    best_score = current_score
    
    # Precompute all possible point positions
    all_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try each position in the current set
        for i in range(30):
            if time.time() - start_time > time_limit:
                break
            old_point = points[i]
            # Try replacing with a random new point
            for _ in range(50):
                new_point = random.choice(all_positions)
                if new_point in points:
                    continue
                points[i] = new_point
                score = _fast_score(points)
                if score > current_score:
                    current_score = score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_points = list(points)
                    break
                else:
                    points[i] = old_point
        if not improved:
            # Try more aggressive moves: swap two points
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    break
                i, j = random.sample(range(30), 2)
                old_i, old_j = points[i], points[j]
                new_i = random.choice(all_positions)
                new_j = random.choice(all_positions)
                if new_i in points or new_j in points or new_i == new_j:
                    continue
                points[i] = new_i
                points[j] = new_j
                score = _fast_score(points)
                if score > current_score:
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_points = list(points)
                else:
                    points[i] = old_i
                    points[j] = old_j
    
    return best_points, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Try rectangular grid configurations (fast)
    grid_points, grid_score = _try_rectangular_grid()
    best_points = grid_points if grid_points else [(x, y) for x in range(6) for y in range(5)]
    best_score = grid_score
    
    # Phase 2: Local search from the best grid configuration
    time_left = DEADLINE - (time.time() - start_time)
    if time_left > 5:
        best_points, best_score = _local_search(best_points, time_left - 5)
    
    # Phase 3: If time remains, try more random restarts
    time_left = DEADLINE - (time.time() - start_time)
    if time_left > 10:
        all_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
        restarts = min(5, int(time_left / 10))
        for r in range(restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Random initial configuration
            random_pts = random.sample(all_positions, 30)
            t_left = DEADLINE - (time.time() - start_time) - 2
            if t_left < 3:
                break
            pts, score = _local_search(random_pts, t_left)
            if score > best_score:
                best_score = score
                best_points = pts
    
    # Final safety check
    if time.time() - start_time > DEADLINE:
        # Return whatever we have
        pass
    
    return best_points

# EVOLVE-BLOCK-END
