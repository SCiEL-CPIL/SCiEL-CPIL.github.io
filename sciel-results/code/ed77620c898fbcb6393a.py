"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
from itertools import combinations
import math
import time

def _compute_best_area_multiplicity(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    from collections import Counter
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs(int(pts[i][0])*(pts[j][1]-pts[k][1]) + 
                         int(pts[j][0])*(pts[k][1]-pts[i][1]) + 
                         int(pts[k][0])*(pts[i][1]-pts[j][1]))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counter = Counter(areas)
    return counter.most_common(1)[0][1]

def _make_grid(w, h, offset_x=0, offset_y=0):
    """Create a w x h grid of points."""
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x + offset_x, y + offset_y))
    return points

def _make_hexagonal(rows, cols):
    """Create a hexagonal-like grid (offset rows)."""
    points = []
    for r in range(rows):
        for c in range(cols):
            # Offset every other row by 0.5? No, integer coords.
            # Standard hex lattice on integer grid: (c + r//2, r) or similar.
            # Let's use (c, r) for even r, (c+1, r) for odd r? 
            # Actually, simple grid is better for integer lattice area properties.
            # Let's stick to a slightly perturbed grid or a true hex packing approximation.
            x = c
            y = r
            if r % 2 != 0:
                x += 1 # Shift
            points.append((x, y))
    # Filter duplicates and out of bounds
    unique_pts = []
    seen = set()
    for p in points:
        if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
            seen.add(p)
            unique_pts.append(p)
    return unique_pts

def _calculate_area_counts(points):
    """Calculate area counts for a set of points. Returns Counter and max score."""
    from collections import Counter
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    area_counts = Counter()
    
    # Vectorized calculation for speed if possible, but n=30 is small.
    # Triple loop is O(N^3) = 27000 ops. Fast enough.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs(int(pts[i][0])*(pts[j][1]-pts[k][1]) + 
                         int(pts[j][0])*(pts[k][1]-pts[i][1]) + 
                         int(pts[k][0])*(pts[i][1]-pts[j][1]))
                if det > 0:
                    area_counts[det] += 1
    max_score = area_counts.most_common(1)[0][1] if area_counts else 0
    return area_counts, max_score

def _simulated_annealing(initial_points, steps=10000, t0=10.0, t_end=0.1):
    """
    Simulated Annealing with 1-point and 2-point moves.
    """
    current_points = [p[:] for p in initial_points]
    n = len(current_points)
    current_counts, current_score = _calculate_area_counts(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Cooling schedule
    t = t0
    cool_rate = (t_end / t0) ** (1.0 / steps)
    
    for step in range(steps):
        # Generate move
        move_type = random.random()
        if move_type < 0.7:
            # 1-point move
            i = random.randint(0, n - 1)
            x, y = current_points[i]
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                continue
            
            new_point = (nx, ny)
            # Check distinctness
            is_distinct = True
            for j, p in enumerate(current_points):
                if j != i and p == new_point:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
            
            # Calculate delta score efficiently?
            # For SA, we need new_score. 
            # Recalculating all areas is O(N^3). 
            # Optimizing delta is complex. With N=30, O(N^3) is ~27k ops.
            # 10000 steps * 27k = 270M ops. Might be too slow in Python.
            # Let's limit steps or optimize.
            # We can precompute areas involving i.
            
            # Remove areas involving i
            old_areas = []
            pts_arr = np.array(current_points, dtype=np.int64)
            pi = pts_arr[i]
            for j in range(n):
                if j == i: continue
                pj = pts_arr[j]
                for k in range(j+1, n):
                    if k == i: continue
                    pk = pts_arr[k]
                    det = abs(int(pi[0])*(int(pj[1]) - int(pk[1])) + 
                              int(pj[0])*(int(pk[1]) - int(pi[1])) + 
                              int(pk[0])*(int(pi[1]) - int(pj[1])))
                    if det > 0:
                        old_areas.append(det)
            
            # Apply move
            current_points[i] = new_point
            pts_arr[i] = new_point # Update array
            
            # Add new areas involving i
            new_areas = []
            pi_new = pts_arr[i]
            for j in range(n):
                if j == i: continue
                pj = pts_arr[j]
                for k in range(j+1, n):
                    if k == i: continue
                    pk = pts_arr[k]
                    det = abs(int(pi_new[0])*(int(pj[1]) - int(pk[1])) + 
                              int(pj[0])*(int(pk[1]) - int(pi_new[1])) + 
                              int(pk[0])*(int(pi_new[1]) - int(pj[1])))
                    if det > 0:
                        new_areas.append(det)
            
            # Update counts
            for a in old_areas:
                current_counts[a] -= 1
                if current_counts[a] == 0:
                    del current_counts[a]
            for a in new_areas:
                current_counts[a] = current_counts.get(a, 0) + 1
            
            new_score = max(current_counts.values()) if current_counts else 0
            
            # Acceptance
            if new_score >= current_score or random.random() < math.exp((new_score - current_score) / t):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Revert
                current_points[i] = (x, y)
                pts_arr[i] = (x, y)
                for a in new_areas:
                    current_counts[a] -= 1
                    if current_counts[a] == 0:
                        del current_counts[a]
                for a in old_areas:
                    current_counts[a] = current_counts.get(a, 0) + 1
        else:
            # 2-point move (swap or move two)
            i, j = random.sample(range(n), 2)
            x1, y1 = current_points[i]
            x2, y2 = current_points[j]
            
            # Move i to x2, y2 and j to x1, y1 (Swap)
            # Or move both randomly? Swap is a good move.
            new_p1 = (x2, y2)
            new_p2 = (x1, y1)
            
            # Check bounds
            if not (-30 <= new_p1[0] <= 30 and -30 <= new_p1[1] <= 30): continue
            if not (-30 <= new_p2[0] <= 30 and -30 <= new_p2[1] <= 30): continue
            
            # Swap doesn't change distinctness if they were distinct before.
            
            # Calculate old areas involving i or j
            # This is more complex. 
            # Areas involving only i (not j)
            # Areas involving only j (not i)
            # Areas involving both i and j
            
            # For simplicity and speed, if 2-point move is too slow, we skip.
            # Given time limits, 1-point move with optimized delta is safer.
            pass
            
        t *= cool_rate
        
    return best_points, best_score

def _try_configurations():
    """Try various configurations and return the best one."""
    initial_configs = []
    
    # 6x5 Grid centered
    grid_6_5 = _make_grid(6, 5, -2, -2)
    initial_configs.append(grid_6_5)
    
    # 5x6 Grid centered
    grid_5_6 = _make_grid(5, 6, -2, -2)
    initial_configs.append(grid_5_6)
    
    # Hexagonal
    hex_pts = _make_hexagonal(6, 6)
    if len(hex_pts) >= 30:
        # Randomly select 30 if more
        random.seed(42)
        hex_pts = random.sample(hex_pts, 30)
    else:
        # Pad with random
        random.seed(42)
        while len(hex_pts) < 30:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in hex_pts:
                hex_pts.append((x, y))
    initial_configs.append(hex_pts)
    
    # Random initial config
    random.seed(123)
    random_points = set()
    while len(random_points) < 30:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        random_points.add((x, y))
    initial_configs.append(list(random_points))
    
    best_pts = None
    best_mult = -1
    
    # Run SA on each config
    for config in initial_configs:
        pts, score = _simulated_annealing(config, steps=5000, t0=5.0, t_end=0.01)
        if score > best_mult:
            best_mult = score
            best_pts = pts
            
    if best_pts is None:
        best_pts = _make_grid(6, 5)
        
    return best_pts

def construct():
    return _try_configurations()
