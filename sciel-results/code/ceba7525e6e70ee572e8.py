"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for equal-area triangles from 30 points."""
import time
from collections import Counter
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin under 90s

    def evaluate(points):
        """Compute the maximum multiplicity of any positive doubled area using NumPy for speed."""
        import numpy as np
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        # Vectorized approach: for each pair (i,j), compute cross product with all k > j
        # Use a list to collect areas, then count
        areas = []
        for i in range(n):
            xi, yi = pts[i]
            for j in range(i+1, n):
                dx = pts[j][0] - xi
                dy = pts[j][1] - yi
                # Cross product of (dx,dy) with (pts[k]-pts[i]) for all k
                diffs_x = pts[:, 0] - xi
                diffs_y = pts[:, 1] - yi
                cross = dx * diffs_y - dy * diffs_x
                # Only count k > j
                cross[j+1:] = np.abs(cross[j+1:])
                # Filter positive areas
                pos = cross[cross > 0]
                if len(pos) > 0:
                    areas.append(pos)
        
        if not areas:
            return 0
        # Concatenate all areas and count
        all_areas = np.concatenate(areas)
        # Use np.unique with return_counts for fast counting
        unique, counts = np.unique(all_areas, return_counts=True)
        return int(counts.max())

    def try_config(points, current_best_score):
        """Try a configuration and return updated best if improved."""
        if len(set(points)) != 30:
            return None, current_best_score
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in points):
            return None, current_best_score
        score = evaluate(points)
        if score > current_best_score:
            return points, score
        return None, current_best_score

    # Model-and-optimize: parameterize by grid dimensions and spacing
    # The key insight: for a regular grid with spacing (sx, sy), the doubled areas
    # are multiples of |sx*sy|, and the multiplicity depends on the combinatorial structure.
    
    # Start with known good configurations
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate(best_points)

    # Systematic search over grid parameters:
    # A grid of width w and height h with spacings sx, sy gives points (i*sx, j*sy)
    # for i in range(w), j in range(h), with w*h = 30
    
    factors_30 = [(1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)]
    
    for w, h in factors_30:
        # Try various spacings
        for sx in range(1, min(31//w + 1, 31)):
            for sy in range(1, min(31//h + 1, 31)):
                if time.time() > deadline:
                    return best_points
                
                # Generate grid points
                pts = [(i * sx, j * sy) for i in range(w) for j in range(h)]
                
                # Check bounds
                max_x = (w - 1) * sx
                max_y = (h - 1) * sy
                if max_x > 30 or max_y > 30:
                    continue
                if max_x + max_y > 60:
                    continue
                    
                result, new_score = try_config(pts, best_score)
                if result is not None:
                    best_points = result
                    best_score = new_score

    # Also try centered grids (shift to center in [-30,30])
    for w, h in factors_30:
        for sx in range(1, 6):
            for sy in range(1, 6):
                if time.time() > deadline:
                    return best_points
                    
                width = (w - 1) * sx
                height = (h - 1) * sy
                
                if width > 60 or height > 60:
                    continue
                    
                # Center the grid
                cx = -width // 2
                cy = -height // 2
                
                pts = [(cx + i * sx, cy + j * sy) for i in range(w) for j in range(h)]
                
                result, new_score = try_config(pts, best_score)
                if result is not None:
                    best_points = result
                    best_score = new_score

    # Try sheared grids: points (i*sx + j*sxy, j*sy)
    for w, h in [(6,5), (5,6), (10,3), (3,10)]:
        for sx in range(1, 5):
            for sy in range(1, 5):
                for sxy in range(-2, 3):
                    if time.time() > deadline:
                        return best_points
                        
                    pts = [(i * sx + j * sxy, j * sy) for i in range(w) for j in range(h)]
                    
                    result, new_score = try_config(pts, best_score)
                    if result is not None:
                        best_points = result
                        best_score = new_score

    # Lattice section approach: select 30 points from a dense grid to maximize equal-area triples
    # Key insight: on a regular grid, triangles with the same vector-difference pattern have equal area
    # We search over grid parameters and which subset of grid points to use
    
    import random
    random.seed(42)
    
    # Try various dense grid subsets
    for grid_size in range(5, 12):
        if time.time() > deadline - 10:
            break
        for gx in range(-1, 2):  # shift
            for gy in range(-1, 2):
                # Generate all grid points in range
                all_grid_pts = [(i + gx, j + gy) for i in range(grid_size) for j in range(grid_size)]
                # Filter to valid range
                all_grid_pts = [(x, y) for x, y in all_grid_pts if -30 <= x <= 30 and -30 <= y <= 30]
                if len(all_grid_pts) < 30:
                    continue
                
                # Try different subsets: prefer points that form regular patterns
                # Strategy: take points from a sub-grid
                for sub_gx in range(1, grid_size // 2 + 1):
                    for sub_gy in range(1, grid_size // 2 + 1):
                        if time.time() > deadline - 5:
                            break
                        # Points at (i*sub_gx + gx, j*sub_gy + gy)
                        sub_pts = [(i * sub_gx + gx, j * sub_gy + gy) 
                                   for i in range(grid_size) 
                                   for j in range(grid_size)
                                   if 0 <= i * sub_gx + gx < grid_size and 0 <= j * sub_gy + gy < grid_size]
                        if len(sub_pts) >= 30:
                            # Take first 30 or a random subset
                            if len(sub_pts) == 30:
                                candidate = sub_pts
                            else:
                                random.shuffle(sub_pts)
                                candidate = sub_pts[:30]
                            result, new_score = try_config(candidate, best_score)
                            if result is not None:
                                best_points = result
                                best_score = new_score

    # Local search: hill-climbing on individual point moves with larger neighborhood
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and time.time() < deadline - 5 and iteration < max_iterations:
        improved = False
        iteration += 1
        for idx in range(30):
            if time.time() > deadline - 5:
                break
            old_x, old_y = best_points[idx]
            # Try moves in a neighborhood
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Pairwise swap local search: try swapping two points with nearby alternatives
    if time.time() < deadline - 10:
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            i, j = random.sample(range(30), 2)
            old_i = best_points[i]
            old_j = best_points[j]
            # Try moving point i to a new location and point j to another
            for dx1 in range(-2, 3):
                for dy1 in range(-2, 3):
                    if dx1 == 0 and dy1 == 0:
                        continue
                    new_i = (old_i[0] + dx1, old_i[1] + dy1)
                    if not (-30 <= new_i[0] <= 30 and -30 <= new_i[1] <= 30):
                        continue
                    for dx2 in range(-2, 3):
                        for dy2 in range(-2, 3):
                            if dx2 == 0 and dy2 == 0:
                                continue
                            new_j = (old_j[0] + dx2, old_j[1] + dy2)
                            if not (-30 <= new_j[0] <= 30 and -30 <= new_j[1] <= 30):
                                continue
                            new_points = list(best_points)
                            new_points[i] = new_i
                            new_points[j] = new_j
                            if len(set(new_points)) != 30:
                                continue
                            score = evaluate(new_points)
                            if score > best_score:
                                best_score = score
                                best_points = new_points
                                break
                        if best_score > 0 and best_points != [old_i] * 30:  # check if improved
                            if new_points[i] != old_i or new_points[j] != old_j:
                                break
                    if best_points[i] != old_i or best_points[j] != old_j:
                        break
                if best_points[i] != old_i or best_points[j] != old_j:
                    break

    # Random restarts for remaining time
    if time.time() < deadline - 3:
        random.seed(123)
        for restart in range(100):
            if time.time() > deadline - 2:
                break
            
            # Generate random configuration near the best
            if random.random() < 0.5:
                # Small perturbation of best
                new_points = list(best_points)
                for _ in range(random.randint(1, 4)):
                    idx = random.randint(0, 29)
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    new_x = new_points[idx][0] + dx
                    new_y = new_points[idx][1] + dy
                    if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                        new_points[idx] = (new_x, new_y)
            else:
                # Random grid-like structure
                w, h = random.choice([(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)])
                sx = random.randint(1, max(1, 30 // w))
                sy = random.randint(1, max(1, 30 // h))
                cx = random.randint(-5, 5)
                cy = random.randint(-5, 5)
                new_points = [(cx + i*sx + random.randint(-1,1), cy + j*sy + random.randint(-1,1)) 
                              for i in range(w) for j in range(h)]
            
            result, new_score = try_config(new_points, best_score)
            if result is not None:
                best_points = result
                best_score = new_score

    return best_points
# EVOLVE-BLOCK-END
