import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    """Count incidences between points (N,2) and lines (M,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    # Vectorized check: (M, 1) * (1, N) + (M, 1) * (1, N) + (M, 1) * (1, 1)
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _get_best_lines(points, k=27):
    """
    Find the k lines with the highest incidence count for the given points.
    Lines are derived from pairs of points.
    """
    n = len(points)
    if n < 2:
        return np.zeros((k, 3), dtype=np.int64)
    
    xs = points[:, 0]
    ys = points[:, 1]
    
    # Generate all candidate lines from pairs of points
    # i < j
    # A = y_j - y_i
    # B = x_i - x_j
    # C = x_j * y_i - x_i * y_j
    
    # To avoid O(N^2) Python loop for normalization, we generate arrays
    # But normalization requires gcd which is not vectorized in NumPy standard.
    # So we loop, but it's only ~1400 iterations.
    
    line_set = {}
    
    # Precompute coordinates as lists for faster access in loop?
    # xs, ys are numpy arrays. Accessing xs[i] is fast.
    
    for i in range(n):
        x1 = xs[i]
        y1 = ys[i]
        for j in range(i + 1, n):
            x2 = xs[j]
            y2 = ys[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
                
            key = _normalize_line(A, B, C)
            
            if key not in line_set:
                line_set[key] = 0
                
    # If we don't have enough lines, add generic ones (unlikely for 54 pts)
    if len(line_set) < k:
        c = 0
        while len(line_set) < k:
            key = _normalize_line(0, 1, -c)
            if key not in line_set:
                line_set[key] = 0
            c += 1
            if c > 1000: break

    # Convert to arrays for vectorized counting
    # We only need to count for unique lines
    unique_lines = np.array(list(line_set.keys()), dtype=np.int64)
    
    # Count incidences
    A = unique_lines[:, 0]
    B = unique_lines[:, 1]
    C = unique_lines[:, 2]
    
    vals = A[:, None] * xs[None, :] + B[:, None] * ys[None, :] + C[:, None]
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k
    # If we have fewer than k, pad
    if len(unique_lines) < k:
        # Pad with dummy lines that have 0 incidence
        dummy = np.zeros((k - len(unique_lines), 3), dtype=np.int64)
        dummy[:, 1] = 1
        dummy[:, 2] = -10000 # Far away
        unique_lines = np.vstack([unique_lines, dummy])
        counts = np.concatenate([counts, np.zeros(k - len(counts))])
        
    # Sort indices by count descending
    # We want the top k
    top_indices = np.argsort(counts)[::-1][:k]
    
    return unique_lines[top_indices]

def _generate_grid_points(k, m, dx=1, dy=1, ox=0, oy=0):
    """Generate a k x m grid of points."""
    xs = np.arange(k) * dx + ox
    ys = np.arange(m) * dy + oy
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Candidate initial configurations
    candidates = []
    
    # 6x9 grid
    candidates.append(_generate_grid_points(6, 9))
    # 9x6 grid
    candidates.append(_generate_grid_points(9, 6))
    # 3x18 grid
    candidates.append(_generate_grid_points(3, 18))
    # 18x3 grid
    candidates.append(_generate_grid_points(18, 3))
    # 2x27 grid
    candidates.append(_generate_grid_points(2, 27))
    # 27x2 grid
    candidates.append(_generate_grid_points(27, 2))
    
    # Evaluate candidates
    for pts in candidates:
        if len(pts) != 54:
            continue
        # Check uniqueness
        if len(set(map(tuple, pts))) != 54:
            continue
            
        lines = _get_best_lines(pts, 27)
        score = _count_incidence(pts, lines)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()
            
    if best_points is None:
        best_points = _generate_grid_points(6, 9)
        best_lines = _get_best_lines(best_points, 27)
        best_score = _count_incidence(best_points, best_lines)
        
    points = best_points.copy()
    lines = best_lines.copy()
    current_score = best_score
    
    # Simulated Annealing / Hill Climbing
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart or cool down further
            T = max(T_min, T * 0.5)
            no_improve_count = 0
            
        i = random.randint(0, 53)
        
        # Try a random move
        # Bias towards small moves
        move_type = random.random()
        if move_type < 0.7:
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
        elif move_type < 0.9:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
        else:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            
        if dx == 0 and dy == 0:
            continue
            
        new_x = int(points[i, 0]) + dx
        new_y = int(points[i, 1]) + dy
        
        # Check uniqueness
        # Vectorized check for collision
        # points[:, 0] == new_x and points[:, 1] == new_y
        # Exclude i
        collision = ((points[:, 0] == new_x) & (points[:, 1] == new_y))
        if np.sum(collision) > 0:
            # If the collision is at i, it's fine (no move)
            # But we already ensured dx,dy not both 0, so collision must be at j != i
            # If collision is only at i, it means the point didn't move? No, coords changed.
            # So if sum > 0, there is a collision with another point.
            continue
        
        new_points = points.copy()
        new_points[i, 0] = new_x
        new_points[i, 1] = new_y
        
        new_lines = _get_best_lines(new_points, 27)
        new_score = _count_incidence(new_points, new_lines)
        
        delta = new_score - current_score
        
        if delta > 0:
            points = new_points
            lines = new_lines
            current_score = new_score
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
        else:
            # Accept with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                points = new_points
                lines = new_lines
                current_score = new_score
                no_improve_count += 1
            else:
                no_improve_count += 1
                
        T = max(T_min, T * cooling_rate)
        
        # Periodic restart to escape deep local minima
        if no_improve_count > 5000:
            # Reset to best known
            points = best_points.copy()
            lines = best_lines.copy()
            current_score = best_score
            T = 1.0
            no_improve_count = 0
            
    # Final validation
    final_points = best_points.astype(np.int64)
    final_lines = best_lines.astype(np.int64)
    
    if final_points.shape != (54, 2):
        final_points = _generate_grid_points(6, 9)
        final_lines = _get_best_lines(final_points, 27)
        
    if final_lines.shape != (27, 3):
        final_lines = final_lines[:27]
        if len(final_lines) < 27:
            pad = np.zeros((27 - len(final_lines), 3), dtype=np.int64)
            pad[:, 1] = 1
            pad[:, 2] = -1000
            final_lines = np.vstack([final_lines, pad])
            
    return final_points, final_lines
