import numpy as np
from math import gcd
import time

def normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """Count total incidences between points and lines."""
    x = points[:, 0]
    y = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    
    vals = A[:, np.newaxis] * x[np.newaxis, :] + B[:, np.newaxis] * y[np.newaxis, :] + C[:, np.newaxis]
    inc = np.sum(vals == 0)
    return inc

def get_top_k_lines(points, k=27):
    """Get the top k lines by incidence count for a given set of points."""
    n_pts = len(points)
    if n_pts < 2:
        return np.array([], dtype=np.int64), 0
    
    lines_set = set()
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            A = int(y1 - y2)
            B = int(x2 - x1)
            C = int(x1 * y2 - x2 * y1)
            norm = normalize_line(A, B, C)
            if norm != (0, 0, 0):
                lines_set.add(norm)
    
    if len(lines_set) < k:
        return np.array(list(lines_set), dtype=np.int64), 0
    
    lines_arr = np.array(list(lines_set), dtype=np.int64)
    
    x = points[:, 0]
    y = points[:, 1]
    
    A = lines_arr[:, 0]
    B = lines_arr[:, 1]
    C = lines_arr[:, 2]
    
    vals = A[:, np.newaxis] * x[np.newaxis, :] + B[:, np.newaxis] * y[np.newaxis, :] + C[:, np.newaxis]
    counts = np.sum(vals == 0, axis=1)
    
    sorted_indices = np.argsort(counts)[::-1]
    top_k_indices = sorted_indices[:k]
    
    selected_lines = lines_arr[top_k_indices]
    score = int(np.sum(counts[top_k_indices]))
    
    return selected_lines, score

def local_search(points, max_iterations=5000, time_limit=300):
    """Local search to improve point configuration."""
    start_time = time.time()
    current_points = points.copy()
    current_lines, current_score = get_top_k_lines(current_points)
    
    if current_score == 0:
        return current_points, current_lines, 0
    
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    # Try moving each point to nearby positions
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Shuffle point order for diversity
        np.random.seed(iteration)
        point_order = np.random.permutation(len(current_points))
        
        for idx in point_order:
            if time.time() - start_time > time_limit:
                break
            
            x, y = current_points[idx]
            
            # Try small moves: (dx, dy) in a small neighborhood
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x = int(x + dx)
                    new_y = int(y + dy)
                    
                    # Create new point set
                    new_points = current_points.copy()
                    new_points[idx] = [new_x, new_y]
                    
                    # Check for duplicate points
                    if len(set(map(tuple, new_points))) != len(new_points):
                        continue
                    
                    # Get top k lines for new configuration
                    new_lines, new_score = get_top_k_lines(new_points)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        best_lines = new_lines.copy()
                        current_points = new_points
                        current_lines = new_lines
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try larger moves if no improvement
            for idx in point_order:
                if time.time() - start_time > time_limit:
                    break
                
                x, y = current_points[idx]
                
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        if abs(dx) <= 2 and abs(dy) <= 2:
                            continue
                        
                        new_x = int(x + dx)
                        new_y = int(y + dy)
                        
                        new_points = current_points.copy()
                        new_points[idx] = [new_x, new_y]
                        
                        if len(set(map(tuple, new_points))) != len(new_points):
                            continue
                        
                        new_lines, new_score = get_top_k_lines(new_points)
                        
                        if new_score > best_score:
                            best_score = new_score
                            best_points = new_points.copy()
                            best_lines = new_lines.copy()
                            current_points = new_points
                            current_lines = new_lines
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # No improvement found, break
            break
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 550  # Leave 80s margin from 630s limit
    
    # Factors of 54
    factors = [1, 2, 3, 6, 9, 18, 27, 54]
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # First pass: try all rectangular grids
    for W in factors:
        if time.time() - start_time > deadline:
            break
        
        H = 54 // W
        xs = np.arange(W)
        ys = np.arange(H)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        lines_arr, score = get_top_k_lines(points)
        
        if len(lines_arr) >= 27 and score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines_arr.copy()
    
    # If no valid grid found, fallback
    if best_points is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
        return best_points, best_lines
    
    # Second pass: local search from best grid
    if time.time() - start_time < deadline - 10:
        search_time_limit = min(200, deadline - (time.time() - start_time) - 10)
        improved_points, improved_lines, improved_score = local_search(
            best_points, max_iterations=10000, time_limit=search_time_limit
        )
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
            best_lines = improved_lines
    
    return best_points, best_lines
