# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incident_lines(points, lines, point_idx):
    """Count how many lines pass through a given point."""
    x, y = points[point_idx]
    return np.sum(lines[:, 0] * x + lines[:, 1] * y + lines[:, 2] == 0)

def _count_incident_points(lines, points, line_idx):
    """Count how many points lie on a given line."""
    A, B, C = lines[line_idx]
    return np.sum(A * points[:, 0] + B * points[:, 1] + C == 0)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580.0
    
    n_points = 54
    n_lines = 27
    
    # Initial configuration: 6x9 grid
    k = 6
    m = 9
    xs = np.arange(k)
    ys = np.arange(m)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Lines: all vertical, horizontal, and some diagonal
    lines_list = []
    # Vertical lines: x = i for i in 0..5
    for i in range(k):
        lines_list.append((1, 0, -i))
    # Horizontal lines: y = j for j in 0..8
    for j in range(m):
        lines_list.append((0, 1, -j))
    # That's 15 lines. Need 12 more.
    # Add diagonal lines of slope 1
    # Line: x - y = c, i.e., 1*x + (-1)*y + (-c) = 0
    # For c in range: min x-y = 0-8 = -8, max x-y = 5-0 = 5
    # Pick 12 values of c
    for c in range(-8, 6):
        if len(lines_list) >= n_lines:
            break
        lines_list.append((1, -1, -c))
    
    # Trim to exactly 27 lines
    lines_list = lines_list[:n_lines]
    lines = np.array(lines_list, dtype=np.int64)
    
    # Normalize and check for duplicates
    norm_lines = []
    for A, B, C in lines:
        norm_lines.append(_normalize_line(int(A), int(B), int(C)))
    
    # Remove duplicates
    seen = set()
    unique_lines = []
    unique_norms = []
    for i in range(n_lines):
        if norm_lines[i] not in seen:
            seen.add(norm_lines[i])
            unique_lines.append(lines[i])
            unique_norms.append(norm_lines[i])
    
    # If we have fewer than 27, add more lines
    while len(unique_lines) < n_lines:
        # Add a random line through two random points
        a = random.randint(0, n_points - 1)
        b = random.randint(0, n_points - 1)
        while b == a:
            b = random.randint(0, n_points - 1)
        x_a, y_a = points[a]
        x_b, y_b = points[b]
        A = y_b - y_a
        B = x_a - x_b
        C = -(A * x_a + B * y_a)
        norm = _normalize_line(int(A), int(B), int(C))
        if norm not in seen and (A != 0 or B != 0):
            seen.add(norm)
            unique_lines.append([A, B, C])
            unique_norms.append(norm)
        else:
            continue
    
    lines = np.array(unique_lines[:n_lines], dtype=np.int64)
    norm_lines = unique_norms[:n_lines]
    
    # Compute incidence matrix
    M = (lines[0:1, 0] * points[0:, 0:1] + lines[0:1, 1] * points[0:, 1:2] + lines[0:1, 2] == 0)
    # Full matrix
    M = (lines[:, 0][np.newaxis, :] * points[:, 0][:, np.newaxis] + 
         lines[:, 1][np.newaxis, :] * points[:, 1][:, np.newaxis] + 
         lines[:, 2][np.newaxis, :] == 0)
    
    I = np.sum(M)
    best_I = I
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.99995
    
    # Precompute for speed
    point_set = set(map(tuple, points.tolist()))
    
    while time.time() < deadline:
        # Generate a move
        if random.random() < 0.4:
            # Move a point to a location that maximizes incidences
            i = random.randint(0, n_points - 1)
            # Try several candidate positions
            best_delta = -1000
            best_pos = None
            for _ in range(20):
                # Random position near current
                new_px = points[i, 0] + random.randint(-2, 2)
                new_py = points[i, 1] + random.randint(-2, 2)
                if (new_px, new_py) in point_set and (new_px, new_py) != (points[i, 0], points[i, 1]):
                    continue
                # Count incidences at new position
                new_M_i = (lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2] == 0)
                delta = np.sum(new_M_i) - np.sum(M[i])
                if delta > best_delta:
                    best_delta = delta
                    best_pos = (new_px, new_py)
            
            if best_pos is None:
                continue
            
            new_px, new_py = best_pos
            # Check duplicate
            if (new_px, new_py) in point_set:
                continue
            
            new_M_i = (lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2] == 0)
            delta_I = np.sum(new_M_i) - np.sum(M[i])
            
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                M[i] = new_M_i
                point_set.discard((points[i, 0], points[i, 1]))
                points[i, 0] = new_px
                points[i, 1] = new_py
                point_set.add((new_px, new_py))
                I += delta_I
                
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
        
        else:
            # Retarget a line to pass through two points that maximize incidences
            j = random.randint(0, n_lines - 1)
            # Try several pairs of points
            best_delta = -1000
            best_line = None
            best_norm = None
            for _ in range(30):
                a = random.randint(0, n_points - 1)
                b = random.randint(0, n_points - 1)
                while b == a:
                    b = random.randint(0, n_points - 1)
                
                x_a, y_a = points[a]
                x_b, y_b = points[b]
                A = y_b - y_a
                B = x_a - x_b
                C = -(A * x_a + B * y_a)
                
                if A == 0 and B == 0:
                    continue
                
                norm = _normalize_line(int(A), int(B), int(C))
                # Check duplicate
                is_dup = False
                for l_idx in range(n_lines):
                    if l_idx == j:
                        continue
                    if norm_lines[l_idx] == norm:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                new_M_j = (A * points[:, 0] + B * points[:, 1] + C == 0)
                delta = np.sum(new_M_j) - np.sum(M[:, j])
                if delta > best_delta:
                    best_delta = delta
                    best_line = [A, B, C]
                    best_norm = norm
            
            if best_line is None:
                continue
            
            A_new, B_new, C_new = best_line
            new_M_j = (A_new * points[:, 0] + B_new * points[:, 1] + C_new == 0)
            delta_I = np.sum(new_M_j) - np.sum(M[:, j])
            
            if delta_I > 0 or random.random() < np.exp(delta_I / T):
                M[:, j] = new_M_j
                lines[j, 0] = A_new
                lines[j, 1] = B_new
                lines[j, 2] = C_new
                norm_lines[j] = best_norm
                I += delta_I
                
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final validation
    # Ensure no duplicate points
    pt_tuples = [tuple(p) for p in best_points]
    if len(set(pt_tuples)) != n_points:
        # Fix duplicates by moving points
        seen = set()
        for i in range(n_points):
            while tuple(best_points[i]) in seen:
                best_points[i, 1] += 1
            seen.add(tuple(best_points[i]))
    
    # Ensure no duplicate lines
    seen_norms = set()
    final_lines = []
    for i in range(n_lines):
        norm = _normalize_line(int(best_lines[i, 0]), int(best_lines[i, 1]), int(best_lines[i, 2]))
        if norm not in seen_norms and (best_lines[i, 0] != 0 or best_lines[i, 1] != 0):
            seen_norms.add(norm)
            final_lines.append(best_lines[i])
    
    while len(final_lines) < n_lines:
        # Add a line through two points
        a = random.randint(0, n_points - 1)
        b = random.randint(0, n_points - 1)
        while b == a:
            b = random.randint(0, n_points - 1)
        x_a, y_a = best_points[a]
        x_b, y_b = best_points[b]
        A = y_b - y_a
        B = x_a - x_b
        C = -(A * x_a + B * y_a)
        if A == 0 and B == 0:
            continue
        norm = _normalize_line(int(A), int(B), int(C))
        if norm not in seen_norms:
            seen_norms.add(norm)
            final_lines.append([A, B, C])
        else:
            continue
    
    final_lines = np.array(final_lines[:n_lines], dtype=np.int64)
    
    return best_points, final_lines

# EVOLVE-BLOCK-END
