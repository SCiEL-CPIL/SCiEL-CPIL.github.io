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

def _incidence_count(points, lines):
    # points: (N, 2), lines: (M, 3)
    # Vectorized calculation
    # A*x + B*y + C == 0
    # Use broadcasting
    px = points[:, 0].reshape(-1, 1)
    py = points[:, 1].reshape(-1, 1)
    A = lines[:, 0].reshape(1, -1)
    B = lines[:, 1].reshape(1, -1)
    C = lines[:, 2].reshape(1, -1)
    
    total = A * px + B * py + C
    return np.sum(total == 0)

def _get_best_lines(points):
    n = len(points)
    # Generate all unique lines passing through at least 2 points
    # For N=54, pairs are ~1400. This is fine.
    lines_set = set()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A != 0 or B != 0:
                # Normalize
                g = math.gcd(abs(A), abs(B))
                g = math.gcd(g, abs(C))
                if g == 0: continue
                A, B, C = A // g, B // g, C // g
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A, B, C = -A, -B, -C
                lines_set.add((A, B, C))
    
    # Convert to array
    if not lines_set:
        # Fallback
        lines_arr = np.zeros((27, 3), dtype=np.int64)
        for i in range(27):
            lines_arr[i] = [1, 0, -i]
        return lines_arr

    lines_list = list(lines_set)
    
    # If we have more than 27 lines, we need to pick the best 27.
    # If fewer, we pad with arbitrary lines (though this shouldn't happen for N=54).
    if len(lines_list) > 27:
        lines_arr = np.array(lines_list, dtype=np.int64)
        # Calculate incidence count for each line
        px = points[:, 0].reshape(-1, 1)
        py = points[:, 1].reshape(-1, 1)
        A = lines_arr[:, 0].reshape(1, -1)
        B = lines_arr[:, 1].reshape(1, -1)
        C = lines_arr[:, 2].reshape(1, -1)
        
        total = A * px + B * py + C
        counts = np.sum(total == 0, axis=1)
        
        # Get indices of top 27
        idx = np.argsort(-counts)
        top_idx = idx[:27]
        return lines_arr[top_idx]
    else:
        lines_arr = np.array(lines_list, dtype=np.int64)
        # Pad if necessary (unlikely)
        if len(lines_arr) < 27:
            pad_size = 27 - len(lines_arr)
            pad = np.zeros((pad_size, 3), dtype=np.int64)
            for i in range(pad_size):
                pad[i] = [1, 0, -100 - i]
            lines_arr = np.vstack([lines_arr, pad])
        return lines_arr

def _run_hill_climbing(points, deadline):
    n_points = len(points)
    current_score = _incidence_count(points, _get_best_lines(points))
    best_score = current_score
    best_points = points.copy()
    best_lines = _get_best_lines(points)
    
    # Temperature for simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < deadline - 10:
        # Pick a random point
        i = random.randint(0, n_points - 1)
        
        # Generate candidate moves
        # 1. Small step
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        new_x = points[i, 0] + dx
        new_y = points[i, 1] + dy
        
        # Check collision
        # Efficient collision check:
        # Create a set of points? Or vectorized check.
        # Vectorized check is fast enough for N=54.
        mask = (points[:, 0] == new_x) & (points[:, 1] == new_y)
        if np.sum(mask) > 0:
            # If the point is already there (and it's not i), it's a collision.
            # Since we picked i, if mask has i, it's the same point (dx=0, dy=0).
            # If mask has another index, it's a collision.
            if np.any(mask & (np.arange(n_points) != i)):
                continue
            # If dx=0 and dy=0, score doesn't change. Skip.
            if dx == 0 and dy == 0:
                continue

        new_points = points.copy()
        new_points[i, 0] = new_x
        new_points[i, 1] = new_y
        
        new_lines = _get_best_lines(new_points)
        new_score = _incidence_count(new_points, new_lines)
        
        # Acceptance
        if new_score > current_score:
            points = new_points
            lines = new_lines
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
        else:
            # Delta E = new_score - current_score (negative)
            delta_E = new_score - current_score
            if delta_E == 0:
                if random.random() < 0.5:
                    points = new_points
                    lines = new_lines
            else:
                # Boltzmann factor
                prob = math.exp(delta_E / T)
                if random.random() < prob:
                    points = new_points
                    lines = new_lines
                    current_score = new_score
                    
        T *= cooling
        if T < T_min:
            T = T_min

    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590
    
    # Try multiple initial configurations
    configs = []
    
    # 6x9 Grid
    k1, k2 = 6, 9
    xs = np.arange(1, k1 + 1)
    ys = np.arange(1, k2 + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points1 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    configs.append(points1)
    
    # 9x6 Grid
    k1, k2 = 9, 6
    xs = np.arange(1, k1 + 1)
    ys = np.arange(1, k2 + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points2 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    configs.append(points2)
    
    # 3x18 Grid
    k1, k2 = 3, 18
    xs = np.arange(1, k1 + 1)
    ys = np.arange(1, k2 + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points3 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    configs.append(points3)
    
    # 18x3 Grid
    k1, k2 = 18, 3
    xs = np.arange(1, k1 + 1)
    ys = np.arange(1, k2 + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points4 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    configs.append(points4)
    
    # Random points in a box
    points5 = np.random.randint(-10, 10, size=(54, 2)).astype(np.int64)
    configs.append(points5)
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Allocate time per config
    # We have ~590s. 5 configs. ~118s each.
    # But some might be better.
    # Let's run them sequentially with a shared deadline.
    
    for i, init_points in enumerate(configs):
        if time.time() > deadline - 20:
            break
            
        # Run hill climbing
        p, l, s = _run_hill_climbing(init_points, deadline)
        
        if s > best_score:
            best_score = s
            best_points = p
            best_lines = l
            
    return best_points, best_lines
