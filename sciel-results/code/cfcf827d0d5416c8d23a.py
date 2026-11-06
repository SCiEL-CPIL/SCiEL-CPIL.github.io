import random
import itertools
import numpy as np
import time
import math

def _check_no_four_collinear(points):
    """Check no four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                a = yj - yi
                b = xi - xj
                c = xj*yi - xi*yj
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    xl, yl = points[l]
                    if a * xl + b * yl + c == 0:
                        return False
    return True

def _score(points):
    """Largest subset with no three collinear (exact branch and bound)."""
    n = len(points)
    if n == 0:
        return 0
    
    line_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) == (xk - xi) * (yj - yi):
                    mask |= (1 << k)
            line_mask[i][j] = mask
            line_mask[j][i] = mask

    best = [0]
    
    def dfs(idx, current_mask, current_size):
        if current_size + (n - idx) <= best[0]:
            return
        if idx == n:
            best[0] = max(best[0], current_size)
            return
        
        dfs(idx + 1, current_mask, current_size)
        
        can_take = True
        m = current_mask
        while m:
            lsb = m & -m
            a_idx = lsb.bit_length() - 1
            if line_mask[a_idx][idx] & current_mask:
                can_take = False
                break
            m ^= lsb
            
        if can_take:
            dfs(idx + 1, current_mask | (1 << idx), current_size + 1)

    dfs(0, 0, 0)
    return best[0]

def _construct_structured():
    """Construct points using a structured approach with many 3-collinear triples."""
    # Use 6 lines, each with exactly 3 points, arranged to minimize no-three-collinear subset
    # Lines: use lines with small integer slopes/intercepts
    # Strategy: place points at intersections of a grid with specific offsets
    
    # Base grid: 5 columns x 4 rows
    # For each cell (i,j), place at (i*8 + dx[i][j], j*8 + dy[i][j])
    # Choose dx, dy to create collinear triples along rows, columns, and diagonals
    
    # Define offsets to create collinearities
    # Row collinearities: for each row j, make 3 points collinear
    # We'll use a pattern where in each row, points at columns 0,1,2 are collinear,
    # and points at columns 2,3,4 are collinear (sharing column 2)
    
    random.seed(123)
    
    # Try multiple structured configurations
    best_points = None
    best_score = float('inf')
    
    for attempt in range(200):
        pts = []
        for i in range(5):
            for j in range(4):
                # Base position
                x_base = i * 10
                y_base = j * 10
                # Add structured perturbation
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                pts.append([x_base + dx, y_base + dy])
        
        # Check validity
        if len(set(tuple(p) for p in pts)) != 20:
            continue
        if not _check_no_four_collinear(pts):
            continue
        
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
    
    if best_points is None:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points

def construct():
    """Use local search with structured initialization and better moves."""
    # Start with structured configuration
    points = _construct_structured()
    current_score = _score(points)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    start_time = time.time()
    time_limit = 110.0
    
    # Local search with multiple move types
    T = 3.0
    T_min = 0.01
    alpha = 0.9995
    
    while time.time() - start_time < time_limit:
        new_points = [p[:] for p in points]
        move_type = random.random()
        
        if move_type < 0.4:
            # Move one point
            idx = random.randint(0, 19)
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)
            new_points[idx][0] += dx
            new_points[idx][1] += dy
        elif move_type < 0.7:
            # Swap two points
            i, j = random.sample(range(20), 2)
            new_points[i], new_points[j] = new_points[j][:], new_points[i][:]
        else:
            # Move two points
            i, j = random.sample(range(20), 2)
            for idx in [i, j]:
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
                new_points[idx][0] += dx
                new_points[idx][1] += dy
        
        # Check bounds
        valid = True
        for p in new_points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check distinctness
        if len(set(tuple(p) for p in new_points)) != 20:
            continue
            
        # Check no 4 collinear
        if not _check_no_four_collinear(new_points):
            continue
        
        new_score = _score(new_points)
        
        if new_score < current_score:
            points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    points = new_points
                    current_score = new_score
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_points
