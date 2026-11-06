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

def _count_3_lines(points):
    """Count the number of triples of collinear points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            # Line equation: (xj - xi) * (y - yi) - (yj - yi) * (x - xi) = 0
            # Let A = xj - xi, B = yi - yj
            # A * y - B * x + (B * xi - A * yi) = 0
            A = xj - xi
            B = yi - yj
            C = B * xi - A * yi
            collinear = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if A * yk - B * xk + C == 0:
                    collinear += 1
            # If there are exactly 1 other point, it forms a unique 3-point line.
            # Note: If there are 2 other points, it's a 4-collinear set, which is invalid.
            # So we only count if collinear == 1.
            if collinear == 1:
                count += 1
    return count

def _score(points):
    """Largest subset with no three collinear (exact branch and bound)."""
    n = len(points)
    if n == 0:
        return 0
        
    # Precompute collinearity masks
    # line_mask[i][j] is a bitmask of points k such that i, j, k are collinear.
    # We need to check if adding point i to a set S creates a triple.
    # This happens if there exists a, b in S such that a, b, i are collinear.
    # This is equivalent to: for any a in S, if i is on a line with a and some b in S.
    # Let's precompute for each pair (i, j) the mask of points collinear with them.
    
    collinear_with = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            A = xj - xi
            B = yi - yj
            C = B * xi - A * yi
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if A * yk - B * xk + C == 0:
                    mask |= (1 << k)
            collinear_with[i][j] = mask
            collinear_with[j][i] = mask

    best = [0]
    
    def dfs(idx, current_mask, current_size):
        if current_size + (n - idx) <= best[0]:
            return
        if idx == n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Option 1: Skip idx
        dfs(idx + 1, current_mask, current_size)
        
        # Option 2: Take idx
        # Check if idx can be added to current_mask
        can_take = True
        m = current_mask
        while m:
            lsb = m & -m
            a_idx = lsb.bit_length() - 1
            # If any point b in current_mask is collinear with a_idx and idx, we have a triple.
            # collinear_with[a_idx][idx] contains all b such that a_idx, idx, b are collinear.
            # If (collinear_with[a_idx][idx] & current_mask) != 0, then there is a b in current_mask.
            if collinear_with[a_idx][idx] & current_mask:
                can_take = False
                break
            m ^= lsb
            
        if can_take:
            dfs(idx + 1, current_mask | (1 << idx), current_size + 1)

    dfs(0, 0, 0)
    return best[0]

def construct():
    random.seed(42)
    # Initialize with a random valid configuration
    def get_initial_points():
        for _ in range(1000):
            pts = []
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                pts.append([x, y])
            if len(set(tuple(p) for p in pts)) != 20:
                continue
            if _check_no_four_collinear(pts):
                return pts
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]

    points = get_initial_points()
    current_3_lines = _count_3_lines(points)
    best_points = [p[:] for p in points]
    best_3_lines = current_3_lines
    
    start_time = time.time()
    time_limit = 110.0
    
    improved = True
    while time.time() - start_time < time_limit:
        improved = False
        # Try to improve by moving one point
        for idx in range(20):
            if time.time() - start_time > time_limit:
                break
            
            orig_x, orig_y = points[idx]
            
            # Try small perturbations
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if abs(new_x) > 100 or abs(new_y) > 100:
                        continue
                    
                    # Create new points list
                    new_points = [p[:] for p in points]
                    new_points[idx] = [new_x, new_y]
                    
                    # Check distinctness
                    if len(set(tuple(p) for p in new_points)) != 20:
                        continue
                    
                    # Check no 4 collinear
                    if not _check_no_four_collinear(new_points):
                        continue
                    
                    new_3_lines = _count_3_lines(new_points)
                    
                    if new_3_lines > current_3_lines:
                        points = new_points
                        current_3_lines = new_3_lines
                        if current_3_lines > best_3_lines:
                            best_3_lines = current_3_lines
                            best_points = [p[:] for p in points]
                        improved = True
                        # Restart loop for this idx with new base
                        break
                if improved:
                    break
            if improved:
                break # Restart outer loop if any point moved? 
                # Actually, standard hill climbing usually restarts the scan.
                # But to be faster, we can just continue. Let's restart to be safe.
                break
        
        if not improved:
            # If no improvement, try a random jump (simulated annealing style)
            # or just break if we are running out of time.
            # Given the time limit, let's do a few random restarts if stuck.
            if time.time() - start_time < time_limit - 10:
                # Perturb a random point significantly
                idx = random.randint(0, 19)
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                new_x = points[idx][0] + dx
                new_y = points[idx][1] + dy
                if abs(new_x) <= 100 and abs(new_y) <= 100:
                    new_points = [p[:] for p in points]
                    new_points[idx] = [new_x, new_y]
                    if len(set(tuple(p) for p in new_points)) == 20 and _check_no_four_collinear(new_points):
                        new_3_lines = _count_3_lines(new_points)
                        # Accept if it's not worse than a threshold or random
                        if new_3_lines >= current_3_lines - 2:
                            points = new_points
                            current_3_lines = new_3_lines
                            if current_3_lines > best_3_lines:
                                best_3_lines = current_3_lines
                                best_points = [p[:] for p in points]
            else:
                break

    # Final exact score check for best_points
    # We return best_points, but the evaluator will compute the score.
    # We just need to return a valid configuration.
    return best_points
