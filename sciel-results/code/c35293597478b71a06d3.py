"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_4_lines(points):
    n = len(points)
    lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            dx, dy, c = key
            count = 0
            for k in range(n):
                x, y = points[k]
                if dy * x - dx * y == c:
                    count += 1
            if count == 4:
                lines.add(key)
    return len(lines)

def _is_valid(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            dx, dy, c = key
            count = 0
            for k in range(n):
                x, y = points[k]
                if dy * x - dx * y == c:
                    count += 1
            if count >= 5:
                return False
    return True

def construct():
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in points]
    best_score = _count_4_lines(best_points)
    
    start_time = time.time()
    limit = 110.0 # Safety margin
    
    # Fast local search
    # We will try to move points to improve score
    # To speed up, we only check validity and score for the new point's lines
    
    while time.time() - start_time < limit:
        improved = False
        for idx in range(16):
            if time.time() - start_time > limit - 5.0:
                break
            x, y = points[idx]
            # Try nearby moves
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    
                    # Check collision
                    collision = False
                    for i in range(16):
                        if i == idx:
                            continue
                        if points[i][0] == nx and points[i][1] == ny:
                            collision = True
                            break
                    if collision:
                        continue
                    
                    new_points = [p[:] for p in points]
                    new_points[idx] = [nx, ny]
                    
                    # Check validity (no 5 collinear)
                    # Only lines through the moved point can be affected
                    valid = True
                    for i in range(16):
                        if i == idx:
                            continue
                        key = _line_key(new_points[idx], new_points[i])
                        if key is None:
                            continue
                        dx_l, dy_l, c = key
                        count = 0
                        for j in range(16):
                            x_j, y_j = new_points[j]
                            if dy_l * x_j - dx_l * y_j == c:
                                count += 1
                        if count >= 5:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    score = _count_4_lines(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try a random jump
            if random.random() < 0.1:
                idx = random.randint(0, 15)
                nx = random.randint(-10, 10)
                ny = random.randint(-10, 10)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    collision = False
                    for i in range(16):
                        if i == idx:
                            continue
                        if points[i][0] == nx and points[i][1] == ny:
                            collision = True
                            break
                    if not collision:
                        new_points = [p[:] for p in points]
                        new_points[idx] = [nx, ny]
                        if _is_valid(new_points):
                            score = _count_4_lines(new_points)
                            if score > best_score:
                                best_score = score
                                best_points = new_points
                                points = new_points
                            else:
                                # Accept with some probability? No, greedy is safer for time
                                pass
            else:
                # If no improvement, break to save time?
                # Or continue searching?
                # Given the small search space, it might be stuck.
                # Let's break to return best found.
                break
                
    return best_points
# EVOLVE-BLOCK-END
