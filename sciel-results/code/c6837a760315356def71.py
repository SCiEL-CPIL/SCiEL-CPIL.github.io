"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def normalize_line(p1, p2):
    """Return canonical line key from two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 1, x1)
    if dy == 0:
        return (1, 0, y1)
    g = gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    c = dy_n * x1 - dx_n * y1
    return (dx_n, dy_n, c)

def count_4_lines_fast(points):
    """Fast count of lines with exactly 4 points. Returns -1 if any 5 collinear."""
    n = len(points)
    if n != 16:
        return 0
    
    # Use a dict mapping line key -> bitmask of points on that line
    line_mask = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = normalize_line(points[i], points[j])
            if key is None:
                continue
            mask = (1 << i) | (1 << j)
            if key in line_mask:
                line_mask[key] |= mask
            else:
                line_mask[key] = mask
    
    count4 = 0
    for key, mask in line_mask.items():
        popcount = bin(mask).count('1')
        if popcount >= 5:
            return -1
        if popcount == 4:
            count4 += 1
    return count4

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Start with the 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in grid]
    best_score = count_4_lines_fast(best_points)
    
    random.seed(12345)
    
    # Strategy 1: Local search from grid with structured moves
    current = [p[:] for p in grid]
    current_score = best_score
    
    # Precompute which lines each point is on for faster updates
    # But for simplicity, we'll use the fast counter
    
    # Phase 1: Try all single-point moves from grid
    for idx in range(16):
        if time.time() > deadline:
            return best_points
        old = current[idx][:]
        # Try moves in a structured pattern
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = old[0] + dx, old[1] + dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                current[idx] = [nx, ny]
                # Check distinct
                pts_set = set(tuple(p) for p in current)
                if len(pts_set) < 16:
                    current[idx] = old
                    continue
                s = count_4_lines_fast(current)
                if s > current_score:
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_points = [p[:] for p in current]
                else:
                    current[idx] = old
    
    # Phase 2: Iterated local search with random restarts
    for restart in range(50):
        if time.time() > deadline:
            return best_points
        
        # Perturb from best or random
        if restart % 3 == 0:
            current = [p[:] for p in best_points]
            # Make 1-3 random small changes
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 15)
                current[idx] = [current[idx][0] + random.randint(-3, 3),
                               current[idx][1] + random.randint(-3, 3)]
        else:
            # Random restart near grid
            current = []
            for x in range(4):
                for y in range(4):
                    current.append([x + random.randint(-2, 2), y + random.randint(-2, 2)])
        
        # Ensure distinct
        for _ in range(100):
            if len(set(tuple(p) for p in current)) == 16:
                break
            idx = random.randint(0, 15)
            current[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
        
        current_score = count_4_lines_fast(current)
        if current_score < 0:
            continue
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current]
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(16):
                if time.time() > deadline:
                    return best_points
                old = current[idx][:]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = old[0] + dx, old[1] + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        current[idx] = [nx, ny]
                        pts_set = set(tuple(p) for p in current)
                        if len(pts_set) < 16:
                            current[idx] = old
                            continue
                        s = count_4_lines_fast(current)
                        if s > current_score:
                            current_score = s
                            improved = True
                            if s > best_score:
                                best_score = s
                                best_points = [p[:] for p in current]
                        else:
                            current[idx] = old
    
    return best_points
# EVOLVE-BLOCK-END
