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
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Baseline: 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in grid]
    best_score = count_4_lines_fast(best_points)
    
    random.seed(42)
    
    # List of move ranges to try
    # Small, Medium, Large
    move_ranges = [
        [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-10, 11) for dy in range(-10, 11) if (dx, dy) != (0, 0)]
    ]
    
    # Add some random large moves
    large_moves = [(random.randint(-50, 50), random.randint(-50, 50)) for _ in range(20)]
    move_ranges.append(large_moves)
    
    def local_search(current, current_score):
        nonlocal best_points, best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(16):
                if time.time() > deadline:
                    return
                old = current[idx][:]
                
                # Try moves from smallest to largest
                for moves in move_ranges:
                    for dx, dy in moves:
                        if time.time() > deadline:
                            return
                        nx, ny = old[0] + dx, old[1] + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        
                        current[idx] = [nx, ny]
                        
                        # Check distinct
                        if len(set(tuple(p) for p in current)) < 16:
                            current[idx] = old
                            continue
                        
                        s = count_4_lines_fast(current)
                        if s > current_score:
                            current_score = s
                            improved = True
                            if s > best_score:
                                best_score = s
                                best_points = [p[:] for p in current]
                            # If we improved, break out of move loops for this index
                            # to re-evaluate from the new state
                            current[idx] = old # Restore to try other moves? No, keep the improvement
                            # Actually, if we accept the move, we should break and restart the loop
                            # or continue with the new state.
                            # Standard hill climbing: accept and break to restart scan
                            return 
                        else:
                            current[idx] = old
        
    # Phase 1: Local search from grid
    current = [p[:] for p in grid]
    current_score = count_4_lines_fast(current)
    if current_score > best_score:
        best_score = current_score
        best_points = [p[:] for p in current]
    local_search(current, current_score)
    
    # Phase 2: Random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Generate a new random configuration
        # Mix of random and perturbed best
        if restart % 5 == 0:
            # Perturb best
            current = [p[:] for p in best_points]
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, 15)
                current[idx] = [current[idx][0] + random.randint(-20, 20),
                               current[idx][1] + random.randint(-20, 20)]
        else:
            # Random points in a box
            # Use a box that is large enough to allow diverse lines
            # but small enough to keep coordinates reasonable
            size = random.choice([10, 20, 50])
            current = [[random.randint(-size, size), random.randint(-size, size)] for _ in range(16)]
            
        # Ensure distinct
        attempts = 0
        while len(set(tuple(p) for p in current)) < 16 and attempts < 100:
            idx = random.randint(0, 15)
            current[idx] = [random.randint(-50, 50), random.randint(-50, 50)]
            attempts += 1
        
        if len(set(tuple(p) for p in current)) < 16:
            continue
            
        current_score = count_4_lines_fast(current)
        if current_score < 0:
            continue
            
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current]
            
        # Local search
        local_search(current, current_score)
        
    return best_points
# EVOLVE-BLOCK-END
