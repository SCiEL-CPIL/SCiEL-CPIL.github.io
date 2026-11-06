"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

def _line_key(p1, p2):
    """Return canonical key for the line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        g = 1
    dx //= g
    dy //= g
    # Canonical direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: y - y1 = (dy/dx)(x - x1) if dx != 0, else x = x1
    # Use (a, b, c) for ax + by + c = 0
    # a = dy, b = -dx, c = dx*y1 - dy*x1
    a = dy
    b = -dx
    c = dx * y1 - dy * x1
    # Normalize sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _count_lines(points):
    """Count lines with exactly 4 points and check for 5+ collinear.
    Returns (num_4_lines, max_collinear)."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    # For each line, count how many points are on it
    num_4 = 0
    max_collinear = 0
    for key, pair_count in line_counts.items():
        # pair_count = C(k, 2) where k is number of points on line
        # k*(k-1)/2 = pair_count
        # Solve for k
        # k = (1 + sqrt(1 + 8*pair_count)) / 2
        disc = 1 + 8 * pair_count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
        else:
            # Should not happen for valid line keys
            continue
        if k == 4:
            num_4 += 1
        if k > max_collinear:
            max_collinear = k
    return num_4, max_collinear

def _score(points):
    """Score: number of 4-point lines, with heavy penalty for 5+ collinear."""
    n4, max_c = _count_lines(points)
    if max_c >= 5:
        return n4 - 10000 * (max_c - 4)
    return n4

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(best_points)
    
    # Also try some other initial configurations
    init_configs = []
    # 4x4 grid
    init_configs.append([[x, y] for x in range(4) for y in range(4)])
    # 4x4 grid shifted
    init_configs.append([[x, y] for x in range(1, 5) for y in range(1, 5)])
    
    # Try a few random initial configurations
    for _ in range(10):
        pts = []
        while len(pts) < 16:
            p = [random.randint(-50, 50), random.randint(-50, 50)]
            if p not in pts:
                pts.append(p)
        init_configs.append(pts)
    
    best_score = -1
    for cfg in init_configs:
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in cfg]
    
    # Simulated annealing
    current = [p[:] for p in best_points]
    current_score = best_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Move: pick a random point and perturb it
        idx = random.randint(0, 15)
        old_x, old_y = current[idx]
        # Random perturbation
        new_x = old_x + random.randint(-3, 3)
        new_y = old_y + random.randint(-3, 3)
        # Clamp
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))
        
        # Check if point already exists
        if [new_x, new_y] in current:
            T *= cooling
            continue
        
        current[idx] = [new_x, new_y]
        new_score = _score(current)
        
        if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current]
        else:
            current[idx] = [old_x, old_y]
        
        T *= cooling
    
    # Local refinement: try all single-point moves
    for _ in range(50):  # iterations of local search
        improved = False
        for idx in range(16):
            old_x, old_y = best_points[idx]
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx = old_x + dx
                    ny = old_y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    if [nx, ny] in best_points:
                        continue
                    best_points[idx] = [nx, ny]
                    s = _score(best_points)
                    if s > best_score:
                        best_score = s
                        improved = True
                    else:
                        best_points[idx] = [old_x, old_y]
                    if time.time() > deadline:
                        return [[int(x), int(y)] for x, y in best_points]
        if not improved:
            break
    
    return [[int(x), int(y)] for x, y in best_points]
# EVOLVE-BLOCK-END
