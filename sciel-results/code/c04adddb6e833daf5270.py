import time
import random
from itertools import combinations
from math import gcd

def _count_exact_four(pts):
    """Count lines containing exactly 4 of the given points."""
    n = len(pts)
    count = 0
    # Use set of normalized line representations to avoid double counting
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Normalize direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line equation: dy*x - dx*y = c
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    for key, idx_set in line_map.items():
        if len(idx_set) == 4:
            count += 1
    return count

def _score(pts):
    """Score: number of exact-4 lines, with penalty if any 5+ collinear."""
    n = len(pts)
    # Check no five collinear
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    for key, idx_set in line_map.items():
        if len(idx_set) >= 5:
            return -1  # invalid
    
    return _count_exact_four(pts)

def _perturb(pts, limit=1000):
    """Randomly perturb a few points slightly."""
    new_pts = [p[:] for p in pts]
    for _ in range(random.randint(1, 3)):
        idx = random.randint(0, len(new_pts)-1)
        new_pts[idx][0] += random.randint(-5, 5)
        new_pts[idx][1] += random.randint(-5, 5)
        # Clamp
        new_pts[idx][0] = max(-limit, min(limit, new_pts[idx][0]))
        new_pts[idx][1] = max(-limit, min(limit, new_pts[idx][1]))
    return new_pts

def _random_grid_like(size, offset_x=0, offset_y=0, limit=1000):
    """Generate a slightly perturbed grid."""
    pts = []
    for x in range(size):
        for y in range(size):
            px = x + offset_x + random.randint(-2, 2)
            py = y + offset_y + random.randint(-2, 2)
            if -limit <= px <= limit and -limit <= py <= limit:
                pts.append([px, py])
    return pts[:16]

def _evaluate_and_refine(pts, deadline):
    """Local search refinement."""
    best = pts[:]
    best_score = _score(pts)
    if best_score < 0:
        return best, best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(pts)):
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in pts]
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    if abs(new_pts[i][0]) > 1000 or abs(new_pts[i][1]) > 1000:
                        continue
                    # Check distinctness
                    if len(set(map(tuple, new_pts))) != len(new_pts):
                        continue
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best = new_pts[:]
                        pts = new_pts[:]
                        improved = True
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    best_pts = None
    best_score = -1
    
    # Start with 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    s = _score(grid)
    if s > best_score:
        best_score = s
        best_pts = grid[:]
    
    # Try various initial configurations
    random.seed(42)
    
    configs = []
    
    # Perturbed grids
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        pts = _random_grid_like(4)
        if len(pts) == 16 and len(set(map(tuple, pts))) == 16:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts[:]
    
    # Local search from best
    if best_pts and time.time() < deadline - 20:
        refined, rs = _evaluate_and_refine(best_pts, deadline)
        if rs > best_score:
            best_score = rs
            best_pts = refined[:]
    
    # Try more diverse starting points
    for trial in range(100):
        if time.time() > deadline - 15:
            break
        pts = []
        for _ in range(16):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            pts.append([x, y])
        if len(set(map(tuple, pts))) != 16:
            continue
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts[:]
            if time.time() < deadline - 20:
                refined, rs = _evaluate_and_refine(pts, deadline)
                if rs > best_score:
                    best_score = rs
                    best_pts = refined[:]
    
    # Final refinement
    if best_pts and time.time() < deadline - 5:
        refined, rs = _evaluate_and_refine(best_pts, deadline)
        if rs > best_score:
            best_score = rs
            best_pts = refined[:]
    
    # Ensure validity
    if best_pts is None:
        return [[x, y] for x in range(4) for y in range(4)]
    
    # Final check
    if _score(best_pts) < 0:
        return [[x, y] for x in range(4) for y in range(4)]
    
    return best_pts

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
