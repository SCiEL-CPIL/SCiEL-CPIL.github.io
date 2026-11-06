"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def _compute_score(points):
    """Compute the max multiplicity of a single positive doubled area among all triples."""
    n = len(points)
    if n < 3:
        return 0
    areas = []
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                # doubled area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                d = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _inbounds(p):
    x, y = p
    return -30 <= x <= 30 and -30 <= y <= 30

def _build_grid():
    """Build a 6x5 grid centered at origin."""
    pts = []
    for x in range(-2, 4):
        for y in range(-2, 3):
            pts.append((x, y))
    return pts

def _local_improve(points, deadline, max_iter=50000):
    """Hill climbing: try moving one point to a random nearby position."""
    current = list(points)
    best_score = _compute_score(current)
    n = len(current)
    step = 0
    while time.time() < deadline:
        step += 1
        if step % 100 == 0 and time.time() > deadline - 5:
            break
        # Pick a random point and a random small perturbation
        i = random.randint(0, n-1)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
        new_x = current[i][0] + dx
        new_y = current[i][1] + dy
        if not _inbounds((new_x, new_y)):
            continue
        # Check if this creates a duplicate
        new_pt = (new_x, new_y)
        if new_pt in current and new_pt != current[i]:
            continue
        trial = current[:]
        trial[i] = new_pt
        sc = _compute_score(trial)
        if sc > best_score:
            current = trial
            best_score = sc
    return current, best_score

def construct():
    start = time.time()
    deadline = start + 80  # 80s deadline, 10s safety margin
    
    # Initial grid
    points = _build_grid()
    best_points = points[:]
    best_score = _compute_score(points)
    
    # Try multiple restarts with different initial configurations
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        if attempts == 1:
            pts = _build_grid()
        elif attempts == 2:
            # Slightly sheared grid
            pts = []
            for x in range(-2, 4):
                for y in range(-2, 3):
                    pts.append((x, y + (1 if x % 2 == 0 else 0)))
        elif attempts == 3:
            # Another variant
            pts = []
            for x in range(-2, 4):
                for y in range(-2, 3):
                    pts.append((x + (1 if y % 2 == 0 else 0), y))
        else:
            # Random perturbation of best
            pts = best_points[:]
            for i in range(len(pts)):
                if random.random() < 0.3:
                    dx = random.randint(-2, 2)
                    dy = random.randint(-2, 2)
                    nx, ny = pts[i][0]+dx, pts[i][1]+dy
                    if _inbounds((nx, ny)) and (nx, ny) not in pts:
                        pts[i] = (nx, ny)
        
        # Ensure 30 distinct points in bounds
        seen = set()
        valid_pts = []
        for p in pts:
            if p not in seen and _inbounds(p):
                seen.add(p)
                valid_pts.append(p)
        if len(valid_pts) != 30:
            continue
        
        improved_pts, sc = _local_improve(valid_pts, deadline - 5)
        if sc > best_score:
            best_score = sc
            best_points = improved_pts[:]
    
    return best_points
# EVOLVE-BLOCK-END
