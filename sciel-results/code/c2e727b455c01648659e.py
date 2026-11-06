import time
import random
from itertools import combinations
from collections import Counter

def _doubled_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def _score(points):
    areas = Counter()
    # Use indices for speed
    n = len(points)
    for i in range(n):
        p1 = points[i]
        for j in range(i+1, n):
            p2 = points[j]
            for k in range(j+1, n):
                p3 = points[k]
                d = _doubled_area(p1, p2, p3)
                if d > 0:
                    areas[d] += 1
    if not areas:
        return 0
    return max(areas.values())

def _make_grid(m, n, x0=0, y0=0):
    """m columns, n rows, starting at (x0,y0)"""
    return [(x0 + i, y0 + j) for i in range(m) for j in range(n)]

def construct():
    start = time.time()
    deadline = start + 85  # 5s safety margin
    
    best_score = 0
    best_points = None
    
    # 1. Initialize with best rectangular grid
    # Translation doesn't change area multiplicities, so we only need one representative per shape.
    # We center them to stay in bounds.
    factors = [(1,30),(2,15),(3,10),(5,6),(6,5),(10,3),(15,2),(30,1)]
    
    for (m, n) in factors:
        # Center the grid
        x0 = -m // 2
        y0 = -n // 2
        pts = _make_grid(m, n, x0, y0)
        
        # Check bounds
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        
        if time.time() > deadline:
            break
            
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score(best_points)
    
    # 2. Local Search (Hill Climbing)
    # We try to improve the best grid found.
    # Move strategy: Random swap or random move to a nearby position.
    # To ensure distinctness, we maintain a set of points.
    
    current_points = best_points.copy()
    current_score = best_score
    current_set = set(current_points)
    
    # Pre-compute valid moves? No, just check on the fly.
    
    # We will use a simple random restart / hill climbing hybrid.
    # Since the score is integer and rugged, we might get stuck.
    # We'll do a random walk with acceptance if score improves.
    
    while time.time() < deadline:
        # Choose a random point to move
        idx = random.randint(0, 29)
        old_pt = current_points[idx]
        
        # Generate a new position
        # Mix of local moves (shift) and global moves (random)
        if random.random() < 0.5:
            # Local move
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            new_pt = (old_pt[0] + dx, old_pt[1] + dy)
        else:
            # Global move
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
        
        # Check bounds and distinctness
        if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
            continue
        if new_pt == old_pt:
            continue
        if new_pt in current_set:
            continue
            
        # Apply move
        current_points[idx] = new_pt
        current_set.remove(old_pt)
        current_set.add(new_pt)
        
        # Calculate new score
        new_score = _score(current_points)
        
        # Accept or reject
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Reject (Revert)
            current_points[idx] = old_pt
            current_set.remove(new_pt)
            current_set.add(old_pt)
            
    return best_points
