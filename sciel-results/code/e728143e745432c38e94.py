"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Canonical form: first nonzero of (a,b) positive
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    # Check not all collinear
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def _try_move(pts, idx, new_x, new_y):
    """Try moving point idx to (new_x, new_y), return new pts if valid."""
    if new_x == pts[idx][0] and new_y == pts[idx][1]:
        return None
    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
        return None
    # Check duplicate
    for i, p in enumerate(pts):
        if i != idx and p[0] == new_x and p[1] == new_y:
            return None
    new_pts = [list(p) for p in pts]
    new_pts[idx] = [new_x, new_y]
    return new_pts

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try multiple initial configurations and pick the best
    initial_configs = []
    
    # Config 1: 3x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Config 2: Points on 4 lines with many intersections
    # 3 horizontal lines with 4 points each
    initial_configs.append([
        [0,0],[1,0],[2,0],[3,0],
        [0,1],[1,1],[2,1],[3,1],
        [0,2],[1,2],[2,2],[3,2]
    ])
    
    # Config 3: Triangle with points on sides
    # Vertices at (0,0), (12,0), (0,12)
    # Side 1 (y=0): (0,0),(4,0),(8,0),(12,0)
    # Side 2 (x=0): (0,0),(0,4),(0,8),(0,12)
    # Side 3 (x+y=12): (12,0),(8,4),(4,8),(0,12)
    # But vertices are shared, so total = 4+4+4-3 = 9 points, need 3 more
    # Add interior points: (4,4), (8,4) is on line 3... 
    # Let me use: (0,0),(4,0),(8,0),(12,0),(0,4),(0,8),(0,12),(8,4),(4,8)
    # That's 9 points. Add (4,4), (8,8)? No, (8,8) not on any line.
    # Add (4,4), (2,6), (6,2)
    initial_configs.append([
        [0,0],[4,0],[8,0],[12,0],
        [0,4],[0,8],[0,12],
        [8,4],[4,8],
        [4,4],[2,6],[6,2]
    ])
    
    # Config 4: 4 points on each of 3 concurrent lines (through origin)
    # Line 1: y=0: (0,0),(1,0),(2,0),(3,0)
    # Line 2: y=x: (0,0),(1,1),(2,2),(3,3)
    # Line 3: y=2x: (0,0),(1,2),(2,4),(3,6)
    # Shared: (0,0) counted 3 times, so total = 4+4+4-2 = 10 points, need 2 more
    # Add (1,3), (2,1)
    initial_configs.append([
        [0,0],[1,0],[2,0],[3,0],
        [1,1],[2,2],[3,3],
        [1,2],[2,4],[3,6],
        [1,3],[2,1]
    ])
    
    # Config 5: Complete quadrilateral-like
    # 4 lines, each with 3 points, 6 intersection points + 6 other points
    # Lines: y=0, y=1, y=x, y=x+1
    # Intersections: (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,0),(1,1),(2,2),(1,2),(2,3),(3,1)
    # That's messy. Let me try simpler.
    
    # Config 6: 2x6 grid
    initial_configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # Config 7: Points on 2 parallel lines with 6 each
    initial_configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # Find best initial config
    best_pts = None
    best_score = float('inf')
    for config in initial_configs:
        if not _is_valid(config):
            continue
        score = _count_ordinary(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
    
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
    
    # Local search with random restarts
    random.seed(42)
    
    # Phase 1: Random moves with larger steps
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 11)
        orig_x, orig_y = best_pts[idx]
        
        # Try several random moves
        for _ in range(3):
            # Mix of small and large moves
            if random.random() < 0.7:
                step = random.randint(1, 10)
            else:
                step = random.randint(10, 100)
            
            new_x = orig_x + random.randint(-step, step)
            new_y = orig_y + random.randint(-step, step)
            
            new_pts = _try_move(best_pts, idx, new_x, new_y)
            if new_pts is None:
                continue
            
            score = _count_ordinary(new_pts)
            if score < best_score:
                best_pts = new_pts
                best_score = score
                break
    
    # Phase 2: Systematic hill climbing with small moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(12):
            if time.time() > deadline:
                break
            orig_x, orig_y = best_pts[idx]
            for dx in range(-6, 7):
                if time.time() > deadline:
                    break
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = orig_x + dx, orig_y + dy
                    new_pts = _try_move(best_pts, idx, new_x, new_y)
                    if new_pts is None:
                        continue
                    score = _count_ordinary(new_pts)
                    if score < best_score:
                        best_pts = new_pts
                        best_score = score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Phase 3: Try 2-point swaps/moves
    if time.time() < deadline:
        for _ in range(10000):
            if time.time() > deadline:
                break
            i, j = random.sample(range(12), 2)
            # Try swapping
            new_pts = [list(p) for p in best_pts]
            new_pts[i], new_pts[j] = best_pts[j][:], best_pts[i][:]
            if _is_valid(new_pts):
                score = _count_ordinary(new_pts)
                if score < best_score:
                    best_pts = new_pts
                    best_score = score
            # Try moving one point to where the other was, and other to a new spot
            orig_i = best_pts[i][:]
            orig_j = best_pts[j][:]
            new_pts = [list(p) for p in best_pts]
            new_pts[i] = orig_j[:]
            # Move j to a random nearby spot
            step = random.randint(1, 20)
            new_pts[j] = [orig_j[0] + random.randint(-step, step), orig_j[1] + random.randint(-step, step)]
            if _is_valid(new_pts):
                score = _count_ordinary(new_pts)
                if score < best_score:
                    best_pts = new_pts
                    best_score = score
    
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
