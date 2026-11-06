import time
import random
from itertools import combinations

_DEADLINE = 115.0

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _valid_no_four(points):
    n = len(points)
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                return False
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
            if slopes[(dx, dy)] >= 3:
                return False
    return True

def _build_pair_conflict_graph(points):
    """
    Build graph where edge (i,j) exists if there is a k such that i,j,k are collinear.
    """
    n = len(points)
    conflict = [0] * n
    # For each pair (i,j), check if there is a k collinear with them.
    # This is O(N^3). N=20, so 8000 ops. Fast.
    for i in range(n):
        for j in range(i + 1, n):
            # Check all k != i, j
            collinear = False
            for k in range(n):
                if k == i or k == j:
                    continue
                # Check if i, j, k are collinear
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    collinear = True
                    break
            if collinear:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    return conflict

def _max_independent_set(conflict, n):
    """
    Find MIS size in the conflict graph.
    """
    best = [0]
    all_mask = (1 << n) - 1
    
    def dfs(available, count):
        if count + bin(available).count('1') <= best[0]:
            return
        if available == 0:
            if count > best[0]:
                best[0] = count
            return
        
        # Pick a vertex with max degree in available subgraph?
        # Or just first one.
        v = None
        # Find first set bit
        for i in range(n):
            if available & (1 << i):
                v = i
                break
        if v is None:
            return
        
        # Exclude v
        dfs(available & ~(1 << v), count)
        
        # Include v
        new_avail = available & ~(1 << v) & ~conflict[v]
        dfs(new_avail, count + 1)
        
    dfs(all_mask, 0)
    return best[0]

def _score(points):
    conflict = _build_pair_conflict_graph(points)
    return _max_independent_set(conflict, len(points))

def construct():
    start = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = 21
    
    def consider(pts):
        nonlocal best_pts, best_score
        if not _valid_no_four(pts):
            return
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_pts = list(pts)
            if best_score <= 12:
                return True
        return False

    # Initial candidates
    # 5x4 grid
    base = [(x, y) for x in range(5) for y in range(4)]
    
    # Try some structured perturbations
    # Shift rows by x
    for shift_type in range(5):
        pts = []
        for y in range(4):
            for x in range(5):
                # Shift x by y? Or shift y by x?
                if shift_type == 0:
                    pts.append((x, y))
                elif shift_type == 1:
                    pts.append((x, y + x))
                elif shift_type == 2:
                    pts.append((x + y, y))
                elif shift_type == 3:
                    pts.append((x, y + (x % 2) * 2))
                elif shift_type == 4:
                    pts.append((x + (y % 2) * 2, y))
        if consider(pts):
            return [list(p) for p in best_pts]

    # Random search around 5x4 grid
    for attempt in range(20000):
        if time.time() - start > _DEADLINE:
            break
        pts = list(base)
        for _ in range(rng.randint(3, 10)):
            idx = rng.randrange(20)
            dx = rng.randint(-4, 4)
            dy = rng.randint(-4, 4)
            x, y = pts[idx]
            nx, ny = x + dx, y + dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                pts[idx] = (nx, ny)
        if consider(pts):
            break

    # Local search from best
    if best_pts:
        current_pts = list(best_pts)
        current_score = best_score
        
        for step in range(50000):
            if time.time() - start > _DEADLINE:
                break
            
            idx = rng.randrange(20)
            x, y = current_pts[idx]
            moved = False
            for _ in range(10):
                nx = x + rng.randint(-3, 3)
                ny = y + rng.randint(-3, 3)
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            new_score = _score(current_pts)
                            if new_score < current_score:
                                current_score = new_score
                                moved = True
                                if current_score < best_score:
                                    best_score = current_score
                                    best_pts = list(current_pts)
                                    if best_score <= 12:
                                        break
                                break
                            elif new_score == current_score and rng.random() < 0.1:
                                moved = True
                                break
                        current_pts[idx] = old_pt
            if best_score <= 12:
                break

    if best_pts is None:
        return [[x, x * x] for x in range(-10, 10)]
    
    return [list(p) for p in best_pts]
