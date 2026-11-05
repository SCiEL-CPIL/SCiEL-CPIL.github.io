"""Search-based program for maximizing 4-point lines among 16 integer points."""
import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 150.0  # wall-clock limit
_SAFETY = 5.0      # safety margin
_DEADLINE_EFF = _START_TIME + _DEADLINE - _SAFETY


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _count_4lines_and_check(points):
    """Returns (count_of_4lines, is_valid)."""
    n = len(points)
    if n != 16:
        return 0, False
    
    # Check bounds and distinctness
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return 0, False
        if (x, y) in seen:
            return 0, False
        seen.add((x, y))
        
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx, dy = x2 - x1, y2 - y1
            g = _gcd(abs(dx), abs(dy))
            if g:
                dx, dy = dx // g, dy // g
            else:
                continue
            
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            
            if key in line_points:
                line_points[key].add(i)
                line_points[key].add(j)
            else:
                line_points[key] = {i, j}
                
    count = 0
    for pts_on_line in line_points.values():
        if len(pts_on_line) >= 5:
            return 0, False
        if len(pts_on_line) == 4:
            count += 1
            
    return count, True


def _score(points):
    c, valid = _count_4lines_and_check(points)
    return c if valid else 0


def _grid44():
    return [[x, y] for x in range(4) for y in range(4)]


def _affine_transform(points, a, b, c, d, tx, ty):
    """Apply affine transform (x,y) -> (a*x + b*y + tx, c*x + d*y + ty)."""
    out = []
    for x, y in points:
        nx = a * x + b * y + tx
        ny = c * x + d * y + ty
        out.append([nx, ny])
    return out


def _local_search(points, rng, max_iters=500):
    """Local search: move random points to random nearby positions."""
    pts = [list(p) for p in points]
    best = [list(p) for p in pts]
    best_score = _score(best)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE_EFF:
            break
        
        # Choose a move type
        if rng.random() < 0.5:
            # Single point move
            i = rng.randrange(16)
            old = list(pts[i])
            # Random jump, sometimes small, sometimes larger
            radius = rng.choice([1, 2, 3, 5, 10])
            dx = rng.randint(-radius, radius)
            dy = rng.randint(-radius, radius)
            nx = pts[i][0] + dx
            ny = pts[i][1] + dy
            
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            
            # Check distinct
            dup = False
            for k in range(16):
                if k != i and pts[k][0] == nx and pts[k][1] == ny:
                    dup = True
                    break
            if dup:
                continue
                
            pts[i] = [nx, ny]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best = [list(p) for p in pts]
            else:
                pts[i] = old
        else:
            # Swap two points? Or move two points?
            # Let's try swapping coordinates or moving two points
            i, j = rng.sample(range(16), 2)
            old_i = list(pts[i])
            old_j = list(pts[j])
            
            radius = rng.choice([1, 2, 3])
            dx1 = rng.randint(-radius, radius)
            dy1 = rng.randint(-radius, radius)
            dx2 = rng.randint(-radius, radius)
            dy2 = rng.randint(-radius, radius)
            
            nx1 = pts[i][0] + dx1
            ny1 = pts[i][1] + dy1
            nx2 = pts[j][0] + dx2
            ny2 = pts[j][1] + dy2
            
            if abs(nx1) > 1000 or abs(ny1) > 1000 or abs(nx2) > 1000 or abs(ny2) > 1000:
                continue
            
            # Check distinctness including cross
            if (nx1 == nx2 and ny1 == ny2):
                continue
            
            dup = False
            for k in range(16):
                if k != i and k != j:
                    if (pts[k][0] == nx1 and pts[k][1] == ny1) or (pts[k][0] == nx2 and pts[k][1] == ny2):
                        dup = True
                        break
            if dup:
                continue
                
            pts[i] = [nx1, ny1]
            pts[j] = [nx2, ny2]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best = [list(p) for p in pts]
            else:
                pts[i] = old_i
                pts[j] = old_j
                
    return best, best_score


def construct():
    global _START_TIME, _DEADLINE_EFF
    _START_TIME = time.time()
    _DEADLINE_EFF = _START_TIME + _DEADLINE - _SAFETY

    best_pts = _grid44()
    best_score = _score(best_pts)

    # 1. Try affine transforms of 4x4 grid
    rng = random.Random(42)
    for trial in range(5000):
        if time.time() > _DEADLINE_EFF:
            break
        # Random affine transform
        a = rng.randint(-3, 3)
        b = rng.randint(-3, 3)
        c = rng.randint(-3, 3)
        d = rng.randint(-3, 3)
        if a * d - b * c == 0:
            continue
        tx = rng.randint(-20, 20)
        ty = rng.randint(-20, 20)
        
        pts = _affine_transform(_grid44(), a, b, c, d, tx, ty)
        s, valid = _count_4lines_and_check(pts)
        if valid and s > best_score:
            best_score = s
            best_pts = pts
            # Local search immediately
            best_pts, best_score = _local_search(best_pts, rng, max_iters=100)
            if time.time() > _DEADLINE_EFF:
                break

    # 2. Random restarts with local search
    for seed in range(1000):
        if time.time() > _DEADLINE_EFF:
            break
            
        # Generate a random point cloud
        pts = []
        attempts = 0
        while len(pts) < 16 and attempts < 500:
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            p = [x, y]
            if p not in pts:
                pts.append(p)
            attempts += 1
            
        if len(pts) < 16:
            continue
            
        s, valid = _count_4lines_and_check(pts)
        if valid:
            # Refine
            pts, s = _local_search(pts, rng, max_iters=100)
            if s > best_score:
                best_score = s
                best_pts = pts

    # 3. Final local search on best found
    if time.time() < _DEADLINE_EFF - 10:
        best_pts, best_score = _local_search(best_pts, rng, max_iters=500)

    return [[int(x), int(y)] for x, y in best_pts]
