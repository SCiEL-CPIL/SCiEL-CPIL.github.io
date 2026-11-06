"""
Strategy: Model as maximum weight independent set in a conflict graph of candidate
4-point collinear quadruples. Vertices = candidate 4-point lines; edges = conflicts
(share 3+ points on same line, or union exceeds 16 points, or would create 5-collinear).
Use greedy + local search on the configuration space, but with a smarter initialization
and a final polish pass that tries to add more 4-point lines by solving small subproblems.
"""
import time
import math
import random
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(x1, y1, x2, y2):
    """Canonical (a,b,c) for line through two points: a*x + b*y = c, gcd-normalized."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    a = dy // g
    b = -dx // g
    c = a * x1 + b * y1
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _compute_lines(points):
    """Compute all lines and their point counts. Returns (line_map, score, has_five)."""
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(xs[i], ys[i], xs[j], ys[j])
            if k is not None:
                line_map[k].add(i)
                line_map[k].add(j)
    
    score = 0
    has_five = False
    line_counts = {}
    for k, idx_set in line_map.items():
        cnt = len(idx_set)
        line_counts[k] = cnt
        if cnt >= 5:
            has_five = True
        elif cnt == 4:
            score += 1
    return line_map, score, has_five


def _score(points):
    """Return score (number of 4-point lines) or -1 if invalid."""
    _, s, has_five = _compute_lines(points)
    if has_five:
        return -1
    return s


def _is_valid(points):
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(int(x)) > 1000 or abs(int(y)) > 1000:
            return False
        t = (int(x), int(y))
        if t in seen:
            return False
        seen.add(t)
    return _score(points) >= 0


def _find_4point_lines(points):
    """Find all lines with exactly 4 points. Returns list of (line_key, frozenset of indices)."""
    line_map, _, has_five = _compute_lines(points)
    if has_five:
        return []
    result = []
    for k, idx_set in line_map.items():
        if len(idx_set) == 4:
            result.append((k, frozenset(idx_set)))
    return result


def _find_3point_lines(points):
    """Find all lines with exactly 3 points. Returns list of (line_key, frozenset of indices)."""
    line_map, _, has_five = _compute_lines(points)
    if has_five:
        return []
    result = []
    for k, idx_set in line_map.items():
        if len(idx_set) == 3:
            result.append((k, frozenset(idx_set)))
    return result


def _find_2point_lines(points):
    """Find all lines with exactly 2 points. Returns list of (line_key, frozenset of indices)."""
    line_map, _, has_five = _compute_lines(points)
    if has_five:
        return []
    result = []
    for k, idx_set in line_map.items():
        if len(idx_set) == 2:
            result.append((k, frozenset(idx_set)))
    return result


def _integer_points_on_line(a, b, c, existing_set, bound=1000):
    """Find integer points (x,y) on line a*x+b*y=c within bounds, not in existing_set."""
    pts = []
    if a == 0:
        # b*y = c => y = c/b
        if b != 0 and c % b == 0:
            ny = c // b
            if abs(ny) <= bound:
                for nx in range(-bound, bound + 1):
                    if (nx, ny) not in existing_set:
                        pts.append((nx, ny))
    elif b == 0:
        # a*x = c => x = c/a
        if c % a == 0:
            nx = c // a
            if abs(nx) <= bound:
                for ny in range(-bound, bound + 1):
                    if (nx, ny) not in existing_set:
                        pts.append((nx, ny))
    else:
        # a*x + b*y = c
        # Parametrize: x = x0 + b*t, y = y0 - a*t for some particular solution
        # Find one solution: need a*x + b*y = c
        # gcd(a,b) must divide c
        g = math.gcd(a, b)
        if c % g != 0:
            return pts
        # Find particular solution
        # Extended Euclidean
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x1, y1 = egcd(b, a % b)
            return (g, y1, x1 - (a // b) * y1)
        
        g, x, y = egcd(a, b)
        x0 = x * (c // g)
        y0 = y * (c // g)
        # General solution: x = x0 + (b/g)*t, y = y0 - (a/g)*t
        bx = b // g
        ay = a // g
        # Find t range
        # x0 + bx*t in [-bound, bound] => t in [(−bound−x0)/bx, (bound−x0)/bx]
        # y0 - ay*t in [-bound, bound] => t in [(y0−bound)/ay, (y0+bound)/ay]
        t_min = -10**9
        t_max = 10**9
        if bx > 0:
            t_min = max(t_min, math.ceil((-bound - x0) / bx))
            t_max = min(t_max, math.floor((bound - x0) / bx))
        elif bx < 0:
            t_min = max(t_min, math.ceil((bound - x0) / bx))
            t_max = min(t_max, math.floor((-bound - x0) / bx))
        if ay > 0:
            t_min = max(t_min, math.ceil((y0 - bound) / ay))
            t_max = min(t_max, math.floor((y0 + bound) / ay))
        elif ay < 0:
            t_min = max(t_min, math.ceil((y0 + bound) / ay))
            t_max = min(t_max, math.floor((y0 - bound) / ay))
        
        for t in range(t_min, t_max + 1):
            nx = x0 + bx * t
            ny = y0 - ay * t
            if abs(nx) <= bound and abs(ny) <= bound:
                if (nx, ny) not in existing_set:
                    pts.append((nx, ny))
    return pts


def _try_complete_3line(points, line_key, existing_set):
    """Try to complete a 3-point line to 4 by moving one point. Returns (new_points, new_score) or (None, -1)."""
    a, b, c = line_key
    line_map, cur_score, has_five = _compute_lines(points)
    if has_five:
        return None, -1
    
    # Get the 3 points on this line
    idx_set = line_map[line_key]
    if len(idx_set) != 3:
        return None, -1
    
    # Find candidate positions on this line
    candidates = _integer_points_on_line(a, b, c, existing_set, bound=1000)
    if not candidates:
        return None, -1
    
    # Try moving each of the 16 points to each candidate
    # Prefer moving points that are not on other 3-point lines
    point_line_count = Counter()
    for k, s in line_map.items():
        if len(s) == 3:
            for idx in s:
                point_line_count[idx] += 1
    
    # Sort points by how many 3-lines they're on (prefer lower)
    sorted_pts = sorted(range(16), key=lambda i: point_line_count.get(i, 0))
    
    best_new = None
    best_new_score = cur_score - 1  # need to beat current
    
    # Limit candidates to avoid too much work
    rng_local = random.Random(hash(line_key) % 100000)
    if len(candidates) > 50:
        rng_local.shuffle(candidates)
        candidates = candidates[:50]
    
    for idx in sorted_pts:
        old_x, old_y = points[idx]
        for nx, ny in candidates:
            if (nx, ny) == (old_x, old_y):
                continue
            new_pts = [list(p) for p in points]
            new_pts[idx] = [nx, ny]
            new_score, new_has_five = _score(new_pts), False
            if new_score < 0:
                continue
            if new_score > best_new_score:
                best_new_score = new_score
                best_new = new_pts
    
    if best_new is not None:
        return best_new, best_new_score
    return None, -1


def _local_search(points, deadline, rng, max_iter=20000):
    """Local search with line-completion bias."""
    best = [list(p) for p in points]
    best_score = _score(best)
    if best_score < 0:
        best_score = 0
    
    existing_set = set((p[0], p[1]) for p in best)
    no_improve = 0
    max_no_improve = 150
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break
        
        moved = False
        line_map, _, has_five = _compute_lines(best)
        
        if not has_five and rng.random() < 0.7:
            # Line-completion move
            three_lines = [(k, s) for k, s in line_map.items() if len(s) == 3]
            if three_lines:
                rng.shuffle(three_lines)
                for line_key, idx_set in three_lines[:5]:
                    new_pts, new_score = _try_complete_3line(best, line_key, existing_set)
                    if new_pts is not None and new_score > best_score:
                        existing_set.discard((best[idx_set][0], best[idx_set][1]))  # approximate
                        existing_set = set((p[0], p[1]) for p in new_pts)
                        best = new_pts
                        best_score = new_score
                        no_improve = 0
                        moved = True
                        break
        else:
            # Random perturbation
            idx = rng.randint(0, 15)
            x, y = best[idx]
            for _try in range(8):
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if (nx, ny) in existing_set:
                    continue
                new_pts = [list(p) for p in best]
                new_pts[idx] = [nx, ny]
                s = _score(new_pts)
                if s > best_score:
                    existing_set.discard((x, y))
                    existing_set.add((nx, ny))
                    best = new_pts
                    best_score = s
                    no_improve = 0
                    moved = True
                    break
        if not moved:
            no_improve += 1
    
    return best, best_score


def _make_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]


def _make_perturbed_grid(rng, n_perturb, radius):
    grid = _make_grid_4x4()
    pts = [list(p) for p in grid]
    for _ in range(n_perturb):
        idx = rng.randint(0, 15)
        dx = rng.randint(-radius, radius)
        dy = rng.randint(-radius, radius)
        if dx == 0 and dy == 0:
            continue
        nx, ny = pts[idx][0] + dx, pts[idx][1] + dy
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if (nx, ny) in [(p[0], p[1]) for i, p in enumerate(pts) if i != idx]:
            continue
        pts[idx] = [nx, ny]
    return pts


def _make_random_points(rng, scale=8):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts


def _make_structured_candidates(rng):
    """Generate diverse structured initial candidates."""
    candidates = []
    # 4x4 grid
    candidates.append(_make_grid_4x4())
    # Perturbed grids
    for k in range(1, 8):
        candidates.append(_make_perturbed_grid(rng, k, 3))
    # Random points
    for _ in range(10):
        candidates.append(_make_random_points(rng, scale=10))
    for _ in range(5):
        candidates.append(_make_random_points(rng, scale=5))
    # Grid with row shifts
    for shift in range(1, 4):
        candidates.append([[x, y + (shift if x == 3 else 0)] for x in range(4) for y in range(4)])
        candidates.append([[x + (shift if y == 3 else 0), y] for x in range(4) for y in range(4)])
    # Diagonal perturbations
    for d in range(1, 4):
        candidates.append([[x, y + (d if (x + y) % 2 == 0 else 0)] for x in range(4) for y in range(4)])
    return candidates


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Generate diverse initial candidates
    candidates = _make_structured_candidates(rng)
    
    best_points = None
    best_score = -1
    
    # Score all initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in cand]
    
    if best_score < 0:
        return _make_grid_4x4()
    
    # Multi-restart local search
    remaining = deadline - time.time()
    n_restarts = max(4, int(remaining / 15))  # ~15s per restart
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        restart_deadline = time.time() + min(13, deadline - time.time() - 3)
        
        if r == 0:
            start_pts = best_points
        elif r == 1:
            start_pts = _make_perturbed_grid(rng, rng.randint(2, 5), 4)
        elif r == 2:
            start_pts = _make_random_points(rng, scale=12)
        else:
            # Perturb the best solution
            start_pts = [list(p) for p in best_points]
            n_perturb = rng.randint(3, 7)
            for _ in range(n_perturb):
                idx = rng.randint(0, 15)
                dx = rng.randint(-4, 4)
                dy = rng.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = start_pts[idx][0] + dx, start_pts[idx][1] + dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    if (nx, ny) not in [(p[0], p[1]) for i, p in enumerate(start_pts) if i != idx]:
                        start_pts[idx] = [nx, ny]
        
        pts, s = _local_search(start_pts, restart_deadline, rng)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Final polish: try to add more 4-point lines
    remaining = deadline - time.time()
    if remaining > 10:
        polish_deadline = time.time() + remaining * 0.7
        pts, s = _local_search(best_points, polish_deadline, rng, max_iter=30000)
        if s > best_score:
            best_score = s
            best_points = pts
        
        # One more local search with smaller steps
        if time.time() < deadline - 3:
            pts, s = _local_search(best_points, deadline - 2, rng, max_iter=20000)
            if s > best_score:
                best_score = s
                best_points = pts
    
    # Final validation
    if not _is_valid(best_points):
        return _make_grid_4x4()
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
