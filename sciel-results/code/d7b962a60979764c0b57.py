import time
import random
from itertools import combinations

_DEADLINE = 110.0

def _popcount(x):
    return bin(x).count('1')

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

def _build_conflict_graph(points):
    n = len(points)
    conflict_mask = [0] * n
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes.setdefault((dx, dy), []).append(j)
        for k in slopes.values():
            if len(k) >= 2:
                for a_idx in range(len(k)):
                    for b_idx in range(a_idx + 1, len(k)):
                        a, b = k[a_idx], k[b_idx]
                        conflict_mask[i] |= (1 << a) | (1 << b)
                        conflict_mask[a] |= (1 << i) | (1 << b)
                        conflict_mask[b] |= (1 << i) | (1 << a)
    return conflict_mask

def _max_independent_set(conflict_mask, n, deadline_start):
    best = [0]
    all_mask = (1 << n) - 1
    degrees = [bin(conflict_mask[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    def dfs(available, count):
        nonlocal best
        if time.time() - deadline_start > 109.0:
            return
        pc = bin(available).count('1')
        if count + pc <= best[0]:
            return
        if available == 0:
            if count > best[0]:
                best[0] = count
            return
        v = None
        for idx in order:
            if available & (1 << idx):
                v = idx
                break
        if v is None:
            return
        dfs(available & ~(1 << v), count)
        new_available = available & ~(1 << v) & ~conflict_mask[v]
        dfs(new_available, count + 1)
    
    dfs(all_mask, 0)
    return best[0]

def _score(points):
    n = len(points)
    conflict_mask = _build_conflict_graph(points)
    return _max_independent_set(conflict_mask, n, time.time())

def _score_fast(points):
    n = len(points)
    conflict_mask = _build_conflict_graph(points)
    remaining = (1 << n) - 1
    greedy_size = 0
    rem = remaining
    while rem:
        best_v = -1
        best_deg = 999
        for v in range(n):
            if rem & (1 << v):
                deg = bin(conflict_mask[v] & rem).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
        if best_v < 0:
            break
        greedy_size += 1
        rem &= ~(1 << best_v) & ~conflict_mask[best_v]
    
    best = [greedy_size]
    all_mask = (1 << n) - 1
    degrees = [bin(conflict_mask[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    def dfs(available, count):
        if time.time() > _score_deadline:
            return
        if count + bin(available).count('1') <= best[0]:
            return
        if available == 0:
            if count > best[0]:
                best[0] = count
            return
        v = None
        for idx in order:
            if available & (1 << idx):
                v = idx
                break
        if v is None:
            return
        dfs(available & ~(1 << v), count)
        new_available = available & ~(1 << v) & ~conflict_mask[v]
        dfs(new_available, count + 1)
    
    global _score_deadline
    _score_deadline = time.time() + 0.5
    dfs(all_mask, 0)
    return best[0]

_score_deadline = 0

def _triple_count(points):
    n = len(points)
    count = 0
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
        for k in slopes.values():
            if k >= 3:
                count += combinations(k, 2)
    return count

def _conic_perturb_base(rng, seed_offset=0):
    """Generate points near a conic with controlled perturbations to break 4-collinearities."""
    pts = []
    # Use points on/near y = x^2 with perturbations
    for i in range(-10, 10):
        x = i
        y = x * x
        # Apply small perturbation
        dx = rng.randint(-2, 2)
        dy = rng.randint(-2, 2)
        nx, ny = x + dx, y + dy
        if -100 <= nx <= 100 and -100 <= ny <= 100 and (nx, ny) not in pts:
            pts.append((nx, ny))
    # Ensure exactly 20 distinct points
    while len(pts) < 20:
        x = rng.randint(-10, 10)
        y = rng.randint(-10, 10)
        if (x, y) not in pts:
            pts.append((x, y))
    return pts[:20]

def _structured_config_1():
    """A hand-crafted configuration with many 3-collinear triples."""
    # Try to create a configuration with ~15-20 collinear triples
    pts = [
        (0, 0), (1, 1), (2, 2),      # line y=x, 3 pts
        (0, 1), (1, 2), (2, 3),      # line y=x+1, 3 pts  
        (0, 2), (1, 3), (2, 4),      # line y=x+2, 3 pts
        (3, 0), (4, 1), (5, 2),      # line y=x-3, 3 pts
        (0, 3), (1, 4), (2, 5),      # line y=x+3, 3 pts
        (3, 1), (4, 2), (5, 3),      # line y=x-2, 3 pts
        (3, 2), (4, 3), (5, 4),      # line y=x-1, 3 pts
        (6, 0), (7, 1), (8, 2),      # line y=x-6, 3 pts
        (6, 1), (7, 2), (8, 3),      # line y=x-5, 3 pts
    ]
    # This has too many points; need exactly 20
    return pts[:20]

def _dense_triple_config(rng):
    """Build configuration by placing points to maximize 3-collinear triples."""
    pts = []
    used = set()
    
    # Create several lines with 3 points each, allowing shared points
    line_defs = [
        # (x1, y1, x2, y2) - two points defining a line
        (0, 0, 1, 1),
        (0, 1, 1, 2),
        (1, 0, 2, 1),
        (0, 2, 1, 3),
        (2, 0, 3, 1),
        (0, 3, 1, 4),
        (3, 0, 4, 1),
        (1, 1, 2, 2),
        (2, 1, 3, 2),
        (1, 2, 2, 3),
    ]
    
    for x1, y1, x2, y2 in line_defs:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        g = _gcd(dx, dy)
        dx //= g
        dy //= g
        
        # Find 3 points on this line
        candidates = []
        for t in range(-1, 4):
            px = x1 + t * dx
            py = y1 + t * dy
            if -100 <= px <= 100 and -100 <= py <= 100:
                candidates.append((px, py))
        
        # Pick 3 that minimize overlap with used
        best_triple = None
        best_overlap = 999
        for a in range(len(candidates)):
            for b in range(a + 1, len(candidates)):
                for c in range(b + 1, len(candidates)):
                    triple = [candidates[a], candidates[b], candidates[c]]
                    overlap = sum(1 for p in triple if p in used)
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_triple = triple
        
        if best_triple:
            for p in best_triple:
                if p not in used:
                    pts.append(p)
                    used.add(p)
    
    # Fill remaining with random points
    while len(pts) < 20:
        x = rng.randint(-8, 8)
        y = rng.randint(-8, 8)
        if (x, y) not in used:
            pts.append((x, y))
            used.add((x, y))
    
    return pts[:20]

def construct():
    start = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = 21
    
    def consider(pts):
        nonlocal best_pts, best_score
        if len(pts) != 20 or len(set(pts)) != 20:
            return
        if not _valid_no_four(pts):
            return
        s = _score_fast(pts)
        if s < best_score:
            best_score = s
            best_pts = list(pts)
    
    # Strategy A: Conic-based perturbation
    for seed in range(20000):
        if time.time() - start > 95.0:
            break
        r = random.Random(seed)
        pts = _conic_perturb_base(r)
        consider(pts)
        if best_score <= 12:
            break
    
    # Strategy B: Dense triple configuration
    for seed in range(20000):
        if time.time() - start > 100.0:
            break
        r = random.Random(seed)
        pts = _dense_triple_config(r)
        consider(pts)
        if best_score <= 12:
            break
    
    # Strategy C: Local search from best found
    if best_pts:
        current_pts = list(best_pts)
        current_score = best_score
        current_tc = _triple_count(current_pts)
        
        temp = 5.0
        for step in range(50000):
            if time.time() - start > 108.0:
                break
            temp *= 0.9995
            
            idx = rng.randrange(20)
            x, y = current_pts[idx]
            
            best_move = None
            best_tc = current_tc
            for _ in range(10):
                nx = x + rng.randint(-3, 3)
                ny = y + rng.randint(-3, 3)
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            tc = _triple_count(current_pts)
                            delta_tc = tc - current_tc
                            if delta_tc < 0 or rng.random() < pow(2.718281828, -delta_tc / max(temp, 0.1)):
                                if tc < best_tc:
                                    best_tc = tc
                                    best_move = new_pt
                        current_pts[idx] = old_pt
            
            if best_move is not None:
                old_pt = current_pts[idx]
                current_pts[idx] = best_move
                current_tc = best_tc
                new_score = _score_fast(current_pts)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_pts = list(current_pts)
                elif rng.random() < pow(2.718281828, -(new_score - current_score) / max(temp, 0.1)):
                    current_score = new_score
    
    if best_pts is None:
        return [[x, x * x] for x in range(-10, 10)]
    
    return [list(p) for p in best_pts]
