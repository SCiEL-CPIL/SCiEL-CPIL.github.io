"""Combined approach: structured seeds + incremental line-map maintenance + simulated annealing.
Maximizes lines with exactly four points, no five collinear."""
import time
import math
import random
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 140.0
_BOUND = 1000


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = x1 * dy - y1 * dx
    return (dx, dy, c)


def _build_line_map(points):
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            line_map[k].add(i)
            line_map[k].add(j)
    return line_map


def _score_from_line_map(line_map):
    n4 = 0
    n5 = 0
    for s in line_map.values():
        sz = len(s)
        if sz == 4:
            n4 += 1
        elif sz >= 5:
            n5 += 1
    return n4, n5


def _full_score(points):
    line_map = _build_line_map(points)
    return _score_from_line_map(line_map)


def _valid(points):
    line_map = _build_line_map(points)
    for s in line_map.values():
        if len(s) >= 5:
            return False
    return True


def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_staggered():
    pts = []
    for row in range(4):
        offset = row * 2
        for col in range(4):
            pts.append([col + offset, row])
    return pts


def _seed_diagonal():
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i + j, i - j])
    return pts


def _seed_affine():
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + 2 * y, y])
    return pts


def _seed_random(rng, box=30):
    pts = []
    for _ in range(200):
        if len(pts) >= 16:
            break
        x = rng.randint(-box, box)
        y = rng.randint(-box, box)
        p = [x, y]
        if p not in pts:
            pts.append(p)
    while len(pts) < 16:
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        p = [x, y]
        if p not in pts:
            pts.append(p)
    return pts


def _local_search(points, rng, time_budget):
    n = len(points)
    cur = [list(p) for p in points]
    line_map = _build_line_map(cur)
    cur_n4, cur_n5 = _score_from_line_map(line_map)
    best_key = (cur_n4, -cur_n5)
    best_pts = [list(p) for p in cur]
    
    T = 20.0
    min_T = 0.1
    cooling = 0.995
    
    pos_set = set(map(tuple, cur))
    step = 0
    
    while time.time() < time_budget:
        step += 1
        if step % 5000 == 0:
            T = max(min_T, T * cooling)
        
        i = rng.randrange(n)
        x0, y0 = cur[i]
        
        r = rng.random()
        if r < 0.7:
            w = rng.randint(1, 6)
            nx = x0 + rng.randint(-w, w)
            ny = y0 + rng.randint(-w, w)
        else:
            nx = rng.randint(-60, 60)
            ny = rng.randint(-60, 60)
        
        if abs(nx) > _BOUND or abs(ny) > _BOUND:
            continue
        if (nx, ny) in pos_set:
            continue
        
        new_pos = [nx, ny]
        
        # Find lines through old position
        lines_old = []
        for k, s in line_map.items():
            if i in s:
                lines_old.append(k)
        
        # Remove i from those lines
        for k in lines_old:
            line_map[k].discard(i)
            if not line_map[k]:
                del line_map[k]
        
        # Add i to lines through new position
        lines_new = []
        for j in range(n):
            if j == i:
                continue
            k = _line_key(new_pos, cur[j])
            if k not in line_map:
                line_map[k] = set()
            line_map[k].add(i)
            line_map[k].add(j)
            lines_new.append(k)
        
        # Check validity
        valid = True
        for k in lines_new:
            if len(line_map[k]) >= 5:
                valid = False
                break
        # Also need to check lines that lost a point might now be fine, but lines that gained might be bad
        # Actually we only added to lines_new, and removed from lines_old
        # Lines in lines_old that are not in lines_new lost point i, so they can only get smaller - safe
        # Lines in lines_new gained point i - need to check
        # But also: a line that was in both old and new - we removed then re-added, net zero for i
        # The critical check: any line with >= 5 points
        if valid:
            for k in lines_old:
                if k in line_map and len(line_map[k]) >= 5:
                    valid = False
                    break
        
        if not valid:
            # Rollback: remove i from lines_new, re-add to lines_old
            for k in lines_new:
                line_map[k].discard(i)
                if not line_map[k]:
                    del line_map[k]
            for k in lines_old:
                if k not in line_map:
                    line_map[k] = set()
                line_map[k].add(i)
            continue
        
        new_n4, new_n5 = _score_from_line_map(line_map)
        new_key = (new_n4, -new_n5)
        cur_key = (cur_n4, -cur_n5)
        
        if new_key > cur_key:
            pos_set.discard((x0, y0))
            pos_set.add((nx, ny))
            cur[i] = new_pos
            cur_n4, cur_n5 = new_n4, new_n5
            if new_key > best_key:
                best_key = new_key
                best_pts = [list(p) for p in cur]
        else:
            delta = (new_n4 - cur_n4) - (new_n5 - cur_n5)
            if delta < 0 and rng.random() < math.exp(delta / T):
                pos_set.discard((x0, y0))
                pos_set.add((nx, ny))
                cur[i] = new_pos
                cur_n4, cur_n5 = new_n4, new_n5
            else:
                # Rollback
                for k in lines_new:
                    line_map[k].discard(i)
                    if not line_map[k]:
                        del line_map[k]
                for k in lines_old:
                    if k not in line_map:
                        line_map[k] = set()
                    line_map[k].add(i)
    
    return (best_key[0], -best_key[1], best_pts)


def construct():
    global _START, _DEADLINE
    rng = random.Random(42)
    
    seeds = [
        _seed_grid(),
        _seed_staggered(),
        _seed_diagonal(),
        _seed_affine(),
        _seed_random(rng, 20),
        _seed_random(rng, 40),
        _seed_random(rng, 100),
    ]
    
    seen = set()
    uniq = []
    for s in seeds:
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    
    best_overall = None
    remaining = _DEADLINE - time.time()
    per_budget = max(3.0, remaining / max(1, len(uniq)))
    
    for s in uniq:
        if time.time() >= _DEADLINE - 3.0:
            break
        if not _valid(s):
            continue
        t_end = min(_DEADLINE - 1.0, time.time() + per_budget)
        res = _local_search(s, rng, t_end)
        if best_overall is None or (res[0], -res[1]) > (best_overall[0], -best_overall[1]):
            best_overall = res
    
    if best_overall is None:
        return _seed_grid()
    
    pts = best_overall[2]
    
    if not _valid(pts):
        return _seed_grid()
    for p in pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return _seed_grid()
    if len(set(map(tuple, pts))) != 16:
        return _seed_grid()
    
    return [list(p) for p in pts]
