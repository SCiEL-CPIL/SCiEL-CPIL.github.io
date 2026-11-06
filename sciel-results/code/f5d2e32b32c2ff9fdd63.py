import time
import random
from collections import Counter

def _max_multiplicity(pts):
    n = len(pts)
    cnt = Counter()
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx1 = xj - xi
            dy1 = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                d = abs(dx1 * (yk - yi) - dy1 * (xk - xi))
                if d:
                    cnt[d] += 1
    return max(cnt.values()) if cnt else 0

def _valid(pts):
    if len(pts) != 30:
        return False
    if len(set(pts)) != 30:
        return False
    return all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts)

def _two_lines(n1, n2, y1, y2, x0, spacing=1):
    return [(x0 + spacing * t, y1) for t in range(n1)] + \
           [(x0 + spacing * t, y2) for t in range(n2)]

def construct():
    deadline = time.time() + 80.0
    rng = random.Random(12345)

    candidates = []
    candidates.append(_two_lines(15, 15, 0, 1, -7, 1))
    candidates.append([(0, y) for y in range(-7, 8)] + [(1, y) for y in range(-7, 8)])
    for n1 in range(10, 21):
        candidates.append(_two_lines(n1, 30 - n1, 0, 1, -7, 1))
    candidates.append(_two_lines(15, 15, 0, 1, -28, 2))
    candidates.append([(x, y) for x in range(6) for y in range(5)])

    best_pts = None
    best_val = -1
    for cand in candidates:
        if not _valid(cand):
            continue
        v = _max_multiplicity(cand)
        if v > best_val:
            best_val = v
            best_pts = list(cand)
        if time.time() > deadline:
            break

    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]

    # Local search: try to improve beyond the closed-form best.
    if time.time() < deadline - 5.0:
        cur = list(best_pts)
        cur_val = best_val
        occupied = set(cur)
        improved = True
        stall = 0
        while time.time() < deadline - 5.0 and stall < 400:
            idx = rng.randrange(30)
            ox, oy = cur[idx]
            # propose a new position
            nx = rng.randint(-30, 30)
            ny = rng.randint(-30, 30)
            if (nx, ny) in occupied:
                stall += 1
                continue
            cur[idx] = (nx, ny)
            occupied.remove((ox, oy))
            occupied.add((nx, ny))
            v = _max_multiplicity(cur)
            if v > cur_val:
                cur_val = v
                best_val = v
                best_pts = list(cur)
                stall = 0
            else:
                # revert
                cur[idx] = (ox, oy)
                occupied.remove((nx, ny))
                occupied.add((ox, oy))
                stall += 1
        # also try targeted moves: move a point along its line / to the other line
    return best_pts
