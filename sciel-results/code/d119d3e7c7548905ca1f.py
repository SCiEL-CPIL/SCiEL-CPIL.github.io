# EVOLVE-BLOCK-START
import time
from math import isqrt

MAXT = 150.0          # hard wall-clock budget (seconds)
LIMIT  = 10**40       # numerator/denominator bound

# ---------- exact integer helpers ----------
def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _gcd3(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

# ---------- geometric predicates (exact, on integer coords) ----------
def _all_dist2_squares(pts):
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx, dy = xj - xi, yj - yi
            if not _is_square(dx * dx + dy * dy):
                return False
    return True

def _no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                if (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi) == 0:
                    return False
    return True

def _no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    ax, ay = pts[i]; bx, by = pts[j]
                    cx, cy = pts[k]; dx, dy = pts[l]
                    # Power of D w.r.t. circle through A,B,C.
                    # = 2*((B-A)x(C-A))((C-A)x(D-A))
                    #   - |C-A|^2*(B-A)x(D-A)
                    #   + |B-A|^2*(C-A)x(D-A)
                    #   divided by 2*|(B-A)x(C-A)|^2  (nonzero iff non-collinear)
                    ABx, ABy = bx - ax, by - ay
                    ACx, ACy = cx - ax, cy - ay
                    ADx, ADy = dx - ax, dy - ay
                    cross = ABx * ACy - ABy * ACx
                    if cross == 0:
                        continue
                    dot1 = ABx * ADx + ABy * ADy
                    dot2 = ACx * ADx + ACy * ADy
                    crossAD  = ABx * ADy - ABy * ADx
                    crossACD = ACx * ADy - ACy * ADx
                    cB = ABx * ABx + ABy * ABy
                    cC = ACx * ACx + ACy * ACy
                    if 2 * cross * crossAD - cC * crossAD + cB * crossACD == 0:
                        return False
    return True

def _valid_set(pts):
    if len(pts) < 3:
        return False
    if len(set(pts)) != len(pts):
        return False
    if not _all_dist2_squares(pts):
        return False
    if not _no_three_collinear(pts):
        return False
    if not _no_four_concyclic(pts):
        return False
    return True

# ---------- candidate point tests ----------
def _try_add(pts, p):
    if p in pts:
        return False
    x, y = p
    for xi, yi in pts:
        dx, dy = x - xi, y - yi
        if not _is_square(dx * dx + dy * dy):
            return False
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            if (xj - xi) * (y - yi) - (yj - yi) * (x - xi) == 0:
                return False
    return True

def _try_replace(pts, drop, p):
    if p in pts:
        return False
    base = pts[:drop] + pts[drop + 1:]
    return _try_add(base, p)

# ---------- greedy construction ----------
def _greedy(radius):
    best = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            p = (x, y)
            if _try_add(best, p):
                best.append(p)
    return best

# ---------- local search: perturb-and-replace ----------
def _local_search(pts, max_rounds=200, radius=200):
    current = list(pts)
    best = list(pts)
    for r in range(max_rounds):
        improved = False
        for drop in range(len(current)):
            if improved:
                break
            for dx in range(-15, 16):
                if improved:
                    break
                for dy in range(-15, 16):
                    np_ = (current[drop][0] + dx, current[drop][1] + dy)
                    if _try_replace(current, drop, np_):
                        cand = current[:drop] + [np_] + current[drop + 1:]
                        if len(cand) > len(best) and _valid_set(cand):
                            best = cand
                            current = cand
                            improved = True
                            break
        if not improved:
            break
    return best

# ---------- main ----------
def construct():
    t0 = time.time()
    deadline = t0 + MAXT - 30.0   # safety margin: return well before the 150s kill

    # Stage 1: greedy lattice construction, growing the search radius.
    best_pts = _greedy(120)
    if time.time() > deadline:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}

    # Stage 2: local search (perturb-and-replace) to escape symmetric traps.
    improved = _local_search(best_pts, max_rounds=300, radius=200)
    if len(improved) > len(best_pts):
        best_pts = improved
    if time.time() > deadline:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}

    # Stage 3: greedy add a 7th point (or more) with a wider radius.
    for r2 in (150, 200, 250):
        if time.time() > deadline:
            break
        for x in range(-r2, r2 + 1):
            if time.time() > deadline:
                break
            for y in range(-r2, r2 + 1):
                if time.time() > deadline:
                    break
                p = (x, y)
                if _try_add(best_pts, p) and _valid_set(best_pts + [p]):
                    best_pts.append(p)
                    break
            else:
                continue
            break

    # Stage 4: another local-search round if time remains.
    if time.time() < deadline - 10:
        improved2 = _local_search(best_pts, max_rounds=200, radius=200)
        if len(improved2) > len(best_pts):
            best_pts = improved2

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}
# EVOLVE-BLOCK-END
