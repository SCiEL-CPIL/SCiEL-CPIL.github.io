import math, time
from fractions import Fraction

T_START = time.time()
TIME_LIMIT = 118.0
FALLBACK = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

def _time_left():
    return TIME_LIMIT - (time.time() - T_START)

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _is_valid(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if not _is_square(dx * dx + dy * dy):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                (x1, y1), (x2, y2), (x3, y3) = points[i], points[j], points[k]
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    (x1, y1), (x2, y2) = points[i], points[j]
                    (x3, y3), (x4, y4) = points[k], points[l]
                    if ((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2 + x3 * x3 + y3 * y3)
                            - (x2 * x2 + y2 * y2) * (x1 * x1 + y1 * y1 + x3 * x3 + y3 * y3)
                            - (x3 * x3 + y3 * y3) * (x1 * x1 + y1 * y1 + x2 * x2 + y2 * y2)
                            + (x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2 + x4 * x4 + y4 * y4)
                            - (x2 * x2 + y2 * y2) * (x1 * x1 + y1 * y1 + x4 * x4 + y4 * y4)
                            - (x4 * x4 + y4 * y4) * (x1 * x1 + y1 * y1 + x2 * x2 + y2 * y2)) != 0:
                        return False
    return True

def _try_add(points, cand):
    if any(cand == p for p in points):
        return False
    for p in points:
        dx = cand[0] - p[0]
        dy = cand[1] - p[1]
        if not _is_square(dx * dx + dy * dy):
            return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            (x1, y1), (x2, y2) = points[i], points[j]
            if (cand[0] - x1) * (y2 - y1) - (x2 - x1) * (cand[1] - y1) == 0:
                return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                (x1, y1), (x2, y2), (x3, y3) = points[i], points[j], points[k]
                a = (cand[0] * cand[0] + cand[1] * cand[1]) * (x2 * x2 + y2 * y2 + x3 * x3 + y3 * y3)
                b = (x2 * x2 + y2 * y2) * (cand[0] * cand[0] + cand[1] * cand[1] + x3 * x3 + y3 * y3)
                c = (x3 * x3 + y3 * y3) * (cand[0] * cand[0] + cand[1] * cand[1] + x2 * x2 + y2 * y2)
                if a - b - c != 0:
                    return False
    return True

def _add_point(points, R, S):
    x = points[0][0]
    y = points[0][1]
    d = R
    A = points[1][0] - x
    B = points[1][1] - y
    dd = A * A + B * B
    num = (d * d + dd - S * S) * dd
    den = 2 * dd
    if num % den != 0:
        return None
    xx = num // den
    disc = d * d * dd - xx * xx
    if disc < 0 or not _is_square(disc):
        return None
    sq = math.isqrt(disc)
    for sgn in (1, -1):
        yy = sgn * sq // den
        cand = (x + xx, y + yy)
        if _try_add(points, cand):
            return cand
    return None

def _search(points, R, S, deadline):
    n = len(points)
    pts = list(points)
    for r in range(1, R + 1):
        if time.time() > deadline:
            return pts
        for s in range(1, S + 1):
            if time.time() > deadline:
                return pts
            cand = _add_point(pts, r, s)
            if cand is not None:
                pts.append(cand)
                if n + 1 >= 8:
                    return pts
                pts = _search(pts, R, S, deadline)
                if len(pts) > n:
                    return pts
                pts.pop()
    return pts

def _fmt(points):
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

def construct():
    deadline = T_START + TIME_LIMIT - 5.0
    # seed: 5 known integral points in general position
    pts = list(FALLBACK)
    # try to grow the seed
    pts = _search(pts, 90, 90, deadline)
    if len(pts) >= 6 and _is_valid(pts):
        return _fmt(pts)
    # fresh incremental search from a 3-point seed
    for R in (60, 40, 30, 22):
        if time.time() > deadline:
            break
        for S in (R, 60, 45, 35):
            if time.time() > deadline:
                break
            for a in range(1, 12):
                if time.time() > deadline:
                    break
                for b in range(a + 1, 13):
                    if time.time() > deadline:
                        break
                    seed = [(0, 0), (a, 0), (b, 0)]
                    res = _search(seed, R, S, deadline)
                    if len(res) >= 6 and _is_valid(res):
                        return _fmt(res)
    return _fmt(FALLBACK)
