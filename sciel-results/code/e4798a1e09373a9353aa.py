"""Search for integral point sets in general position (no 3 collinear, no 4 concyclic)."""
import math
import time
from fractions import Fraction
from itertools import combinations

_TIME_LIMIT = 110  # safety margin before 120s kill
_DEADLINE = time.time() + _TIME_LIMIT
_best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
_best_D = 1


def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _gcd3(a, b, c):
    return math.gcd(math.gcd(a, b), c)


def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) != 0:
                    continue
                return False
    return True


def _check_no_four_concyclic(pts):
    n = len(pts)
    for quad in combinations(range(n), 4):
        x1, y1 = pts[quad[0]]
        x2, y2 = pts[quad[1]]
        x3, y3 = pts[quad[2]]
        x4, y4 = pts[quad[3]]
        # determinant of 4x4 matrix; zero iff concyclic
        # | x^2+y^2  x  y  1 |
        a1 = x1*x1 + y1*y1; b1 = x1; c1 = y1
        a2 = x2*x2 + y2*y2; b2 = x2; c2 = y2
        a3 = x3*x3 + y3*y3; b3 = x3; c3 = y3
        a4 = x4*x4 + y4*y4; b4 = x4; c4 = y4
        # det = a1*(-b2*c3+b2*c4-b3*c4+b3*c2) - b1*(-a2*c3+a2*c4-a3*c4+a3*c2)
        #     + c1*(-a2*b3+a2*b4-a3*b4+a3*b2) - 1*(-a2*b3*c4+a2*b4*c3-a3*b4*c2
        #       + a3*b3*c4-a4*b3*c2+a4*b4*c3)
        t1 = -b2*c3 + b2*c4 - b3*c4 + b3*c2
        t2 = -a2*c3 + a2*c4 - a3*c4 + a3*c2
        t3 = -a2*b3 + a2*b4 - a3*b4 + a3*b2
        t4 = -a2*b3*c4 + a2*b4*c3 - a3*b4*c2 + a3*b3*c4 - a4*b3*c2 + a4*b4*c3
        det = a1*t1 - b1*t2 + c1*t3 - t4
        if det == 0:
            return False
    return True


def _all_distances_rational(pts, D):
    for i in range(len(pts)):
        x1, y1 = pts[i]
        for j in range(i + 1, len(pts)):
            x2, y2 = pts[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx*dx + D*dy*dy
            if not _is_square(d2):
                return False
    return True


def _is_valid(pts, D):
    if len(pts) < 3:
        return False
    if len(set(pts)) != len(pts):
        return False
    if not _all_distances_rational(pts, D):
        return False
    if not _check_no_three_collinear(pts):
        return False
    if not _check_no_four_concyclic(pts):
        return False
    return True


def _format(pts, D):
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in pts]}


def _try_grow(pts, D, max_coord, deadline):
    """Try to add one integer point with |coord| <= max_coord."""
    n = len(pts)
    # Candidate x, y ranges
    xs = sorted(set([p[0] for p in pts]))
    ys = sorted(set([p[1] for p in pts]))
    xmin = min(xs) - max_coord
    xmax = max(xs) + max_coord
    ymin = min(ys) - max_coord
    ymax = max(ys) + max_coord
    # Only test candidates that pass distance checks quickly
    for x in range(xmin, xmax + 1):
        if time.time() > deadline:
            return None
        for y in range(ymin, ymax + 1):
            # quick distance check
            ok = True
            for (x1, y1) in pts:
                dx = x - x1
                dy = y - y1
                d2 = dx*dx + D*dy*dy
                if not _is_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            # collinearity quick check
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    if (x2 - x1)*(y - y1) - (x - x1)*(y2 - y1) == 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            new_pts = pts + [(x, y)]
            if _check_no_four_concyclic(new_pts):
                return new_pts
    return None


def _try_grow_with_offsets(pts, D, max_offset, deadline):
    """Try adding points using offsets from existing points (for larger search)."""
    n = len(pts)
    # Generate candidate offsets: small integer offsets
    offsets = []
    for dx in range(-max_offset, max_offset + 1):
        for dy in range(-max_offset, max_offset + 1):
            if dx == 0 and dy == 0:
                continue
            offsets.append((dx, dy))
    # Try each base point + offset
    for base_x, base_y in pts:
        for dx, dy in offsets:
            if time.time() > deadline:
                return None
            x = base_x + dx
            y = base_y + dy
            ok = True
            for (x1, y1) in pts:
                ddx = x - x1
                ddy = y - y1
                d2 = ddx*ddx + D*ddy*ddy
                if not _is_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    if (x2 - x1)*(y - y1) - (x - x1)*(y2 - y1) == 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            new_pts = pts + [(x, y)]
            if _check_no_four_concyclic(new_pts):
                return new_pts
    return None


def _perturb(pts, D, deadline):
    """Try moving one point to a nearby integer position to unlock growth."""
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        for dx in range(-30, 31):
            if time.time() > deadline:
                return None
            for dy in range(-30, 31):
                if dx == 0 and dy == 0:
                    continue
                x = x0 + dx
                y = y0 + dy
                new_pts = pts[:i] + [(x, y)] + pts[i+1:]
                if len(set(new_pts)) != n:
                    continue
                if not _all_distances_rational(new_pts, D):
                    continue
                if not _check_no_three_collinear(new_pts):
                    continue
                if not _check_no_four_concyclic(new_pts):
                    continue
                # valid perturbation; try to grow
                grown = _try_grow(new_pts, D, 60, deadline)
                if grown is not None:
                    return grown
                grown = _try_grow_with_offsets(new_pts, D, 40, deadline)
                if grown is not None:
                    return grown
    return None


def construct():
    global _best_points, _best_D
    deadline = time.time() + _TIME_LIMIT
    # Start from known 5-point set
    pts = list(_best_points)
    D = 1
    # Try to grow iteratively
    for step in range(10):
        if time.time() > deadline:
            break
        grown = _try_grow(pts, D, 80, deadline)
        if grown is not None:
            pts = grown
            continue
        grown = _try_grow_with_offsets(pts, D, 50, deadline)
        if grown is not None:
            pts = grown
            continue
        # Try perturbation
        grown = _perturb(pts, D, deadline)
        if grown is not None:
            pts = grown
            continue
        break
    # Update best
    if len(pts) > len(_best_points):
        _best_points = list(pts)
        _best_D = D
    # Return best found
    if len(pts) >= 3 and _is_valid(pts, D):
        return _format(pts, D)
    return _format(_best_points, _best_D)
