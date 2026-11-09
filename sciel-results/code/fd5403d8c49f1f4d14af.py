"""Find maximum integral point set in general position."""
from fractions import Fraction
from math import isqrt
import time

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _dist2_sq(p1, p2, D):
    """Return squared distance as (num, den) for points (x1,y1*sqrt(D)), (x2,y2*sqrt(D)) with integer coords."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy

def _all_rational_distances(pts, D):
    """Check all pairwise distances are rational (i.e., dist^2 is a perfect square integer for D=1)."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _dist2_sq(pts[i], pts[j], D)
            if D == 1:
                if not _is_square(d2):
                    return False
            else:
                # dist^2 = dx^2 + D*dy^2, need this to be a rational square
                # For integral points, need d2 to be a perfect square
                if not _is_square(d2):
                    return False
    return True

def _no_three_collinear(pts):
    """Check no three points are collinear using cross product."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Cross product of (pj-pi) and (pk-pi)
                ax = pts[j][0] - pts[i][0]
                ay = pts[j][1] - pts[i][1]
                bx = pts[k][0] - pts[i][0]
                by = pts[k][1] - pts[i][1]
                cross = ax * by - ay * bx
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(pts):
    """Check no four points are concyclic using determinant method."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Four points (x1,y1),(x2,y2),(x3,y3),(x4,y4) are concyclic iff:
                    # det[[x1^2+y1^2, x1, y1, 1], ...] = 0
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    # Use the determinant formula
                    # | x1  y1  x1^2+y1^2  1 |
                    # | x2  y2  x2^2+y2^2  1 | = 0
                    # | x3  y3  x3^2+y3^2  1 |
                    # | x4  y4  x4^2+y4^2  1 |
                    # Expand along last column:
                    # = (x2^2+y2^2)(x3*y4-x4*y3) - (x3^2+y3^2)(x2*y4-x4*y2)
                    #   + (x4^2+y4^2)(x2*y3-x3*y2) - (x1^2+y1^2)(x3*y4-x4*y3)
                    #   + ... actually let me use a cleaner formula
                    
                    # det = |x1 y1 x1²+y1² 1|
                    #         |x2 y2 x2²+y2² 1|
                    #         |x3 y3 x3²+y3² 1|
                    #         |x4 y4 x4²+y4² 1|
                    
                    # Subtract row1 from rows 2,3,4:
                    # |x1 y1 s1 1|
                    # |x2-x1 y2-y1 s2-s1 0|
                    # |x3-x1 y3-y1 s3-s1 0|
                    # |x4-x1 y4-y1 s4-s1 0|
                    # where si = xi²+yi²
                    # = 1 * det[[x2-x1, y2-y1, s2-s1],
                    #             [x3-x1, y3-y1, s3-s1],
                    #             [x4-x1, y4-y1, s4-s1]]
                    # with sign (-1)^(1+4) = -1, so det = -that 3x3 det
                    
                    s1 = x1*x1 + y1*y1
                    s2 = x2*x2 + y2*y2
                    s3 = x3*x3 + y3*y3
                    s4 = x4*x4 + y4*y4
                    
                    a1 = x2-x1; b1 = y2-y1; c1 = s2-s1
                    a2 = x3-x1; b2 = y3-y1; c2 = s3-s1
                    a3 = x4-x1; b3 = y4-y1; c3 = s4-s1
                    
                    det3 = a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2)
                    if det3 == 0:
                        return False
    return True

def _valid_set(pts, D):
    """Full validity check."""
    n = len(pts)
    if n < 3:
        return False
    # Check distinctness
    for i in range(n):
        for j in range(i+1, n):
            if pts[i] == pts[j]:
                return False
    if not _all_rational_distances(pts, D):
        return False
    if not _no_three_collinear(pts):
        return False
    if not _no_four_concyclic(pts):
        return False
    return True

def _try_add_point(pts, D, deadline):
    """Try to add a 6th point to a 5-point set by searching nearby lattice points."""
    n = len(pts)
    # Compute bounding box
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # Search in expanded region
    margin = 80
    sxmin, sxmax = xmin - margin, xmax + margin
    symin, symax = ymin - margin, ymax + margin
    
    # Precompute existing points for speed
    existing = set(pts)
    
    for x in range(sxmin, sxmax + 1):
        if time.time() > deadline:
            break
        for y in range(symin, symax + 1):
            if time.time() > deadline:
                break
            new_pt = (x, y)
            if new_pt in existing:
                continue
            # Quick check: distances to all existing points must be perfect squares
            ok = True
            for px, py in pts:
                d2 = (x - px)**2 + D * (y - py)**2
                if not _is_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            # Check collinearity with all pairs
            ok = True
            for i in range(n):
                for j in range(i+1, n):
                    ax = pts[j][0] - pts[i][0]
                    ay = pts[j][1] - pts[i][1]
                    bx = x - pts[i][0]
                    by = y - pts[i][1]
                    if ax * by - ay * bx == 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check concyclicity with all triples
            ok = True
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        x4, y4 = x, y
                        s1 = x1*x1 + y1*y1
                        s2 = x2*x2 + y2*y2
                        s3 = x3*x3 + y3*y3
                        s4 = x4*x4 + y4*y4
                        a1 = x2-x1; b1 = y2-y1; c1 = s2-s1
                        a2 = x3-x1; b2 = y3-y1; c2 = s3-s1
                        a3 = x4-x1; b3 = y4-y1; c3 = s4-s1
                        det3 = a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2)
                        if det3 == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return new_pt
    return None

def _try_add_point_wider(pts, D, deadline):
    """Try to add a 6th point with a wider search, prioritizing points near the centroid."""
    n = len(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) // n
    cy = sum(ys) // n
    
    # Generate candidate points sorted by distance from centroid
    margin = 100
    candidates = []
    for x in range(cx - margin, cx + margin + 1):
        for y in range(cy - margin, cy + margin + 1):
            d = (x - cx)**2 + (y - cy)**2
            candidates.append((d, (x, y)))
    candidates.sort()
    
    existing = set(pts)
    
    for d, new_pt in candidates:
        if time.time() > deadline:
            break
        x, y = new_pt
        if new_pt in existing:
            continue
        # Quick distance check
        ok = True
        for px, py in pts:
            d2 = (x - px)**2 + D * (y - py)**2
            if not _is_square(d2):
                ok = False
                break
        if not ok:
            continue
        # Collinearity
        ok = True
        for i in range(n):
            for j in range(i+1, n):
                ax = pts[j][0] - pts[i][0]
                ay = pts[j][1] - pts[i][1]
                bx = x - pts[i][0]
                by = y - pts[i][1]
                if ax * by - ay * bx == 0:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        # Concyclicity
        ok = True
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = x, y
                    s1 = x1*x1 + y1*y1
                    s2 = x2*x2 + y2*y2
                    s3 = x3*x3 + y3*y3
                    s4 = x4*x4 + y4*y4
                    a1 = x2-x1; b1 = y2-y1; c1 = s2-s1
                    a2 = x3-x1; b2 = y3-y1; c2 = s3-s1
                    a3 = x4-x1; b3 = y4-y1; c3 = s4-s1
                    det3 = a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2)
                    if det3 == 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return new_pt
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin: finish before 150s limit
    
    # Start with the known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Verify base is valid
    if not _valid_set(base_points, D):
        # Fallback: return a known valid 3-point set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [5, 1]]]}
    
    best = list(base_points)
    
    # Try to add a 6th point to the base configuration
    if time.time() < deadline - 10:
        new_pt = _try_add_point(base_points, D, deadline - 10)
        if new_pt is not None:
            candidate = base_points + [new_pt]
            if _valid_set(candidate, D):
                best = candidate
    
    # Try alternative 5-point configurations and add 6th point
    # Generate some other 5-point sets by modifying the base
    alternative_bases = [
        # Shifted versions
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (-56, -33), (-16, -63), (0, -33), (16, -63)],
        [(0, 0), (56, -33), (16, -63), (0, -33), (-16, -63)],
        # Different configurations
        [(0, 0), (3, 4), (0, 5), (4, 3), (-3, 4)],
        [(0, 0), (5, 12), (0, 13), (12, 5), (-5, 12)],
        [(0, 0), (8, 15), (0, 17), (15, 8), (-8, 15)],
    ]
    
    for base in alternative_bases:
        if time.time() > deadline - 15:
            break
        if not _valid_set(base, D):
            continue
        if len(base) < len(best):
            continue
        # Try to add 6th point
        new_pt = _try_add_point_wider(base, D, deadline - 15)
        if new_pt is not None:
            candidate = base + [new_pt]
            if _valid_set(candidate, D):
                if len(candidate) > len(best):
                    best = candidate
    
    # Try to add a 7th point if we have 6
    if len(best) >= 6 and time.time() < deadline - 5:
        new_pt = _try_add_point(best, D, deadline - 5)
        if new_pt is not None:
            candidate = best + [new_pt]
            if _valid_set(candidate, D):
                best = candidate
    
    # Try to add 7th point with wider search
    if len(best) >= 6 and time.time() < deadline - 3:
        new_pt = _try_add_point_wider(best, D, deadline - 3)
        if new_pt is not None:
            candidate = best + [new_pt]
            if _valid_set(candidate, D):
                best = candidate
    
    # Final validity check
    if not _valid_set(best, D):
        # Fallback to known valid set
        best = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best]}
