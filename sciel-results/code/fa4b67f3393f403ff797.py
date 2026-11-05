"""Search for integral point sets in general position."""
from fractions import Fraction
from itertools import combinations
import time

_DEADLINE = 140.0  # safety margin before 150s kill

def _is_square(n):
    if n < 0:
        return False
    r = int(n ** 0.5)
    for c in (r - 1, r, r + 1):
        if c >= 0 and c * c == n:
            return True
    return False

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _sq_free(n):
    """Return (d, k) such that n = d * k^2 with d squarefree."""
    if n == 0:
        return 0, 0
    k = 1
    d = n
    f = 2
    while f * f <= d:
        while d % (f * f) == 0:
            d //= (f * f)
            k *= f
        f += 1 if f == 2 else 2
    return d, k

def _dist_sq(p1, p2, D):
    """Squared distance between two points (x1,y1),(x2,y2) in (x, y*sqrt(D)) space."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _is_rational_dist(p1, p2, D):
    """Check if distance is rational (squared distance is a perfect square rational)."""
    ds = _dist_sq(p1, p2, D)
    num, den = ds.numerator, ds.denominator
    if den == 1:
        return _is_square(num)
    # ds = num/den; need num/den to be a perfect square of a rational
    # i.e., num * den must be a perfect square
    return _is_square(num * den)

def _no_three_collinear(pts):
    """Check no three points are collinear. pts are (x, y) Fractions."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                # Cross product
                det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if det == 0:
                    return False
    return True

def _no_four_concyclic(pts, D):
    """Check no four points are concyclic. pts are (x, y) Fractions, D is the discriminant."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    # General circle: x^2 + y^2*D + Ax + By + C = 0
                    # For each point: x_i^2 + y_i^2*D + A*x_i + B*y_i + C = 0
                    # This is a linear system in A, B, C for 3 points, check 4th
                    # Using 3 points to determine circle, check if 4th lies on it
                    # Determinant of 4x4 matrix must be nonzero for no concyclicity
                    # | x_i^2 + D*y_i^2, x_i, y_i, 1 |
                    # | x_j^2 + D*y_j^2, x_j, y_j, 1 |
                    # | x_k^2 + D*y_k^2, x_k, y_k, 1 |
                    # | x_l^2 + D*y_l^2, x_l, y_l, 1 |
                    # Compute determinant
                    a1 = x1*x1 + D*y1*y1
                    b1 = x1
                    c1 = y1
                    a2 = x2*x2 + D*y2*y2
                    b2 = x2
                    c2 = y2
                    a3 = x3*x3 + D*y3*y3
                    b3 = x3
                    c3 = y3
                    a4 = x4*x4 + D*y4*y4
                    b4 = x4
                    c4 = y4
                    # 4x4 determinant via expansion
                    # | a1 b1 c1 1 |
                    # | a2 b2 c2 1 |
                    # | a3 b3 c3 1 |
                    # | a4 b4 c4 1 |
                    # Subtract row1 from rows 2,3,4:
                    # | a1 b1 c1 1 |
                    # | a2-a1 b2-b1 c2-c1 0 |
                    # | a3-a1 b3-b1 c3-c1 0 |
                    # | a4-a1 b4-b1 c4-c1 0 |
                    # = a1*det3 - b1*det3' + c1*det3'' - 1*det3'''
                    # where det3 is 3x3 from last 3 rows, cols 1-3
                    da2 = a2 - a1; db2 = b2 - b1; dc2 = c2 - c1
                    da3 = a3 - a1; db3 = b3 - b1; dc3 = c3 - c1
                    da4 = a4 - a1; db4 = b4 - b1; dc4 = c4 - c1
                    # 3x3 det of [[da2,db2,dc2],[da3,db3,dc3],[da4,db4,dc4]]
                    d3 = (da2*(db3*dc4 - db4*dc3)
                         - db2*(da3*dc4 - da4*dc3)
                         + dc2*(da3*db4 - da4*db3))
                    # -b1 * det of [[da2,dc2,0],[da3,dc3,0],[da4,dc4,0]] with col swap
                    # Actually: -b1 * det([[da2,dc2],[da3,dc3],[da4,dc4]])... no
                    # Let me just compute the 4x4 determinant directly via cofactor expansion on last column
                    # det = 1 * C14 + 1 * C24 + 1 * C34 + 1 * C44 where Cij is cofactor
                    # C14 = (-1)^(1+4) * M14 = -det([[a2,b2,c2],[a3,b3,c3],[a4,b4,c4]])
                    # C24 = (-1)^(2+4) * M24 = +det([[a1,b1,c1],[a3,b3,c3],[a4,b4,c4]])
                    # C34 = (-1)^(3+4) * M34 = -det([[a1,b1,c1],[a2,b2,c2],[a4,b4,c4]])
                    # C44 = (-1)^(4+4) * M44 = +det([[a1,b1,c1],[a2,b2,c2],[a3,b3,c3]])
                    def det3(a, b, c, d, e, f, g, h, i):
                        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
                    M14 = det3(a2, b2, c2, a3, b3, c3, a4, b4, c4)
                    M24 = det3(a1, b1, c1, a3, b3, c3, a4, b4, c4)
                    M34 = det3(a1, b1, c1, a2, b2, c2, a4, b4, c4)
                    M44 = det3(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                    det = -M14 + M24 - M34 + M44
                    if det == 0:
                        return False
    return True

def _validate(pts, D):
    """Full validation: distinct, rational distances, no 3 collinear, no 4 concyclic."""
    n = len(pts)
    if n < 3:
        return False
    # Distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if pts[i] == pts[j]:
                return False
    # Rational distances
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_rational_dist(pts[i], pts[j], D):
                return False
    # No three collinear
    if not _no_three_collinear(pts):
        return False
    # No four concyclic
    if not _no_four_concyclic(pts, D):
        return False
    return True

def _try_add_point(pts, D, new_pt):
    """Check if adding new_pt to pts maintains all constraints."""
    new_pts = pts + [new_pt]
    # Check distance from new point to all existing
    for p in pts:
        if not _is_rational_dist(new_pt, p, D):
            return False
    # Check no three collinear involving new point
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = new_pt
            x2, y2 = pts[i]
            x3, y3 = pts[j]
            det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if det == 0:
                return False
    # Check no four concyclic involving new point
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = new_pt
                x2, y2 = pts[i]
                x3, y3 = pts[j]
                x4, y4 = pts[k]
                a1 = x1*x1 + D*y1*y1
                b1 = x1
                c1 = y1
                a2 = x2*x2 + D*y2*y2
                b2 = x2
                c2 = y2
                a3 = x3*x3 + D*y3*y3
                b3 = x3
                c3 = y3
                a4 = x4*x4 + D*y4*y4
                b4 = x4
                c4 = y4
                def det3(a, b, c, d, e, f, g, h, i):
                    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
                M14 = det3(a2, b2, c2, a3, b3, c3, a4, b4, c4)
                M24 = det3(a1, b1, c1, a3, b3, c3, a4, b4, c4)
                M34 = det3(a1, b1, c1, a2, b2, c2, a4, b4, c4)
                M44 = det3(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                det = -M14 + M24 - M34 + M44
                if det == 0:
                    return False
    return True

def _search_new_point(pts, D, max_coord=200):
    """Try to find a new point that can be added to pts."""
    n = len(pts)
    # Try all integer lattice points in a bounded region
    # For each candidate, check if it works
    best = None
    for x in range(-max_coord, max_coord + 1):
        if time.time() > _DEADLINE:
            break
        for y in range(-max_coord, max_coord + 1):
            if time.time() > _DEADLINE:
                break
            cand = (Fraction(x), Fraction(y))
            # Quick check: not already in pts
            if cand in pts:
                continue
            # Check distances to all existing points
            ok = True
            for p in pts:
                if not _is_rational_dist(cand, p, D):
                    ok = False
                    break
            if not ok:
                continue
            # Check no three collinear
            collinear = False
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = cand
                    x2, y2 = pts[i]
                    x3, y3 = pts[j]
                    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if det == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            # Check no four concyclic
            concyclic = False
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        x1, y1 = cand
                        x2, y2 = pts[i]
                        x3, y3 = pts[j]
                        x4, y4 = pts[k]
                        a1 = x1*x1 + D*y1*y1
                        b1 = x1
                        c1 = y1
                        a2 = x2*x2 + D*y2*y2
                        b2 = x2
                        c2 = y2
                        a3 = x3*x3 + D*y3*y3
                        b3 = x3
                        c3 = y3
                        a4 = x4*x4 + D*y4*y4
                        b4 = x4
                        c4 = y4
                        def det3(a, b, c, d, e, f, g, h, i):
                            return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
                        M14 = det3(a2, b2, c2, a3, b3, c3, a4, b4, c4)
                        M24 = det3(a1, b1, c1, a3, b3, c3, a4, b4, c4)
                        M34 = det3(a1, b1, c1, a2, b2, c2, a4, b4, c4)
                        M44 = det3(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                        det = -M14 + M24 - M34 + M44
                        if det == 0:
                            concyclic = True
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
            if concyclic:
                continue
            # Found a valid point
            return cand
    return None

def _search_with_expanding_radius(pts, D, start_radius=50, max_radius=500):
    """Search for new point with expanding radius."""
    for r in range(start_radius, max_radius + 1, 10):
        if time.time() > _DEADLINE:
            break
        # Search in a ring
        for x in range(-r, r + 1):
            if time.time() > _DEADLINE:
                break
            for y in range(-r, r + 1):
                if time.time() > _DEADLINE:
                    break
                cand = (Fraction(x), Fraction(y))
                if cand in pts:
                    continue
                ok = True
                for p in pts:
                    if not _is_rational_dist(cand, p, D):
                        ok = False
                        break
                if not ok:
                    continue
                # Check no three collinear
                n = len(pts)
                collinear = False
                for i in range(n):
                    for j in range(i + 1, n):
                        x1, y1 = cand
                        x2, y2 = pts[i]
                        x3, y3 = pts[j]
                        det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                        if det == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                # Check no four concyclic
                concyclic = False
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            x1, y1 = cand
                            x2, y2 = pts[i]
                            x3, y3 = pts[j]
                            x4, y4 = pts[k]
                            a1 = x1*x1 + D*y1*y1
                            b1 = x1
                            c1 = y1
                            a2 = x2*x2 + D*y2*y2
                            b2 = x2
                            c2 = y2
                            a3 = x3*x3 + D*y3*y3
                            b3 = x3
                            c3 = y3
                            a4 = x4*x4 + D*y4*y4
                            b4 = x4
                            c4 = y4
                            def det3(a, b, c, d, e, f, g, h, i):
                                return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
                            M14 = det3(a2, b2, c2, a3, b3, c3, a4, b4, c4)
                            M24 = det3(a1, b1, c1, a3, b3, c3, a4, b4, c4)
                            M34 = det3(a1, b1, c1, a2, b2, c2, a4, b4, c4)
                            M44 = det3(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                            det = -M14 + M24 - M34 + M44
                            if det == 0:
                                concyclic = True
                                break
                        if concyclic:
                            break
                    if concyclic:
                        break
                if concyclic:
                    continue
                return cand
    return None

def construct():
    start_time = time.time()
    
    # Start with known 5-point set
    D = 1
    pts = [(Fraction(0), Fraction(0)),
           (Fraction(-56), Fraction(33)),
           (Fraction(-16), Fraction(63)),
           (Fraction(0), Fraction(33)),
           (Fraction(16), Fraction(63))]
    
    best_pts = pts[:]
    
    # Verify the seed
    assert _validate(best_pts, D), "Seed validation failed"
    
    # Try to add points one at a time
    current_pts = pts[:]
    for target_size in range(6, 13):
        if time.time() > _DEADLINE - 10:
            break
        found = False
        # Try multiple strategies
        # Strategy 1: Search in expanding radius
        new_pt = _search_with_expanding_radius(current_pts, D, start_radius=50, max_radius=300)
        if new_pt is not None:
            current_pts = current_pts + [new_pt]
            found = True
            if len(current_pts) > len(best_pts):
                best_pts = current_pts[:]
        
        if not found:
            # Try with a smaller initial radius for finer search
            if time.time() < _DEADLINE - 10:
                new_pt = _search_new_point(current_pts, D, max_coord=150)
                if new_pt is not None:
                    current_pts = current_pts + [new_pt]
                    if len(current_pts) > len(best_pts):
                        best_pts = current_pts[:]
        
        if time.time() > _DEADLINE - 10:
            break
    
    # Final validation
    if not _validate(best_pts, D):
        # Fallback to seed
        best_pts = [(Fraction(0), Fraction(0)),
                    (Fraction(-56), Fraction(33)),
                    (Fraction(-16), Fraction(63)),
                    (Fraction(0), Fraction(33)),
                    (Fraction(16), Fraction(63))]
        D = 1
    
    # Format output
    points_out = []
    for x, y in best_pts:
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': points_out}
