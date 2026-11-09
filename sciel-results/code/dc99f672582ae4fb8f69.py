"""Search for integral point sets in general position."""
import time
from fractions import Fraction
from math import isqrt, gcd

_DEADLINE = 115.0  # seconds, safety margin under 120s
_START = None


def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def _is_squarefree(n):
    if n <= 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % (p * p) == 0:
            return False
        p += 1 if p == 2 else 2
    return True


def _sqfree_part(n):
    """Return (D, k) with n = D*k^2 and D squarefree."""
    D = 1
    k = 1
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            D *= p
            while n % (p * p) == 0:
                n //= (p * p)
                k *= p
        p += 1 if p == 2 else 2
    if n > 1:
        D *= n
    return D, k


def _check_all(pts, D):
    """Full exact check. pts = list of (x_num, x_den, y_num, y_den)."""
    n = len(pts)
    if n < 3:
        return False
    # distinct
    seen = set()
    for x, xd, y, yd in pts:
        xf = Fraction(x, xd)
        yf = Fraction(y, yd)
        if (xf, yf) in seen:
            return False
        seen.add((xf, yf))
    # pairwise distances rational
    for i in range(n):
        xi, xid, yi, yid = pts[i]
        for j in range(i + 1, n):
            xj, xjd, yj, yjd = pts[j]
            dx = Fraction(xi, xid) - Fraction(xj, xjd)
            dy = Fraction(yi, yid) - Fraction(yj, yjd)
            num = dx.numerator * dx.numerator * xid * xid * xjd * xjd \
                  + dy.numerator * dy.numerator * D * yid * yid * yjd * yjd
            den = xid * xjd * yid * yjd
            # distance^2 = num/den; need rational => num/den is a rational square
            # reduce
            g = gcd(num, den)
            num //= g
            den //= g
            # num/den must be a square of a rational
            if not _is_square(num) or not _is_square(den):
                return False
    # no three collinear
    for i in range(n):
        xi, xid, yi, yid = pts[i]
        for j in range(i + 1, n):
            xj, xjd, yj, yjd = pts[j]
            for k in range(j + 1, n):
                xk, xkd, yk, ykd = pts[k]
                # (xj-xi)(yk-yi) - (yj-yi)(xk-xi) == 0 ?
                ax = Fraction(xj, xjd) - Fraction(xi, xid)
                ay = Fraction(yj, yjd) - Fraction(yi, yid)
                bx = Fraction(xk, xkd) - Fraction(xi, xid)
                by = Fraction(yk, ykd) - Fraction(yi, yid)
                det = ax * by - ay * bx
                if det == 0:
                    return False
    # no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic(pts[i], pts[j], pts[k], pts[l], D):
                        return False
    return True


def _concyclic(p1, p2, p3, p4, D):
    """Check if 4 points (x, y*sqrt(D)) are concyclic using exact rationals."""
    x1, x1d, y1, y1d = p1
    x2, x2d, y2, y2d = p2
    x3, x3d, y3, y3d = p3
    x4, x4d, y4, y4d = p4
    X1 = Fraction(x1, x1d)
    Y1 = Fraction(y1, y1d)
    X2 = Fraction(x2, x2d)
    Y2 = Fraction(y2, y2d)
    X3 = Fraction(x3, x3d)
    Y3 = Fraction(y3, y3d)
    X4 = Fraction(x4, x4d)
    Y4 = Fraction(y4, y4d)
    # Circle through 1,2,3: x^2 + y^2*D + A x + B y sqrt(D) + C = 0
    # Using y^2 term: (Y*sqrt(D))^2 = Y^2 D
    # Equation: X^2 + D Y^2 + A X + B Y sqrt(D) + C = 0
    # This has a sqrt(D) term, so separate rational and sqrt parts:
    # (X^2 + D Y^2 + C) + A X + (B Y) sqrt(D) = 0
    # For this to hold for all points, we need the system:
    # X_i^2 + D Y_i^2 + A X_i + C = 0  (rational part)
    # B Y_i = 0  ... no wait, that's wrong.
    # Actually the equation is: X^2 + D*Y^2 + A*X + B*Y*sqrt(D) + C = 0
    # = (X^2 + D*Y^2 + C + A*X) + B*Y*sqrt(D) = 0
    # For this to be zero, we need BOTH: X^2+D*Y^2+C+A*X = 0 AND B*Y = 0
    # That's only true if B=0 or Y=0 for all points, which is too restrictive.
    # 
    # Better approach: use the determinant form. Four points are concyclic iff
    # the determinant of the circle equation is zero.
    # For points (x_i, y_i) in the plane where the metric is x^2 + D*y^2:
    # A circle is: x^2 + D*y^2 + A*x + B*y + C = 0  (B multiplies y, not y*sqrt(D))
    # Wait, the actual y-coordinate is y*sqrt(D), so (y*sqrt(D))^2 = D*y^2.
    # Circle: x^2 + (y*sqrt(D))^2 + A*x + B*(y*sqrt(D)) + C = 0
    # = x^2 + D*y^2 + A*x + B*sqrt(D)*y + C = 0
    # Let B' = B*sqrt(D). Then: x^2 + D*y^2 + A*x + B'*y + C = 0
    # For point i: X_i^2 + D*Y_i^2 + A*X_i + B'*Y_i + C = 0
    # This is linear in A, B', C. Solve for A, B', C from points 1,2,3 and check point 4.
    # Matrix:
    # [X1  Y1  1] [A]   = -(X1^2 + D*Y1^2)
    # [X2  Y2  1] [B']
    # [X3  Y3  1] [C]   = -(X3^2 + D*Y3^2)
    # But B' must be a multiple of sqrt(D), i.e., B'/sqrt(D) must be rational.
    # Actually, we just need to check if the 4 points lie on SOME circle in the
    # Euclidean plane with coordinates (x, y*sqrt(D)).
    # So treat points as (X_i, Z_i) where Z_i = Y_i * sqrt(D).
    # Circle: X^2 + Z^2 + A*X + B*Z + C = 0
    # X_i^2 + Z_i^2 + A*X_i + B*Z_i + C = 0
    # Z_i^2 = D * Y_i^2, so X_i^2 + D*Y_i^2 + A*X_i + B*Y_i*sqrt(D) + C = 0
    # Group rational and sqrt(D) parts:
    # (X_i^2 + D*Y_i^2 + C) + A*X_i + (B*Y_i)*sqrt(D) = 0
    # This must hold. Since sqrt(D) is irrational (D>1) or =1 (D=1):
    # If D=1: X_i^2 + Y_i^2 + A*X_i + B*Y_i + C = 0, standard.
    # If D>1: We need X_i^2 + D*Y_i^2 + C + A*X_i = 0 AND B*Y_i = 0 for all i.
    #   That means B=0 (unless all Y_i=0) and the points lie on an ellipse...
    #   Hmm, this doesn't seem right. Let me reconsider.
    #
    # The correct approach: work in the (x, z) plane where z = y*sqrt(D).
    # Points are (X_i, Z_i) with Z_i = Y_i*sqrt(D).
    # Four points are concyclic iff the circle determinant is zero.
    # det = |X1^2+Z1^2  X1  Z1  1|
    #       |X2^2+Z2^2  X2  Z2  1| = 0
    #       |X3^2+Z3^2  X3  Z3  1|
    #       |X4^2+Z4^2  X4  Z4  1|
    # Since Z_i = Y_i*sqrt(D), Z_i^2 = D*Y_i^2.
    # X_i^2 + Z_i^2 = X_i^2 + D*Y_i^2 (rational)
    # But Z_i itself is irrational (if D>1 and Y_i != 0).
    # So the determinant has both rational and sqrt(D) parts.
    # Let me compute it symbolically.
    # Actually, let me just use the fact that 4 points (x_i, z_i) are concyclic iff
    # there exist A, B, C (real) such that x_i^2 + z_i^2 + A*x_i + B*z_i + C = 0 for all i.
    # Substituting z_i = y_i*sqrt(D):
    # x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
    # = (x_i^2 + D*y_i^2 + C + A*x_i) + (B*y_i)*sqrt(D) = 0
    # If D is not a perfect square (which it isn't since D is squarefree and D>1),
    # then sqrt(D) is irrational, so we need:
    #   x_i^2 + D*y_i^2 + C + A*x_i = 0  for all i
    #   B*y_i = 0  for all i
    # The second condition means B=0 (if any y_i != 0) or all y_i = 0.
    # If B=0: x_i^2 + D*y_i^2 + A*x_i + C = 0 for all i.
    # This is 4 equations in 2 unknowns (A, C) - overdetermined.
    # Check consistency: solve for A, C from points 1,2 and verify 3,4.
    # If all y_i = 0, then all points are on x-axis, which means collinear (excluded).
    #
    # If D=1: standard circle check.
    if D == 1:
        # Standard concyclicity
        # det of 4x4 matrix
        rows = []
        for p in [p1, p2, p3, p4]:
            x, xd, y, yd = p
            X = Fraction(x, xd)
            Y = Fraction(y, yd)
            rows.append((X * X + Y * Y, X, Y, Fraction(1)))
        # Compute determinant
        # Use expansion
        a1, b1, c1, d1 = rows[0]
        a2, b2, c2, d2 = rows[1]
        a3, b3, c3, d3 = rows[2]
        a4, b4, c4, d4 = rows[3]
        det = (a1*(b2*(c3*d4 - c4*d3) - b3*(c2*d4 - c4*d2) + b4*(c2*d3 - c3*d2))
               - b1*(a2*(c3*d4 - c4*d3) - a3*(c2*d4 - c4*d2) + a4*(c2*d3 - c3*d2))
               + c1*(a2*(b3*d4 - b4*d3) - a3*(b2*d4 - b4*d2) + a4*(b2*d3 - b3*d2))
               - d1*(a2*(b3*c4 - b4*c3) - a3*(b2*c4 - b4*c2) + a4*(b2*c3 - b3*c2)))
        return det == 0
    else:
        # D > 1: need B=0 and x_i^2 + D*y_i^2 + A*x_i + C = 0
        # Check if all y_i = 0 (collinear, already excluded)
        # Otherwise B must be 0, so check if points lie on x^2 + D*y^2 + A*x + C = 0
        vals = []
        for p in [p1, p2, p3, p4]:
            x, xd, y, yd = p
            X = Fraction(x, xd)
            Y = Fraction(y, yd)
            vals.append((X * X + Fraction(D) * Y * Y, X))
        # Solve A*X1 + C = -v1, A*X2 + C = -v2
        X1, v1 = vals[0]
        X2, v2 = vals[1]
        if X1 == X2:
            # Can't solve uniquely; check consistency
            return v1 == v2 and vals[2][0] == v1 and vals[3][0] == v1
        A = (v2 - v1) / (X1 - X2)
        C = -v1 - A * X1
        for i in range(2, 4):
            Xi, vi = vals[i]
            if vi + A * Xi + C != 0:
                return False
        return True


def _try_add_point(pts, D, new_pt):
    """Quick check: can new_pt be added to pts?"""
    x, xd, y, yd = new_pt
    # Check distance to all existing points
    for px, pxd, py, pyd in pts:
        dx = Fraction(x, xd) - Fraction(px, pxd)
        dy = Fraction(y, yd) - Fraction(py, pyd)
        num = dx.numerator * dx.numerator * xd * xd * pxd * pxd \
              + dy.numerator * dy.numerator * D * yd * yd * pyd * pyd
        den = xd * pxd * yd * pyd
        g = gcd(num, den)
        num //= g
        den //= g
        if not _is_square(num) or not _is_square(den):
            return False
    # Check collinearity with all pairs
    n = len(pts)
    for i in range(n):
        xi, xid, yi, yid = pts[i]
        for j in range(i + 1, n):
            xj, xjd, yj, yjd = pts[j]
            ax = Fraction(x, xd) - Fraction(xi, xid)
            ay = Fraction(y, yd) - Fraction(yi, yid)
            bx = Fraction(xj, xjd) - Fraction(xi, xid)
            by = Fraction(yj, yjd) - Fraction(yi, yid)
            if ax * by - ay * bx == 0:
                return False
    # Check concyclicity with all triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic(pts[i], pts[j], pts[k], new_pt, D):
                    return False
    return True


def _gen_candidates(pts, D, max_coord=500):
    """Generate candidate points with rational coords in a bounded range."""
    candidates = []
    existing = set()
    for x, xd, y, yd in pts:
        existing.add((Fraction(x, xd), Fraction(y, yd)))
    # Try integer points first (fast)
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (Fraction(x, 1), Fraction(y, 1)) in existing:
                continue
            candidates.append((x, 1, y, 1))
    return candidates


def construct():
    t0 = _now()
    # Start with known 5-point configuration
    D = 1
    base_pts = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    best_pts = list(base_pts)
    best_n = len(base_pts)
    
    # Verify base
    if not _check_all(best_pts, D):
        # Fallback to a trivial valid 3-point set
        best_pts = [(0, 1, 0, 1), (3, 1, 4, 1), (0, 1, 12, 1)]
        best_n = 3
    
    # Try to add points greedily
    for attempt in range(20):
        if _now() > _DEADLINE - 5:
            break
        if best_n >= 12:
            break
        # Generate candidates
        cands = _gen_candidates(best_pts, D, max_coord=300)
        # Try to add one
        added = False
        for cand in cands:
            if _now() > _DEADLINE - 5:
                break
            if _try_add_point(best_pts, D, cand):
                best_pts.append(cand)
                best_n += 1
                added = True
                break
        if not added:
            # Try with smaller/larger ranges or different D
            # Try D=2,3,5,6,7,10,11,13,14,15,...
            break
    
    # If stuck at 5, try other D values with fresh search
    if best_n <= 5 and _now() < _DEADLINE - 10:
        for D_try in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
            if _now() > _DEADLINE - 5:
                break
            # Try to build a set with this D
            pts = [(0, 1, 0, 1)]
            # Add points one by one
            for _ in range(11):
                if _now() > _DEADLINE - 5:
                    break
                found = False
                for x in range(-200, 201):
                    if _now() > _DEADLINE - 5:
                        break
                    for y in range(-200, 201):
                        if _now() > _DEADLINE - 5:
                            break
                        cand = (x, 1, y, 1)
                        if _try_add_point(pts, D_try, cand):
                            pts.append(cand)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break
            if len(pts) > best_n:
                if _check_all(pts, D_try):
                    best_pts = pts
                    best_n = len(pts)
                    D = D_try
                    if best_n >= 8:
                        break
    
    # Final verification
    if not _check_all(best_pts, D):
        # Return the known valid 5-point set
        D = 1
        best_pts = base_pts
        best_n = 5
    
    return {'D': D, 'points': [[[x, xd], [y, yd]] for x, xd, y, yd in best_pts]}
