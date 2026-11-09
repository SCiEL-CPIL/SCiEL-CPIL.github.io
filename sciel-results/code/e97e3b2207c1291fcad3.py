"""Construct integral point sets in general position in Q(sqrt(D))."""
from fractions import Fraction
from math import gcd, isqrt

def _sq(n):
    return n * n

def _is_squarefree(n):
    if n < 1:
        return False
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _make_squarefree(n):
    while True:
        r = isqrt(n)
        found = False
        for d in range(2, r + 1):
            if n % (d * d) == 0:
                n //= (d * d)
                found = True
                break
        if not found:
            return n

def construct():
    # Known 5-point set with integer coordinates and integer pairwise distances
    pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify pairwise distances are integers
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = _sq(dx) + _sq(dy)
            if isqrt(d2) * isqrt(d2) != d2:
                return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    # Check no three collinear
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            for k in range(j + 1, len(pts)):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                if (bx - ax) * (cy - ay) == (by - ay) * (cx - ax):
                    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    # Check no four concyclic
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            for k in range(j + 1, len(pts)):
                for l in range(k + 1, len(pts)):
                    ax, ay = pts[i]
                    bx, by = pts[j]
                    cx, cy = pts[k]
                    dx, dy = pts[l]
                    # Circle through a,b,c: general equation x^2+y^2+Dx+Ey+F=0
                    # Solve for D,E,F
                    # (ax^2+ay^2) + D*ax + E*ay + F = 0
                    # etc.
                    A1 = 2 * ax; B1 = 2 * ay; C1 = -(ax * ax + ay * ay)
                    A2 = 2 * bx; B2 = 2 * by; C2 = -(bx * bx + by * by)
                    A3 = 2 * cx; B3 = 2 * cy; C3 = -(cx * cx + cy * cy)
                    # Solve:
                    # A1*D + B1*E + F = C1
                    # A2*D + B2*E + F = C2
                    # A3*D + B3*E + F = C3
                    # Subtract to eliminate F:
                    # (A1-A2)*D + (B1-B2)*E = C1-C2
                    # (A1-A3)*D + (B1-B3)*E = C1-C3
                    d11 = A1 - A2; d12 = B1 - B2; d13 = C1 - C2
                    d21 = A1 - A3; d22 = B1 - B3; d23 = C1 - C3
                    det = d11 * d22 - d12 * d21
                    if det != 0:
                        Dv = (d13 * d22 - d12 * d23) / det
                        Ev = (d11 * d23 - d13 * d21) / det
                        Fv = C1 - A1 * Dv - B1 * Ev
                        # Check if d is on circle
                        if (dx * dx + dy * dy) + Dv * dx + Ev * dy + Fv == 0:
                            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    # Try to find a 6th point with integer coordinates
    # Search in a bounded region
    found = None
    for x in range(-200, 201):
        for y in range(-200, 201):
            if (x, y) in pts:
                continue
            # Check distance to all existing points is integer
            ok = True
            for px, py in pts:
                dx = x - px
                dy = y - py
                d2 = _sq(dx) + _sq(dy)
                if isqrt(d2) * isqrt(d2) != d2:
                    ok = False
                    break
            if not ok:
                continue
            # Check no three collinear with any two existing
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    ax, ay = pts[i]
                    bx, by = pts[j]
                    if (x - ax) * (by - ay) == (y - ay) * (bx - ax):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check no four concyclic with any three existing
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    for k in range(j + 1, len(pts)):
                        ax, ay = pts[i]
                        bx, by = pts[j]
                        cx, cy = pts[k]
                        A1 = 2 * ax; B1 = 2 * ay; C1 = -(ax * ax + ay * ay)
                        A2 = 2 * bx; B2 = 2 * by; C2 = -(bx * bx + by * by)
                        A3 = 2 * cx; B3 = 2 * cy; C3 = -(cx * cx + cy * cy)
                        d11 = A1 - A2; d12 = B1 - B2; d13 = C1 - C2
                        d21 = A1 - A3; d22 = B1 - B3; d23 = C1 - C3
                        det = d11 * d22 - d12 * d21
                        if det != 0:
                            Dv = (d13 * d22 - d12 * d23) / det
                            Ev = (d11 * d23 - d13 * d21) / det
                            Fv = C1 - A1 * Dv - B1 * Ev
                            if (x * x + y * y) + Dv * x + Ev * y + Fv == 0:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                found = (x, y)
                break
        if found:
            break
    
    if found:
        pts.append(found)
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
