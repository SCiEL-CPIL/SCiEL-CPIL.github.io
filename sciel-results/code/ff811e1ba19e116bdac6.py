"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations
import time

def _is_squarefree(n):
    if n <= 1:
        return True
    x = n
    p = 2
    while p * p <= x:
        if x % (p * p) == 0:
            return False
        while x % p == 0:
            x //= p
        p += 1 if p == 2 else 2
    return True

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    # We will search for sets of integer points (D=1) with integer pairwise distances.
    # This is a special case of rational distances (since 1 is rational).
    # We need no three collinear and no four concyclic.

    # Known small sets of integer-distance points (Erdos-Anning type, but we need general position).
    # The seed has 5 points. Can we find 6, 7, 8?

    # Strategy:
    # 1. Start with a base set of points.
    # 2. Try to add a new point P such that dist(P, Q) is integer for all Q in set.
    # 3. Check general position constraints.
    # 4. Repeat.

    # To make this efficient, we can use a precomputed list of candidate points
    # or a local search.

    # Let's use a known 6-point configuration if it exists, or find one.
    # Actually, let's try a systematic search for a 6th point to the seed's 5 points.
    # Seed points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Let's check if there's a 6th point.

    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Check if base is valid
    def check_valid(pts):
        n = len(pts)
        # Check integer distances
        for i in range(n):
            for j in range(i+1, n):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d2 = dx*dx + dy*dy
                if not _is_square(d2):
                    return False
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    if (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0:
                        return False
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        x4, y4 = pts[l]
                        # Circle equation: x^2+y^2 + Dx + Ey + F = 0
                        # Subtract to eliminate x^2+y^2
                        # (x2^2+y2^2 - x1^2-y1^2) + D(x2-x1) + E(y2-y1) = 0
                        # (x3^2+y3^2 - x1^2-y1^2) + D(x3-x1) + E(y3-y1) = 0
                        # Solve for D, E
                        a1 = x2-x1; b1 = y2-y1; c1 = x2*x2+y2*y2 - x1*x1-y1*y1
                        a2 = x3-x1; b2 = y3-y1; c2 = x3*x3+y3*y3 - x1*x1-y1*y1
                        det = a1*b2 - a2*b1
                        if det == 0:
                            continue # collinear, already checked
                        D = (c1*b2 - c2*b1) / det
                        E = (a1*c2 - a2*c1) / det
                        # Check if (x4,y4) is on the circle
                        # x4^2+y4^2 + D*x4 + E*y4 + F = 0
                        # F = -x1^2-y1^2 - D*x1 - E*y1
                        # So we check: x4^2+y4^2 - x1^2-y1^2 + D*(x4-x1) + E*(y4-y1) == 0
                        # Multiply by det to avoid floats
                        lhs = (x4*x4+y4*y4 - x1*x1-y1*y1)*det + D_num*(x4-x1) + E_num*(y4-y1)
                        # Wait, D and E are fractions. Let's do exact rational.
                        # D = (c1*b2 - c2*b1) / det
                        # E = (a1*c2 - a2*c1) / det
                        # Check: (x4^2+y4^2 - x1^2-y1^2)*det + (c1*b2 - c2*b1)*(x4-x1) + (a1*c2 - a2*c1)*(y4-y1) == 0
                        val = (x4*x4+y4*y4 - x1*x1-y1*y1)*det + (c1*b2 - c2*b1)*(x4-x1) + (a1*c2 - a2*c1)*(y4-y1)
                        if val == 0:
                            return False
        return True

    # We need to search for a 6th point.
    # Let's search in a bounding box around the centroid.
    # Centroid is roughly (0, 38).
    # Let's search x in [-100, 100], y in [-50, 100].
    
    found_6 = None
    for x in range(-100, 101):
        for y in range(-50, 101):
            if (x, y) in base_points:
                continue
            new_pts = base_points + [(x, y)]
            # Quick check: integer distance to all base points
            ok = True
            for px, py in base_points:
                dx = x - px
                dy = y - py
                if not _is_square(dx*dx + dy*dy):
                    ok = False
                    break
            if not ok:
                continue
            # Check general position
            if check_valid(new_pts):
                found_6 = (x, y)
                break
        if found_6:
            break

    if found_6:
        points = base_points + [found_6]
        # Try to find a 7th point
        found_7 = None
        for x in range(-150, 151):
            if time.time() > deadline:
                break
            for y in range(-100, 151):
                if time.time() > deadline:
                    break
                if (x, y) in points:
                    continue
                new_pts = points + [(x, y)]
                ok = True
                for px, py in points:
                    dx = x - px
                    dy = y - py
                    if not _is_square(dx*dx + dy*dy):
                        ok = False
                        break
                if not ok:
                    continue
                if check_valid(new_pts):
                    found_7 = (x, y)
                    break
            if found_7:
                break
        if found_7:
            points = points + [found_7]

    # Format output
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END
