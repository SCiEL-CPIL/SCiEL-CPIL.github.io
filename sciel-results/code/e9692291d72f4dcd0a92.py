"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # We use D = 1 (integer lattice). Points are (x, y) with integer coords.
    # Pairwise squared distances must be perfect squares (rational distances = integer here).
    # No three collinear, no four concyclic.

    # Start with a known valid 5-point set.
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Verify: distances squared:
    # (0,0)-(-56,33): 3136+1089=4225=65^2
    # (0,0)-(-16,63): 256+3969=4225=65^2
    # (0,0)-(0,33): 1089=33^2
    # (0,0)-(16,63): 256+3969=4225=65^2
    # (-56,33)-(-16,63): 1600+900=2500=50^2
    # (-56,33)-(0,33): 3136=56^2
    # (-56,33)-(16,63): 6400+900=7300... wait
    # Let me recheck. Actually the seed says this is valid.
    # (-56,33) to (16,63): dx=72, dy=30, dist^2=5184+900=6084=78^2. Yes!
    # (-16,63) to (0,33): dx=16, dy=-30, dist^2=256+900=1156=34^2
    # (-16,63) to (16,63): dx=32, dy=0, dist^2=1024=32^2
    # (0,33) to (16,63): dx=16, dy=30, dist^2=256+900=1156=34^2
    # All good.

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist2_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def are_collinear(p1, p2, p3):
        # (p2-p1) x (p3-p1) = 0
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def are_concyclic(p1, p2, p3, p4):
        # Four points are concyclic iff the determinant of the circle equation is 0.
        # x^2 + y^2 + Dx + Ey + F = 0
        # For point (x,y): x^2+y^2 + Dx + Ey + F = 0
        # We have 4 equations, 3 unknowns (D,E,F). The system is consistent iff det = 0.
        # Actually, use the standard 4x4 determinant:
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 | = 0
        def det4(m):
            # 4x4 determinant via cofactor expansion
            # m is 4x4 list of lists
            def det3(a):
                return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
                      - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
                      + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
            def det2(a):
                return a[0][0]*a[1][1] - a[0][1]*a[1][0]
            # Expand along first row
            result = 0
            for j in range(4):
                minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
                det_minor = det3(minor)
                result += ((-1) ** j) * m[0][j] * det_minor
            return result

        pts = [p1, p2, p3, p4]
        m = []
        for p in pts:
            x, y = p
            m.append([x*x + y*y, x, y, 1])
        return det4(m) == 0

    def is_valid(points):
        n = len(points)
        # Check all pairwise distances are rational (perfect square squared distances)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist2_sq(points[i], points[j])
                if d2 == 0:
                    return False  # duplicate
                if not is_perfect_square(d2):
                    return False
        # No three collinear
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        # No four concyclic
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Initial 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    # Generate candidate points: integer lattice points within some radius
    # that have perfect-square distances to all existing best points.
    # We'll do a greedy search: try to add points one at a time.

    # Precompute a set of "good" candidate points in a reasonable range.
    # For efficiency, we'll search in expanding shells.

    max_coord = 200  # search range for new points

    # Generate all lattice points in [-max_coord, max_coord]^2
    # Filter to those that are at perfect-square distance from all current points.
    # Then try to add them greedily.

    def try_add_point(current_points, candidate):
        """Check if adding candidate to current_points keeps validity."""
        new_points = current_points + [candidate]
        return is_valid(new_points)

    # Greedy growth: repeatedly find the best candidate to add.
    # "Best" = one that doesn't break validity. We try many candidates.

    # To be efficient, pre-filter candidates against the current set.
    def compatible_with_all(p, pts):
        for q in pts:
            d2 = dist2_sq(p, q)
            if d2 == 0:
                return False
            if not is_perfect_square(d2):
                return False
        return True

    # Search for points to add
    # We'll iterate through candidate points and try to add them.
    # Order candidates by some heuristic (e.g., distance from origin, or just enumerate).

    # Generate candidate list
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))

    # Sort by distance from origin for deterministic ordering
    candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])

    # Greedy: try to add points one by one
    current = list(best_points)
    added = 0

    # We'll do multiple passes, each time trying to add one more point.
    # In each pass, iterate through candidates and try to add the first compatible one.
    # Then repeat until no more can be added or we hit the deadline.

    while time.time() < deadline:
        found = False
        for cand in candidates:
            if cand in current:
                continue
            # Quick check: distance to all current points must be perfect square
            ok = True
            for p in current:
                d2 = dist2_sq(cand, p)
                if d2 == 0:
                    ok = False
                    break
                if not is_perfect_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            # Check collinearity with all pairs in current
            ok = True
            n = len(current)
            for i in range(n):
                for j in range(i + 1, n):
                    if are_collinear(cand, current[i], current[j]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check concyclicity with all triples in current
            ok = True
            n = len(current)
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        if are_concyclic(cand, current[i], current[j], current[k]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Valid! Add it.
            current.append(cand)
            found = True
            added += 1
            break  # restart the candidate loop with the new set

        if not found:
            # No more points can be added with this candidate set.
            # Try expanding the search range.
            if max_coord < 500:
                max_coord += 50
                # Regenerate candidates
                candidates = []
                for x in range(-max_coord, max_coord + 1):
                    for y in range(-max_coord, max_coord + 1):
                        candidates.append((x, y))
                candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
                # Remove already-added points from consideration (they'll be skipped anyway)
            else:
                break

    # Also try a different strategy: use the known 5-point set and try to add
    # points from a broader set using a different ordering.
    # But first, let's see what we got.

    if len(current) > len(best_points):
        best_points = current

    # Final verification
    if not is_valid(best_points):
        # Fallback to the known 5-point set
        best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
