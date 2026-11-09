"""Rational-distance point sets in Q(sqrt(D)) via integer lattice search."""
from fractions import Fraction
import itertools
import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 120  # seconds; leave margin before 150s kill

    # Known 5-point set from seed (D=1, integer lattice)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    def is_square(n):
        if n < 0:
            return False
        r = int(n ** 0.5)
        return r * r == n or (r + 1) * (r + 1) == n

    def dist2(p, q):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + dy * dy

    def are_collinear(a, b, c):
        # Cross product (b-a) x (c-a) == 0
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0

    def are_concyclic(a, b, c, d):
        # Four points concyclic iff the determinant is zero
        # Using the general circle equation: x^2+y^2+Dx+Ey+F=0
        # |x y x^2+y^2 1|
        # determinant of 4x4 matrix
        xa, ya = a
        xb, yb = b
        xc, yc = c
        xd, yd = d
        aa = xa*xa + ya*ya
        ab = xb*xb + yb*yb
        ac = xc*xc + yc*yc
        ad = xd*xd + yd*yd
        # det of [[xa,ya,aa,1],[xb,yb,ab,1],[xc,yc,ac,1],[xd,yd,ad,1]]
        # Expand along last column (all 1s):
        # = det3(xa,ya,aa; xb,yb,ab; xc,yc,ac)
        # - det3(xa,ya,aa; xb,yb,ab; xd,yd,ad)
        # + det3(xa,ya,aa; xc,yc,ac; xd,yd,ad)
        # - det3(xb,yb,ab; xc,yc,ac; xd,yd,ad)
        def det3(r1, r2, r3):
            return (r1[0]*(r2[1]*r3[2]-r2[2]*r3[1])
                    - r1[1]*(r2[0]*r3[2]-r2[2]*r3[0])
                    + r1[2]*(r2[0]*r3[1]-r2[1]*r3[0]))
        r1 = (xa, ya, aa)
        r2 = (xb, yb, ab)
        r3 = (xc, yc, ac)
        r4 = (xd, yd, ad)
        return (det3(r1, r2, r3)
                - det3(r1, r2, r4)
                + det3(r1, r3, r4)
                - det3(r2, r3, r4)) == 0

    def valid_config(pts):
        n = len(pts)
        # Check distinct
        for i in range(n):
            for j in range(i+1, n):
                if pts[i] == pts[j]:
                    return False
        # Check rational distances (integer dist2 must be a perfect square)
        for i in range(n):
            for j in range(i+1, n):
                d2 = dist2(pts[i], pts[j])
                if d2 <= 0:
                    return False
                if not is_square(d2):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(pts[i], pts[j], pts[k]):
                        return False
        # Check no four concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(pts[i], pts[j], pts[k], pts[l]):
                            return False
        return True

    def check_pairwise_dist(pts, new_pt):
        """Check that new_pt has rational distance to all existing points."""
        for p in pts:
            d2 = dist2(p, new_pt)
            if d2 <= 0:
                return False
            if not is_square(d2):
                return False
        return True

    def check_no_collinear(pts, new_pt):
        """Check no three collinear involving new_pt."""
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if are_collinear(pts[i], pts[j], new_pt):
                    return False
        return True

    def check_no_concyclic(pts, new_pt):
        """Check no four concyclic involving new_pt."""
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_concyclic(pts[i], pts[j], pts[k], new_pt):
                        return False
        return True

    def can_add(pts, new_pt):
        if not check_pairwise_dist(pts, new_pt):
            return False
        if not check_no_collinear(pts, new_pt):
            return False
        if not check_no_concyclic(pts, new_pt):
            return False
        return True

    # Generate candidate lattice points
    # Use a range that's large enough to find good configurations
    MAX_COORD = 200
    candidates = []
    for x in range(-MAX_COORD, MAX_COORD + 1):
        for y in range(-MAX_COORD, MAX_COORD + 1):
            candidates.append((x, y))

    # Pre-filter: keep pairs with integer distance
    # Build adjacency: for each point, which other points have integer distance
    # This is expensive for 400x400, so use a smarter approach

    # Strategy: greedy construction with random restarts
    # Start from random seeds, try to grow the set

    best_points = seed_points[:]  # fallback
    best_count = len(seed_points)

    # Try to grow from the seed
    def try_greedy_grow(initial_pts, candidate_pool, max_add=7):
        pts = initial_pts[:]
        for _ in range(max_add):
            if time.time() - start_time > DEADLINE - 5:
                break
            added = False
            # Shuffle candidates for diversity
            shuffled = candidate_pool[:]
            random.shuffle(shuffled)
            for cp in shuffled:
                if cp in pts:
                    continue
                if can_add(pts, cp):
                    pts.append(cp)
                    added = True
                    break
            if not added:
                break
        return pts

    # Try growing from seed
    random.seed(42)
    result = try_greedy_grow(seed_points, candidates, max_add=7)
    if valid_config(result):
        if len(result) > best_count:
            best_points = result[:]
            best_count = len(result)

    # Try random initial pairs and grow
    for attempt in range(200):
        if time.time() - start_time > DEADLINE - 10:
            break
        # Pick random starting pair with integer distance
        p1 = random.choice(candidates)
        # Find p2 with integer distance to p1
        valid_p2 = []
        for p2 in candidates:
            if p2 != p1 and is_square(dist2(p1, p2)):
                valid_p2.append(p2)
        if not valid_p2:
            continue
        p2 = random.choice(valid_p2)
        initial = [p1, p2]
        result = try_greedy_grow(initial, candidates, max_add=10)
        if valid_config(result) and len(result) > best_count:
            best_points = result[:]
            best_count = len(result)

    # Also try starting from the seed and doing local search
    if time.time() - start_time < DEADLINE - 15:
        # Try to find 6th point for the seed more aggressively
        seed_set = set(seed_points)
        for x in range(-300, 301):
            if time.time() - start_time > DEADLINE - 10:
                break
            for y in range(-300, 301):
                if time.time() - start_time > DEADLINE - 10:
                    break
                pt = (x, y)
                if pt in seed_set:
                    continue
                if can_add(seed_points, pt):
                    new_config = seed_points + [pt]
                    if valid_config(new_config):
                        if len(new_config) > best_count:
                            best_points = new_config[:]
                            best_count = len(new_config)
                        break

    # Final: try to grow best_points further
    if time.time() - start_time < DEADLINE - 5:
        result = try_greedy_grow(best_points, candidates, max_add=5)
        if valid_config(result) and len(result) > best_count:
            best_points = result[:]
            best_count = len(result)

    # Convert to required format
    # Points are (x, y) integers -> (x/1, y/1) with D=1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]
    return {'D': 1, 'points': points_out}
