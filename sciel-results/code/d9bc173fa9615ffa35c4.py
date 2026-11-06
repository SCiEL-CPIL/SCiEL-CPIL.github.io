import time
from fractions import Fraction
from math import isqrt
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = 115.0
    
    def to_frac(n, d=1):
        if d == 1:
            return Fraction(n)
        return Fraction(n, d)
    
    def is_rational_square(fr):
        """Check if a non-negative Fraction is a perfect square of a rational."""
        if fr < 0:
            return False
        if fr == 0:
            return True
        num = fr.numerator
        den = fr.denominator
        # num/den must be a perfect square
        r_num = isqrt(num)
        r_den = isqrt(den)
        return r_num * r_num == num and r_den * r_den == den
    
    def dist_sq(p, q, D):
        """p, q are (Fraction, Fraction). Returns Fraction."""
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + D * dy * dy
    
    def no_three_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                    if cross == 0:
                        return False
        return True
    
    def no_four_concyclic(points, D):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        # Circle: x² + D·y² + Ax + By + C = 0
                        a1 = x2 - x1
                        b1 = y2 - y1
                        c1 = -(x2*x2 - x1*x1) - D*(y2*y2 - y1*y1)
                        a2 = x3 - x1
                        b2 = y3 - y1
                        c2 = -(x3*x3 - x1*x1) - D*(y3*y3 - y1*y1)
                        det = a1 * b2 - a2 * b1
                        if det == 0:
                            continue
                        A_num = c1 * b2 - c2 * b1
                        B_num = a1 * c2 - a2 * c1
                        rhs = x1*x1 + D*y1*y1 - x4*x4 - D*y4*y4
                        lhs = A_num * (x4 - x1) + B_num * (y4 - y1)
                        if lhs == det * rhs:
                            return False
        return True
    
    def validate(points, D):
        n = len(points)
        if n < 3:
            return False
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq(points[i], points[j], D)
                if not is_rational_square(d2):
                    return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points, D):
            return False
        return True
    
    def format_points(points, D):
        return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]}
    
    # Known 5-point solution for D=1
    pts5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]
    
    best = None
    if validate(pts5, 1):
        best = format_points(pts5, 1)
    else:
        best = format_points([(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))], 1)

    # Search for larger sets in D=2 and D=3
    # Parametrization for x^2 + D y^2 = z^2:
    # x = m^2 - D n^2
    # y = 2 m n
    # z = m^2 + D n^2
    
    def get_displacements(D, max_m, max_n):
        disps = set()
        for m in range(1, max_m + 1):
            for n in range(0, max_n + 1):
                if m == 0 and n == 0:
                    continue
                # Generate primitive integer solutions
                x = m * m - D * n * n
                y = 2 * m * n
                # Add scaled versions to get rational coordinates with small denominators
                # We want points (x, y) such that distances are rational.
                # If we use integer vectors, distances are sqrt(int). We need sqrt(int) to be rational, so int must be square.
                # The parametrization ensures x^2 + D y^2 = z^2, so distance from origin is z (integer).
                # Distance between two such points P1, P2 is sqrt((x1-x2)^2 + D(y1-y2)^2).
                # This is NOT necessarily rational just because individual distances from origin are.
                # So we must check pairwise distances.
                
                # To keep numbers small, we can scale by 1/k.
                # Let's generate a pool of candidate points relative to origin.
                for k in [1, 2, 3, 4, 5]:
                    px = Fraction(x, k)
                    py = Fraction(y, k)
                    disps.add((px, py))
                    disps.add((-px, py))
                    disps.add((px, -py))
                    disps.add((-px, -py))
        return list(disps)

    def backtrack_search(D, target_n, max_depth, time_limit):
        start_time = time.time()
        disps = get_displacements(D, 15, 15)
        # Sort by magnitude to prefer "close" points, which might reduce concyclicity issues? Or maybe large spread is better?
        # Let's try sorting by x^2 + D*y^2 (distance from origin squared)
        def mag_sq(p):
            return p[0]*p[0] + D*p[1]*p[1]
        disps.sort(key=mag_sq)
        
        # We start with origin
        current_set = [(Fraction(0), Fraction(0))]
        best_found = None
        
        # To speed up, we can pre-filter disps that are "compatible" with the current set.
        # But dynamic filtering is complex. Let's just iterate.
        
        def is_valid_addition(pt, current_set, D):
            # Check distinct
            if pt in current_set:
                return False
            # Check distances
            for p in current_set:
                d2 = dist_sq(pt, p, D)
                if not is_rational_square(d2):
                    return False
            # Check collinearity
            new_set = current_set + [pt]
            if not no_three_collinear(new_set):
                return False
            # Check concyclicity
            if len(new_set) >= 4 and not no_four_concyclic(new_set, D):
                return False
            return True

        def search(current_set, depth):
            nonlocal best_found
            if time.time() - start_time > time_limit:
                return
            
            if len(current_set) > (len(best_found) if best_found else 0):
                best_found = list(current_set)
            
            if len(current_set) >= target_n:
                return
                
            # Try adding each displacement
            # Optimization: Only consider disps that haven't been tried in this branch?
            # Since we are building from origin, the "position" of the new point is just the displacement vector.
            # But wait, the set is not necessarily centered at origin in a way that all points are displacements from origin.
            # The points are absolute coordinates.
            # If I add p1, p2, p3... they are absolute.
            # So the candidate for the next point is NOT just a displacement from origin.
            # It is ANY point in the plane.
            # However, if all pairwise distances are rational, the set can be translated.
            # So we can fix one point at (0,0).
            # The other points must be at rational distances from (0,0).
            # So the other points MUST be in the set of points at rational distance from (0,0).
            # For D=2, points (x,y) with x^2 + 2y^2 = rational square.
            # Our `disps` list contains such points.
            # So yes, we can just pick the next point from `disps`.
            
            for pt in disps:
                if is_valid_addition(pt, current_set, D):
                    search(current_set + [pt], depth + 1)
                    if time.time() - start_time > time_limit:
                        return
                    if best_found and len(best_found) >= target_n:
                        return

        search(current_set, 0)
        return best_found

    # Try D=2
    if time.time() - t0 < deadline - 10:
        res = backtrack_search(2, 8, 8, deadline - 10 - (time.time() - t0))
        if res and len(res) > (len(best['points']) if best else 0):
            if validate(res, 2):
                best = format_points(res, 2)

    # Try D=3
    if time.time() - t0 < deadline - 5:
        res = backtrack_search(3, 8, 8, deadline - 5 - (time.time() - t0))
        if res and len(res) > (len(best['points']) if best else 0):
            if validate(res, 3):
                best = format_points(res, 3)

    return best
