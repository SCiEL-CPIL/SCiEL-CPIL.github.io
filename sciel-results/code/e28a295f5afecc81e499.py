"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # 10s safety margin before 120s kill

    def gcd2(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_squarefree(n):
        if n <= 1:
            return True
        p = 2
        while p * p <= n:
            if n % (p * p) == 0:
                return False
            p += 1 if p == 2 else 2
        return True

    def make_frac(num, den=1):
        if den == 0:
            return None
        return Fraction(num, den)

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_valid(points, D):
        """
        points: list of (x_frac, y_frac)
        D: integer
        Returns True if all constraints are met.
        """
        n = len(points)
        if n < 3:
            return False
        
        # distinct
        seen = set()
        for (x, y) in points:
            key = (x, y)
            if key in seen:
                return False
            seen.add(key)

        # pairwise rational distances
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx * dx + D * dy * dy
                if d2 < 0:
                    return False
                # check perfect square rational
                # d2 = num / den
                # sqrt(d2) rational iff num*den is a square and den is a square?
                # Actually, sqrt(num/den) = sqrt(num*den)/den.
                # So we need num*den to be a perfect square.
                num = d2.numerator
                den = d2.denominator
                # Check if num * den is a perfect square
                prod = num * den
                if not is_perfect_square(prod):
                    return False
                # Also, the result is sqrt(prod)/den.
                # Since den is a square (it is a square of a rational denominator? No, den is int).
                # Wait. Fraction is reduced.
                # If d2 = a/b in lowest terms.
                # sqrt(a/b) is rational iff a and b are both perfect squares.
                # Proof: a/b = (c/d)^2 => a*d^2 = b*c^2. Since gcd(a,b)=1, a must be square, b must be square.
                if not is_perfect_square(num):
                    return False
                if not is_perfect_square(den):
                    return False

        # no three collinear
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    # cross product (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
                    if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                        return False

        # no four concyclic
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    for l in range(k + 1, n):
                        x4, y4 = points[l]
                        # Determinant of 4x4 matrix
                        # | x1 y1 x1^2+y1^2 1 |
                        # | x2 y2 x2^2+y2^2 1 |
                        # | x3 y3 x3^2+y3^2 1 |
                        # | x4 y4 x4^2+y4^2 1 | = 0
                        # Note: y^2 term in circle equation is y^2, but our coordinates are (x, y*sqrt(D)).
                        # Distance squared is (x1-x2)^2 + D(y1-y2)^2.
                        # This is NOT a standard Euclidean circle in the (x,y) plane unless D=1.
                        # For D != 1, "concyclic" means lying on a conic of the form:
                        # x^2 + D y^2 + Ax + By + C = 0
                        # Determinant:
                        # | x1 y1 x1^2+D*y1^2 1 |
                        # | x2 y2 x2^2+D*y2^2 1 |
                        # | x3 y3 x3^2+D*y3^2 1 |
                        # | x4 y4 x4^2+D*y4^2 1 | = 0
                        
                        a1 = x1 * x1 + D * y1 * y1
                        a2 = x2 * x2 + D * y2 * y2
                        a3 = x3 * x3 + D * y3 * y3
                        a4 = x4 * x4 + D * y4 * y4
                        
                        # Expansion of det(M)
                        # M = [[x1, y1, a1, 1], ...]
                        # Using Sarrus-like or cofactor expansion.
                        # Let's use the formula for 4x4 det.
                        # det = x1*(y2*(a3-a4) - y3*(a2-a4) + y4*(a2-a3)) -
                        #       y1*(x2*(a3-a4) - x3*(a2-a4) + x4*(a2-a3)) +
                        #       a1*(x2*y3 - x3*y2 - x2*y4 + x4*y2 + x3*y4 - x4*y3) -
                        #       (x2*y3*a4 - x2*y4*a3 - x3*y2*a4 + x3*y4*a2 + x4*y2*a3 - x4*y3*a2)
                        
                        det = (x1 * (y2 * (a3 - a4) - y3 * (a2 - a4) + y4 * (a2 - a3)) -
                               y1 * (x2 * (a3 - a4) - x3 * (a2 - a4) + x4 * (a2 - a3)) +
                               a1 * (x2 * y3 - x3 * y2 - x2 * y4 + x4 * y2 + x3 * y4 - x4 * y3) -
                               (x2 * y3 * a4 - x2 * y4 * a3 - x3 * y2 * a4 + x3 * y4 * a2 + x4 * y2 * a3 - x4 * y3 * a2))
                        if det == 0:
                            return False
        return True

    def try_search(D, limit, target_size):
        """
        Search for target_size points in [-limit, limit]^2 for given D.
        """
        candidates = []
        for x in range(-limit, limit + 1):
            for y in range(-limit, limit + 1):
                candidates.append((Fraction(x), Fraction(y)))
        
        best = None
        current = []
        
        def backtrack(idx):
            nonlocal best
            if time.time() > deadline:
                return
            if len(current) >= target_size:
                if best is None or len(current) > len(best):
                    best = list(current)
                return
            
            # Pruning: if not enough candidates left to reach target
            if len(candidates) - idx < target_size - len(current):
                return

            for i in range(idx, len(candidates)):
                cand = candidates[i]
                new_current = current + [cand]
                if check_valid(new_current, D):
                    current.append(cand)
                    backtrack(i + 1)
                    current.pop()
                    if best is not None and len(best) >= target_size:
                        return

        backtrack(0)
        return best

    # Strategy:
    # 1. Try to find 6 points with D=1 in a small box.
    # 2. If found, return 6.
    # 3. If not, try 5 points (which we know exist) or return the best found.
    
    # Known 5-point set (fallback)
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D_5 = 1
    fracs_5 = [(Fraction(x), Fraction(y)) for x, y in points_5]
    
    # Try to find 6 points
    found_6 = None
    # Limit 10 is 21x21 = 441 candidates.
    # 6 points from 441 is C(441, 6) which is huge, but pruning is strong.
    # We try a few D values and limits.
    
    for D in [1, 2, 3]:
        if not is_squarefree(D):
            continue
        # Try small limits first
        for limit in [5, 8, 10]:
            result = try_search(D, limit, 6)
            if result:
                found_6 = (D, result)
                break
        if found_6:
            break
            
    if found_6:
        D, fracs = found_6
        points_out = [[[int(x.numerator), int(x.denominator)], [int(y.numerator), int(y.denominator)]] for x, y in fracs]
        return {'D': D, 'points': points_out}
    else:
        # Fallback to 5 points
        points_out = [[[int(x.numerator), int(x.denominator)], [int(y.numerator), int(y.denominator)]] for x, y in fracs_5]
        return {'D': D_5, 'points': points_out}

# EVOLVE-BLOCK-END
