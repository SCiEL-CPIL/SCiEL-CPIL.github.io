"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def _exact_det_4x4(p1, p2, p3, p4):
        """Exact determinant for 4 concyclic check using rational arithmetic."""
        rows = []
        for p in [p1, p2, p3, p4]:
            x, y = p
            # Convert to Fractions for exact arithmetic
            xf = Fraction(x.numerator, x.denominator) if isinstance(x, Fraction) else Fraction(x, 1)
            yf = Fraction(y.numerator, y.denominator) if isinstance(y, Fraction) else Fraction(y, 1)
            rows.append([xf, yf, xf*xf + yf*yf, Fraction(1, 1)])
        
        # Compute 4x4 determinant
        def det4(m):
            result = Fraction(0, 1)
            signs = [1, -1, 1, -1]
            for j in range(4):
                minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
                result += signs[j] * m[0][j] * det3(minor)
            return result
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        
        return det4(rows)

    def all_constraints_satisfied(points):
        """Check all pairwise distances are integers, no 3 collinear, no 4 concyclic."""
        n = len(points)
        # Check all pairwise distances are integers
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                d2 = dx*dx + dy*dy
                if d2 < 0:
                    return False
                # d2 must be a perfect square of an integer
                # Since x, y are Fractions, d2 is a Fraction
                if d2.denominator != 1:
                    return False
                if not is_perfect_square(d2.numerator):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        # Check no four concyclic
        for combo in combinations(range(n), 4):
            if _exact_det_4x4(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]) == 0:
                return False
        return True

    def try_add_point(current_points, candidate):
        """Try adding a candidate point to current_points."""
        new_points = current_points + [candidate]
        if all_constraints_satisfied(new_points):
            return new_points
        return None

    # Start with known 5-point set from seed
    # Convert to Fractions for uniform handling
    initial_pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = [(Fraction(x, 1), Fraction(y, 1)) for x, y in initial_pts]
    
    # Verify initial set
    if not all_constraints_satisfied(best_points):
        # Fallback to a simpler valid set if the known one fails (should not happen)
        best_points = [(Fraction(0, 1), Fraction(0, 1)), (Fraction(3, 1), Fraction(4, 1)), (Fraction(4, 1), Fraction(0, 1))]

    max_coord_bound = 500
    max_dist = 200
    
    # Strategy: Search for new points by intersecting circles
    # We pick two existing points P1, P2 and iterate over integer distances d1, d2
    # to find intersection points that are rational.
    
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        n = len(best_points)
        
        # Pick two reference points to define the circle intersection
        # We try all pairs to increase chances
        for i in range(n):
            for j in range(i+1, n):
                if elapsed() > DEADLINE:
                    break
                P1 = best_points[i]
                P2 = best_points[j]
                x1, y1 = P1
                x2, y2 = P2
                
                # Distance between P1 and P2
                d12_sq = (x2-x1)**2 + (y2-y1)**2
                d12 = math.isqrt(d12_sq.numerator) if d12_sq.denominator == 1 else None
                if d12 is None or d12*d12 != d12_sq.numerator:
                    continue # P1, P2 distance must be integer for the set to be valid, which it is
                
                # Iterate possible integer distances from P1 and P2
                # The new point P must satisfy |P-P1|=d1, |P-P2|=d2
                # Triangle inequality: |d1-d2| <= d12 <= d1+d2
                # We search for d1, d2 in a range
                # To keep it efficient, we limit the range
                min_d1 = 1
                max_d1 = max_dist
                min_d2 = 1
                max_d2 = max_dist
                
                # We can narrow down d1, d2 based on d12
                # |d1 - d2| <= d12 <= d1 + d2
                # So d2 is in [d1 - d12, d1 + d12]
                
                for d1 in range(min_d1, max_d1 + 1):
                    if elapsed() > DEADLINE:
                        break
                    # d2 must satisfy triangle inequality
                    min_d2_for_d1 = max(1, d1 - d12)
                    max_d2_for_d1 = d1 + d12
                    if max_d2_for_d1 > max_dist:
                        max_d2_for_d1 = max_dist
                        
                    for d2 in range(min_d2_for_d1, max_d2_for_d1 + 1):
                        # Solve for intersection of circles
                        # (x-x1)^2 + (y-y1)^2 = d1^2
                        # (x-x2)^2 + (y-y2)^2 = d2^2
                        # Subtracting the two equations gives a linear equation for x, y
                        # 2(x2-x1)x + 2(y2-y1)y = d1^2 - d2^2 + x2^2 - x1^2 + y2^2 - y1^2
                        
                        A = 2 * (x2 - x1)
                        B = 2 * (y2 - y1)
                        C = d1**2 - d2**2 + (x2**2 - x1**2) + (y2**2 - y1**2)
                        
                        if A == 0 and B == 0:
                            continue # P1 == P2, not possible for distinct points
                        
                        # The line is Ax + By = C
                        # We need to find points on this line at distance d1 from P1.
                        # This is a standard circle-line intersection.
                        
                        # Parametric form of the line:
                        # If A != 0: x = t, y = (C - A*t)/B (if B!=0)
                        # Better: Use vector projection.
                        # Let v = P2 - P1 = (dx, dy)
                        # Let P = P1 + s*v + t*perp(v)
                        # This might be complex with Fractions.
                        
                        # Let's use the explicit formula for circle-line intersection.
                        # Distance from P1 to the line Ax+By-C=0:
                        # dist = |A*x1 + B*y1 - C| / sqrt(A^2 + B^2)
                        # For intersection to exist, dist <= d1
                        # Also, the intersection points are rational iff sqrt(A^2+B^2) is rational?
                        # No, the coordinates involve sqrt(d1^2 - dist^2).
                        # For coordinates to be rational, (d1^2 - dist^2) * (A^2+B^2) must be a perfect square?
                        
                        # Let's just compute the discriminant.
                        # Substitute y = (C - A*x)/B into (x-x1)^2 + (y-y1)^2 = d1^2
                        # This leads to a quadratic in x.
                        
                        if B == 0:
                            # Vertical line x = C/A
                            if A == 0: continue
                            x_val = Fraction(C, A)
                            # Check if x_val is rational (it is)
                            # Find y: (x_val-x1)^2 + (y-y1)^2 = d1^2
                            # (y-y1)^2 = d1^2 - (x_val-x1)^2
                            rhs = d1**2 - (x_val-x1)**2
                            if rhs < 0:
                                continue
                            # y1 +/- sqrt(rhs)
                            # For y to be rational, rhs must be a perfect square of a rational
                            # rhs is a Fraction.
                            if rhs.denominator != 1:
                                # It could still be a square of a fraction, e.g., (1/2)^2 = 1/4
                                # Let's check if sqrt(numerator)/sqrt(denominator) is rational
                                # i.e., numerator and denominator are perfect squares
                                num = rhs.numerator
                                den = rhs.denominator
                                if not is_perfect_square(num) or not is_perfect_square(den):
                                    continue
                                sqrt_rhs = Fraction(math.isqrt(num), math.isqrt(den))
                            else:
                                if not is_perfect_square(rhs.numerator):
                                    continue
                                sqrt_rhs = Fraction(math.isqrt(rhs.numerator), 1)
                            
                            y_val1 = y1 + sqrt_rhs
                            y_val2 = y1 - sqrt_rhs
                            
                            for y_val in [y_val1, y_val2]:
                                cand = (x_val, y_val)
                                # Check if candidate is already in set
                                if any(cand[0] == p[0] and cand[1] == p[1] for p in best_points):
                                    continue
                                # Check bounds
                                if abs(x_val.numerator) > max_coord_bound * x_val.denominator:
                                    continue
                                if abs(y_val.numerator) > max_coord_bound * y_val.denominator:
                                    continue
                                
                                result = try_add_point(best_points, cand)
                                if result is not None:
                                    best_points = result
                                    improved = True
                                    if len(best_points) >= 12:
                                        break
                                    if elapsed() > DEADLINE:
                                        break
                        else:
                            # General case
                            # We can swap x and y roles if B is 0, but here B!=0.
                            # Let's express x in terms of y or use a unified method.
                            # Actually, the logic is symmetric.
                            # Let's stick to the substitution method.
                            # To avoid large numbers, we can normalize A, B, C?
                            # No, Fractions handle it.
                            
                            # Equation: (x-x1)^2 + ((C-Ax)/B - y1)^2 = d1^2
                            # Let Y = (C - B*y1)/B = C/B - y1
                            # (x-x1)^2 + ((C-Ax)/B - y1)^2 = d1^2
                            # (x-x1)^2 + ( (C - A*x - B*y1) / B )^2 = d1^2
                            # B^2 (x-x1)^2 + (C - B*y1 - A*x)^2 = B^2 d1^2
                            # Let K = C - B*y1
                            # B^2 (x^2 - 2x x1 + x1^2) + (K - A x)^2 = B^2 d1^2
                            # B^2 x^2 - 2 B^2 x1 x + B^2 x1^2 + K^2 - 2 K A x + A^2 x^2 = B^2 d1^2
                            # (A^2 + B^2) x^2 - (2 B^2 x1 + 2 K A) x + (B^2 x1^2 + K^2 - B^2 d1^2) = 0
                            
                            a_q = A**2 + B**2
                            b_q = -(2 * B**2 * x1 + 2 * K * A)
                            c_q = B**2 * x1**2 + K**2 - B**2 * d1**2
                            
                            disc = b_q**2 - 4 * a_q * c_q
                            if disc < 0:
                                continue
                            
                            # x = (-b_q +/- sqrt(disc)) / (2 a_q)
                            # For x to be rational, disc must be a perfect square of a rational
                            if disc.denominator != 1:
                                num = disc.numerator
                                den = disc.denominator
                                if not is_perfect_square(num) or not is_perfect_square(den):
                                    continue
                                sqrt_disc = Fraction(math.isqrt(num), math.isqrt(den))
                            else:
                                if not is_perfect_square(disc.numerator):
                                    continue
                                sqrt_disc = Fraction(math.isqrt(disc.numerator), 1)
                            
                            x_val1 = (-b_q + sqrt_disc) / (2 * a_q)
                            x_val2 = (-b_q - sqrt_disc) / (2 * a_q)
                            
                            for x_val in [x_val1, x_val2]:
                                # Find y from line equation: A x + B y = C => y = (C - A x) / B
                                y_val = (C - A * x_val) / B
                                cand = (x_val, y_val)
                                if any(cand[0] == p[0] and cand[1] == p[1] for p in best_points):
                                    continue
                                if abs(x_val.numerator) > max_coord_bound * x_val.denominator:
                                    continue
                                if abs(y_val.numerator) > max_coord_bound * y_val.denominator:
                                    continue
                                
                                result = try_add_point(best_points, cand)
                                if result is not None:
                                    best_points = result
                                    improved = True
                                    if len(best_points) >= 12:
                                        break
                                    if elapsed() > DEADLINE:
                                        break
                    if elapsed() > DEADLINE:
                        break
                if elapsed() > DEADLINE:
                    break
            if elapsed() > DEADLINE:
                break

    # Convert to required format
    D = 1
    points_out = []
    for x, y in best_points:
        # x, y are Fractions
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
