"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import random
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n < 2:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _g2(x1, y1, x2, y2, D):
    """Squared distance between (x1,y1) and (x2,y2) in the lattice Z[sqrt(D)]/Q^2."""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _g(x1, y1, x2, y2, D):
    """Distance (not squared) as a Fraction if rational, else None."""
    g2_val = _g2(x1, y1, x2, y2, D)
    if g2_val < 0:
        return None
    # Check if g2_val is a perfect square rational
    # g2_val is a Fraction or int. We need sqrt(g2_val) to be rational.
    # Since inputs are integers (or Fractions), g2_val is a Fraction.
    # Let g2_val = a/b. We need a/b = (c/d)^2 => a*d^2 = b*c^2.
    # Simpler: convert to Fraction, take sqrt of numerator and denominator.
    frac = Fraction(g2_val)
    num = frac.numerator
    den = frac.denominator
    if num < 0 or den < 0:
        return None
    # Check if num and den are perfect squares
    r_num = math.isqrt(num)
    r_den = math.isqrt(den)
    if r_num * r_num != num or r_den * r_den != den:
        return None
    return Fraction(r_num, r_den)

def _is_rational_dist(x1, y1, x2, y2, D):
    return _g(x1, y1, x2, y2, D) is not None

def _no_three_collinear(points, D):
    """Check no three points are collinear.
    Points are (x, y*sqrt(D)). Collinearity condition:
    For points P1, P2, P3, the area of triangle is 0 iff:
    (x2-x1)(y3*sqrt(D) - y1*sqrt(D)) - (x3-x1)(y2*sqrt(D) - y1*sqrt(D)) = 0
    sqrt(D) * [ (x2-x1)(y3-y1) - (x3-x1)(y2-y1) ] = 0
    Since D>0, sqrt(D) != 0, so we need:
    (x2-x1)(y3-y1) - (x3-x1)(y2-y1) = 0
    This is the standard 2D collinearity check on (x, y) coordinates.
    """
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
                cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic.
    Four points (x_i, y_i*sqrt(D)) are concyclic iff there exists a circle passing through all four.
    A circle in this plane: (x - h)^2 + (y*sqrt(D) - k)^2 = R^2
    => x^2 - 2hx + h^2 + D*y^2 - 2Dky + D*k^2 = R^2
    => x^2 + D*y^2 - 2hx - 2Dky + (h^2 + D*k^2 - R^2) = 0
    Let A=1, B=D, C=-2h, E=-2Dk, F=h^2+D*k^2-R^2.
    Equation: x^2 + D*y^2 + C*x + E*y + F = 0.
    For 4 points to be concyclic, the 4x5 system for (C, E, F) with the quadratic terms fixed must be consistent.
    Actually, we can use the determinant condition.
    For points P1..P4, they are concyclic iff the determinant of the matrix:
    | x1^2+D*y1^2, x1, y1, 1 |
    | x2^2+D*y2^2, x2, y2, 1 |
    | x3^2+D*y3^2, x3, y3, 1 |
    | x4^2+D*y4^2, x4, y4, 1 |
    is zero? Not exactly, because the coefficient of x^2 and y^2 are fixed (1 and D).
    The general circle equation is x^2 + D*y^2 + ax + by + c = 0.
    For 4 points, we have 4 linear equations in a, b, c:
    a*x_i + b*y_i + c = -(x_i^2 + D*y_i^2)
    This is a 4x3 system. It has a solution iff the 4x4 matrix:
    | x1, y1, 1, -(x1^2+D*y1^2) |
    | x2, y2, 1, -(x2^2+D*y2^2) |
    | x3, y3, 1, -(x3^2+D*y3^2) |
    | x4, y4, 1, -(x4^2+D*y4^2) |
    has rank < 4, i.e., determinant is 0.
    """
    n = len(points)
    for combo in combinations(range(n), 4):
        i, j, k, l = combo
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        x4, y4 = points[l]
        
        # Matrix rows: [x, y, 1, -(x^2 + D*y^2)]
        # We compute the determinant of the 4x4 matrix.
        # Since numbers can be large, we use exact integer arithmetic.
        # M = [[x1, y1, 1, -v1],
        #      [x2, y2, 1, -v2],
        #      [x3, y3, 1, -v3],
        #      [x4, y4, 1, -v4]]
        # where vi = xi^2 + D*yi^2
        
        v1 = x1*x1 + D*y1*y1
        v2 = x2*x2 + D*y2*y2
        v3 = x3*x3 + D*y3*y3
        v4 = x4*x4 + D*y4*y4
        
        # Determinant calculation
        # | x1  y1  1  -v1 |
        # | x2  y2  1  -v2 |
        # | x3  y3  1  -v3 |
        # | x4  y4  1  -v4 |
        
        # We can expand along the third column (which has 1s).
        # Or use the formula for determinant of 4x4.
        # Let's compute:
        # det = x1*(y2*(1*(-v4) - (-v3)*1) - 1*(y2*(-v4) - (-v3)*y4) + 1*(y2*(-v3) - y3*(-v4)))
        #       - y1*(x2*(1*(-v4) - (-v3)*1) - 1*(x2*(-v4) - (-v3)*x4) + 1*(x2*(-v3) - x3*(-v4)))
        #       + 1*(x2*(y2? No, this is getting messy. Let's use a helper.
        
        def det4(m):
            # m is 4x4 list of lists
            return (
                m[0][0] * (m[1][1]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]))
                - m[0][1] * (m[1][0]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]))
                + m[0][2] * (m[1][0]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1] - m[2][1]*m[3][0]))
                - m[0][3] * (m[1][0]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1] - m[2][1]*m[3][0]))
            )
        
        m = [
            [x1, y1, 1, -v1],
            [x2, y2, 1, -v2],
            [x3, y3, 1, -v3],
            [x4, y4, 1, -v4]
        ]
        if det4(m) == 0:
            return False
    return True

def _validate(points, D):
    """Check all conditions: distinct, rational distances, no 3 collinear, no 4 concyclic."""
    n = len(points)
    if n < 3:
        return False
    # Check distinct
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
    # Check rational distances
    for i in range(n):
        for j in range(i+1, n):
            if not _is_rational_dist(points[i][0], points[i][1], points[j][0], points[j][1], D):
                return False
    # Check no 3 collinear
    if not _no_three_collinear(points, D):
        return False
    # Check no 4 concyclic
    if not _no_four_concyclic(points, D):
        return False
    return True

def _generate_candidate(D, max_coord=1000, n_points=5):
    """Generate a random candidate set of points."""
    points = []
    for _ in range(n_points):
        x = random.randint(-max_coord, max_coord)
        y = random.randint(-max_coord, max_coord)
        points.append((x, y))
    return points

def _scale_to_int(points, D):
    """Scale points to have integer coordinates if possible."""
    # Points are already integers in our generation, so this is a no-op.
    # But if we used rationals, we'd scale.
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s to be safe
    
    # Start with the known 5-point solution
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_D = 1
    best_n = len(best_points)
    
    # Try to find a 6-point solution
    # Strategy: Random search over integer points with D=1 (Euclidean plane)
    # This is a hard problem. 6-point integral sets in general position are rare.
    # Let's try a local search or random sampling.
    
    # First, verify the best solution
    if not _validate(best_points, best_D):
        # This shouldn't happen, but just in case
        best_points = [(0, 0), (3, 4), (0, 5)]
        best_D = 1
        best_n = 3
    
    # Try to find better solutions
    # We'll try D=1 first (Euclidean), then other D.
    # For D=1, we need integer points with integer distances.
    
    # Random search for 6 points
    found_6 = False
    while not found_6 and time.time() < deadline:
        # Try to build a set of 6 points incrementally
        # Start with the known 5 points and try to add a 6th
        base_points = best_points.copy()
        for _ in range(1000):
            if time.time() >= deadline:
                break
            # Try adding a random point
            x = random.randint(-200, 200)
            y = random.randint(-200, 200)
            new_point = (x, y)
            candidate = base_points + [new_point]
            if _validate(candidate, 1):
                found_6 = True
                best_points = candidate
                best_D = 1
                best_n = 6
                break
    
    # If we found 6, try to find 7
    if best_n == 6:
        found_7 = False
        while not found_7 and time.time() < deadline:
            base_points = best_points.copy()
            for _ in range(1000):
                if time.time() >= deadline:
                    break
                x = random.randint(-200, 200)
                y = random.randint(-200, 200)
                new_point = (x, y)
                candidate = base_points + [new_point]
                if _validate(candidate, 1):
                    found_7 = True
                    best_points = candidate
                    best_D = 1
                    best_n = 7
                    break
    
    # If still 5, try other D values with random search
    if best_n == 5 and time.time() < deadline:
        for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 93, 94, 95, 97]:
            if time.time() >= deadline:
                break
            if not _is_squarefree(D):
                continue
            # Try to find 6 points with this D
            for _ in range(100):
                if time.time() >= deadline:
                    break
                points = []
                for _ in range(6):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    points.append((x, y))
                if _validate(points, D):
                    best_points = points
                    best_D = D
                    best_n = 6
                    break
    
    # Format the result
    result_points = []
    for x, y in best_points:
        result_points.append([[x, 1], [y, 1]])
    
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
