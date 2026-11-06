import time
import math
from fractions import Fraction
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit

    def is_rational_sqrt(num, den):
        """Check if num/den is a perfect square of a rational."""
        if num == 0:
            return True
        if den < 0:
            num = -num
            den = -den
        g = math.gcd(abs(num), abs(den))
        n = abs(num) // g
        d = abs(den) // g
        if n == 0:
            return True
        root_n = math.isqrt(n)
        root_d = math.isqrt(d)
        if root_n * root_n != n or root_d * root_d != d:
            return False
        return True

    def dist_sq_is_rational_square(p1, p2, D):
        """Check if squared distance between p1 and p2 is a rational square."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + D * dy*dy
        if d2 < 0:
            return False
        return is_rational_sqrt(d2.numerator, d2.denominator)

    def check_no_3_collinear(points):
        """Check no 3 points are collinear in (x, y) coordinates."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    # Cross product in (x, y) space
                    val = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    if val == 0:
                        return False
        return True

    def check_no_4_concyclic(points, D):
        """Check no 4 points are concyclic.
        Four points (x_i, y_i*sqrt(D)) are concyclic iff there exist A,B,C,R such that
        (x_i)^2 + D*(y_i)^2 + A*x_i + B*y_i + C = R^2 for all four points.
        This is equivalent to the 4x4 determinant being zero:
        |x_i^2 + D*y_i^2, x_i, y_i, 1| for i=1..4
        """
        n = len(points)
        for combo in combinations(range(n), 4):
            p1, p2, p3, p4 = (points[i] for i in combo)
            # Build 4x4 matrix
            # Row i: [x_i^2 + D*y_i^2, x_i, y_i, 1]
            r1 = [p1[0]**2 + D*p1[1]**2, p1[0], p1[1], Fraction(1)]
            r2 = [p2[0]**2 + D*p2[1]**2, p2[0], p2[1], Fraction(1)]
            r3 = [p3[0]**2 + D*p3[1]**2, p3[0], p3[1], Fraction(1)]
            r4 = [p4[0]**2 + D*p4[1]**2, p4[0], p4[1], Fraction(1)]
            # Compute determinant of 4x4 matrix
            def det4(m):
                a,b,c,d = m[0]
                e,f,g,h = m[1]
                i,j,k,l = m[2]
                m_,n_,o,p = m[3]
                return (
                    a*(f*(k*p-l*o) - g*(j*p-l*n_) + h*(j*o-k*n_)) -
                    b*(e*(k*p-l*o) - g*(i*p-l*m_) + h*(i*o-k*m_)) +
                    c*(e*(j*p-l*n_) - f*(i*p-l*m_) + h*(i*n_-j*m_)) -
                    d*(e*(j*o-k*n_) - f*(i*o-k*m_) + g*(i*n_-j*m_))
                )
            if det4([r1, r2, r3, r4]) == 0:
                return False
        return True

    def is_valid_set(points, D):
        """Full validity check."""
        n = len(points)
        # Check distinctness
        for i in range(n):
            for j in range(i+1, n):
                if points[i] == points[j]:
                    return False
        # Check all pairwise distances are rational
        for i in range(n):
            for j in range(i+1, n):
                if not dist_sq_is_rational_square(points[i], points[j], D):
                    return False
        # Check no 3 collinear
        if not check_no_3_collinear(points):
            return False
        # Check no 4 concyclic
        if not check_no_4_concyclic(points, D):
            return False
        return True

    def to_output_format(points, D):
        return {
            'D': D,
            'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]
        }

    # Known 5-point set for D=1 (verified)
    known_5pts = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]

    best_result = to_output_format(known_5pts, 1)
    best_size = 5

    # Verify the known set
    if is_valid_set(known_5pts, 1):
        best_size = 5
    else:
        # Fallback to a simpler valid set
        simple_4pts = [
            (Fraction(0), Fraction(0)),
            (Fraction(3), Fraction(0)),
            (Fraction(0), Fraction(4)),
            (Fraction(5), Fraction(0)),
        ]
        if is_valid_set(simple_4pts, 1):
            best_result = to_output_format(simple_4pts, 1)
            best_size = 4

    def generate_circle_points(center, D, radius, num_params=200):
        """Generate points on circle of given radius around center using rational parameterization.
        For D=1: circle (x-a)^2 + (y-b)^2 = r^2
        Param: x = a + r*(1-t^2)/(1+t^2), y = b + 2*r*t/(1+t^2) for rational t
        For general D: (x-a)^2 + D*(y-b)^2 = r^2
        This is an ellipse. Param: x = a + r*(1-t^2)/(1+t^2), y = b + 2*r*t/(D*(1+t^2))
        Actually for D != 1, we need: (x-a)^2 + D*(y-b)^2 = r^2
        Let u = x-a, v = y-b. Then u^2 + D*v^2 = r^2.
        u = r*(1-t^2)/(1+t^2), D*v^2 = r^2 - u^2 = r^2 * 4t^2/(1+t^2)^2
        So v = ± 2*r*t/(D*(1+t^2))... wait that's not right.
        D*v^2 = r^2*4t^2/(1+t^2)^2
        v^2 = r^2*4t^2/(D*(1+t^2)^2)
        v = ± 2*r*t/(sqrt(D)*(1+t^2))
        For v to be rational, we need sqrt(D) to divide... this only works for D=1 or D a perfect square.
        For general D, this parametric approach doesn't give rational y directly.
        
        Alternative for general D: We need points (x,y) with rational x,y such that
        (x-a)^2 + D*(y-b)^2 = r^2 for rational r.
        This is: (x-a)^2 + D*(y-b)^2 = r^2
        For D=1, use the standard circle parameterization.
        For D>1, we can use: if we set x-a = r*(m^2-n^2)/(m^2+n^2) and y-b = r*2mn/(D*(m^2+n^2))... no.
        
        Let me think differently. For D=1, the standard rational circle parametrization works.
        For general D, we need (x-a)^2 + D*(y-b)^2 = r^2 where r is rational.
        This means (x-a)^2 + D*(y-b)^2 must be a perfect square of a rational.
        Let u = x-a, v = y-b. We need u^2 + D*v^2 = s^2 for some rational s.
        This is a conic. If we have one rational point, we can parameterize all others.
        The point (u,v) = (s, 0) is on it. Line through (s,0) with rational slope m:
        v = m*(u-s)
        u^2 + D*m^2*(u-s)^2 = s^2
        u^2 + D*m^2*(u^2 - 2su + s^2) = s^2
        (1+D*m^2)*u^2 - 2*D*m^2*s*u + D*m^2*s^2 - s^2 = 0
        One root is u=s (the known point). The other:
        u = s*(D*m^2-1)/(D*m^2+1)  ... let me verify.
        Sum of roots = 2*D*m^2*s/(1+D*m^2)
        One root is s, so other root = 2*D*m^2*s/(1+D*m^2) - s = s*(2D*m^2 - 1 - D*m^2)/(1+D*m^2) = s*(D*m^2-1)/(D*m^2+1)
        v = m*(u-s) = m*s*(D*m^2-1)/(D*m^2+1) - m*s = m*s*((D*m^2-1) - (D*m^2+1))/(D*m^2+1) = m*s*(-2)/(D*m^2+1) = -2*m*s/(D*m^2+1)
        
        So: u = s*(D*m^2-1)/(D*m^2+1), v = -2*m*s/(D*m^2+1)
        Therefore: x = a + s*(D*m^2-1)/(D*m^2+1), y = b - 2*m*s/(D*m^2+1)
        For rational s and m, x and y are rational. 
        """
        candidates = []
        if D == 1:
            # Standard circle parameterization
            for m_num in range(-30, 31):
                for m_den in range(1, 8):
                    m = Fraction(m_num, m_den)
                    denom = 1 + m*m
                    x = center[0] + radius * (1 - m*m) / denom
                    y = center[1] + 2 * radius * m / denom
                    candidates.append((x, y))
        else:
            # General D parameterization
            for m_num in range(-30, 31):
                for m_den in range(1, 8):
                    m = Fraction(m_num, m_den)
                    denom = D * m * m + 1
                    x = center[0] + radius * (D * m * m - 1) / denom
                    y = center[1] - 2 * m * radius / denom
                    candidates.append((x, y))
        
        return candidates

    def greedy_expand(points, D, max_size=12):
        """Greedily add points to maximize set size."""
        current = list(points)
        if not current:
            return current
        
        while len(current) < max_size:
            if time.time() > deadline:
                break
            
            # Generate candidates: points on circles around existing points
            candidates = []
            for anchor in current:
                # Try several radii (rational distances)
                for r_num in range(1, 25):
                    for r_den in range(1, 6):
                        r = Fraction(r_num, r_den)
                        circ_pts = generate_circle_points(anchor, D, r, 50)
                        for cp in circ_pts:
                            # Check rational distance to all current points
                            valid = True
                            for a in current:
                                if not dist_sq_is_rational_square(cp, a, D):
                                    valid = False
                                    break
                            if valid:
                                candidates.append(cp)
            
            # Remove duplicates and existing points
            seen = set(current)
            unique_cands = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique_cands.append(c)
            
            # Sort by distance to centroid (prefer points that spread out)
            if unique_cands:
                cx = sum(p[0] for p in current) / len(current)
                cy = sum(p[1] for p in current) / len(current)
                unique_cands.sort(key=lambda p: abs((p[0]-cx).numerator) + abs((p[1]-cy).numerator))
            
            added = False
            for cand in unique_cands:
                if time.time() > deadline:
                    break
                new_pts = current + [cand]
                if is_valid_set(new_pts, D):
                    current = new_pts
                    added = True
                    break
            
            if not added:
                # Try random candidates as fallback
                found = False
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    x = Fraction(random.randint(-30, 30), random.randint(1, 10))
                    y = Fraction(random.randint(-30, 30), random.randint(1, 10))
                    cand = (x, y)
                    if cand in set(current):
                        continue
                    valid = True
                    for a in current:
                        if not dist_sq_is_rational_square(cand, a, D):
                            valid = False
                            break
                    if valid:
                        new_pts = current + [cand]
                        if is_valid_set(new_pts, D):
                            current = new_pts
                            found = True
                            break
                if not found:
                    break
        
        return current

    # Strategy 1: Expand the known 5-point set for D=1
    if time.time() < deadline - 10:
        result = greedy_expand(known_5pts, 1, max_size=12)
        if len(result) > best_size:
            best_size = len(result)
            best_result = to_output_format(result, 1)

    # Strategy 2: Try different starting sets for D=1
    if time.time() < deadline - 10:
        # Try a 4-point seed and expand
        seeds_d1 = [
            [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4)), (Fraction(3), Fraction(4))],
            [(Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)), (Fraction(0), Fraction(12)), (Fraction(5), Fraction(12))],
            [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)), (Fraction(1), Fraction(1))],
        ]
        for seed in seeds_d1:
            if time.time() > deadline - 5:
                break
            if not is_valid_set(seed, 1):
                continue
            result = greedy_expand(seed, 1, max_size=12)
            if len(result) > best_size:
                best_size = len(result)
                best_result = to_output_format(result, 1)

    # Strategy 3: Try D=2, 3, etc.
    if time.time() < deadline - 10:
        for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
            if time.time() > deadline - 5:
                break
            
            # For D>1, start with simple configurations
            # (0,0) and (a,0) have distance |a| (rational)
            # (0,0) and (0,b) have distance b*sqrt(D) — need D*b^2 to be a square, so b must make D*b^2 a square
            # For D=2: (0,0) and (0,1) have distance sqrt(2) — not rational
            # So for D>1, we can only easily get horizontal/vertical distances if they work out.
            # (0,0) and (a,0): distance = |a|, always rational.
            # (0,0) and (a,b): distance^2 = a^2 + D*b^2, need this to be a rational square.
            
            # Simple seed: two points on x-axis
            seed = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))]
            if is_valid_set(seed, D):
                result = greedy_expand(seed, D, max_size=12)
                if len(result) > best_size:
                    best_size = len(result)
                    best_result = to_output_format(result, D)
            
            # Another seed: (0,0), (D,0), (0,1) — check if (0,0) to (0,1) works: dist^2 = D*1 = D, need D to be a square. Only works for perfect square D.
            # For D=2: (0,0), (2,0) dist=2. (0,0) to (0,1): dist^2=2, not a square. So skip.
            # Try (0,0), (a,0), (b,c) where a^2 + D*c^2 is a square and b^2 - a*b + ... 
            # Let's just try the simple 2-point seed.

    # Strategy 4: More aggressive expansion on best result
    if time.time() < deadline - 3 and best_size >= 4:
        D_best = best_result['D']
        pts_best = [(Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1])) 
                    for p in best_result['points']]
        improved = greedy_expand(pts_best, D_best, max_size=12)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = to_output_format(improved, D_best)

    return best_result
