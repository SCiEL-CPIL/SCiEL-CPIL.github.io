import time
import math
from fractions import Fraction
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 135.0  # 15s safety margin before 150s limit

    # Known good 5-point set for D=1
    known_5pts = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]

    best_size = 5
    best_result = {
        'D': 1,
        'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in known_5pts]
    }

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
        root_n = int(math.isqrt(n))
        root_d = int(math.isqrt(d))
        if root_n * root_n != n or root_d * root_d != d:
            return False
        return True

    def dist_sq_rational(p1, p2, D):
        """Returns True if distance squared between p1 and p2 is a rational square."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + D * dy*dy
        if d2 < 0:
            return False
        return is_rational_sqrt(d2.numerator, d2.denominator)

    def check_no_3_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    val = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    if val == 0:
                        return False
        return True

    def det4(m):
        a,b,c,d = m[0]
        e,f,g,h = m[1]
        i,j,k,l = m[2]
        m_,n,o,p = m[3]
        return (
            a*(f*(k*p-l*o) - g*(j*p-l*n) + h*(j*o-k*n)) -
            b*(e*(k*p-l*o) - g*(i*p-l*m_) + h*(i*o-k*m_)) +
            c*(e*(j*p-l*n) - f*(i*p-l*m_) + h*(i*n-j*m_)) -
            d*(e*(j*o-k*n) - f*(i*o-k*m_) + g*(i*n-j*m_))
        )

    def check_no_4_concyclic_with_D(points, D):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        r1 = [p1[0]**2 + D*p1[1]**2, p1[0], p1[1], Fraction(1)]
                        r2 = [p2[0]**2 + D*p2[1]**2, p2[0], p2[1], Fraction(1)]
                        r3 = [p3[0]**2 + D*p3[1]**2, p3[0], p3[1], Fraction(1)]
                        r4 = [p4[0]**2 + D*p4[1]**2, p4[0], p4[1], Fraction(1)]
                        M = [r1, r2, r3, r4]
                        if det4(M) == 0:
                            return False
        return True

    def is_valid_set(points, D):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if points[i] == points[j]:
                    return False
        for i in range(n):
            for j in range(i+1, n):
                if not dist_sq_rational(points[i], points[j], D):
                    return False
        if not check_no_3_collinear(points):
            return False
        if not check_no_4_concyclic_with_D(points, D):
            return False
        return True

    def to_output_format(points, D):
        return {
            'D': D,
            'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]
        }

    # Verify the known 5-point set
    if is_valid_set(known_5pts, 1):
        best_size = 5
        best_result = to_output_format(known_5pts, 1)

    # Strategy: For D=1, use Pythagorean triples to find points with rational distances
    # Key insight: if we have points (x1, y1) and (x2, y2), the distance squared is
    # (x1-x2)^2 + (y1-y2)^2. For this to be a rational square, we need the difference
    # vector to be part of a Pythagorean triple.
    
    # Better strategy: Use the fact that for D=1, we can parameterize circles.
    # Given two points P1=(x1,y1) and P2=(x2,y2) with rational distance r,
    # any point P3 on the circle centered at P1 with radius r1 (rational) and
    # on the circle centered at P2 with radius r2 (rational) will work if
    # the intersection points have rational coordinates.
    
    # The intersection of two circles with rational centers and radii gives
    # points that lie on a line (radical axis). If the line has rational slope
    # and intercept, and the circle equations have rational coefficients,
    # the intersection points have coordinates that are rational functions
    # of the parameters.
    
    # Actually, let's use a more direct approach:
    # For D=1, we can use the parametrization of rational points on the unit circle.
    # A point (x, y) has rational distance to (0,0) iff x^2 + y^2 is a rational square.
    # This is equivalent to (x, y) being on a circle with rational radius centered at origin.
    # The rational points on such a circle can be parameterized.
    
    # Even better: use the known construction via Eisenstein triples or similar.
    
    # Let's try a systematic search for D=1 using the property that
    # if (a,b,c) is a Pythagorean triple, then moving by (a/c, b/c) * r
    # from any point gives rational distance r.
    
    # Generate Pythagorean triples
    def pythagorean_triples(max_val=200):
        triples = []
        for m in range(1, int(math.sqrt(max_val)) + 1):
            for n in range(m+1, int(math.sqrt(max_val)) + 1):
                if (m - n) % 2 == 0 or (m*n) % 2 == 0:
                    a = m*m - n*n
                    b = 2*m*n
                    c = m*m + n*n
                    if a <= max_val and b <= max_val and c <= max_val:
                        triples.append((a, b, c))
        return triples

    def generate_candidates_circle_intersection(anchors, D, max_triples=100):
        """
        Given anchor points, find new points by intersecting circles.
        For each pair of anchors (A, B), and for rational radii r1, r2,
        find points P such that dist(P, A) = r1 and dist(P, B) = r2.
        
        For D=1, this reduces to standard circle-circle intersection.
        The key is to choose r1, r2 such that the intersection has rational coordinates.
        
        A simpler approach: for each anchor A, and for each Pythagorean triple (a,b,c),
        the point A + (a/c * r, b/c * r) has distance r from A for any rational r.
        We need this point to also have rational distance to all other anchors.
        """
        candidates = []
        triples = pythagorean_triples(100)[:max_triples]
        
        for anchor in anchors:
            for (a, b, c) in triples:
                # Direction vector (a/c, b/c)
                dx = Fraction(a, c)
                dy = Fraction(b, c)
                # Try various rational distances r
                for r_num in range(1, 30):
                    for r_den in range(1, 5):
                        r = Fraction(r_num, r_den)
                        cand = (anchor[0] + dx * r, anchor[1] + dy * r)
                        # Also try negative direction
                        cand_neg = (anchor[0] - dx * r, anchor[1] - dy * r)
                        for cand_pt in [cand, cand_neg]:
                            # Check if this point has rational distance to all anchors
                            valid = True
                            for a_pt in anchors:
                                if cand_pt == a_pt:
                                    valid = False
                                    break
                                if not dist_sq_rational(cand_pt, a_pt, D):
                                    valid = False
                                    break
                            if valid:
                                candidates.append(cand_pt)
        
        return candidates

    def generate_candidates_grid(D, x_range=30, y_range=30, x_den_max=4, y_den_max=4):
        """Grid search for small rational coordinates."""
        candidates = []
        for x_num in range(-x_range, x_range+1):
            for x_den in range(1, x_den_max+1):
                for y_num in range(-y_range, y_range+1):
                    for y_den in range(1, y_den_max+1):
                        x = Fraction(x_num, x_den)
                        y = Fraction(y_num, y_den)
                        candidates.append((x, y))
        return candidates

    def greedy_expand(points, D, max_size=12, deadline=135.0):
        """Greedily add points to maximize set size."""
        current = list(points)
        if not current:
            return current
        
        while len(current) < max_size:
            if time.time() > deadline:
                break
            
            # Generate candidates
            cands = generate_candidates_circle_intersection(current[:3], D, max_triples=50)
            
            # Also try grid search for small coordinates
            grid_cands = generate_candidates_grid(D, x_range=15, y_range=15, x_den_max=3, y_den_max=3)
            cands.extend(grid_cands)
            
            # Remove duplicates and points already in current
            seen = set(current)
            unique_cands = []
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    unique_cands.append(c)
            
            # Sort by distance to centroid (prefer points close to existing set)
            if unique_cands:
                cx = sum(p[0] for p in current) / len(current)
                cy = sum(p[1] for p in current) / len(current)
                unique_cands.sort(key=lambda p: (p[0]-cx)**2 + D*(p[1]-cy)**2)
            
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
                break
        
        return current

    # Try D=1 first with the known 5-point set
    for attempt in range(3):
        if time.time() > deadline:
            break
        shuffled = list(known_5pts)
        random.shuffle(shuffled)
        result = greedy_expand(shuffled, 1, max_size=12, deadline=deadline)
        if result and len(result) > best_size:
            best_size = len(result)
            best_result = to_output_format(result, 1)

    # Try other D values
    D_candidates = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19]
    
    for D in D_candidates:
        if time.time() > deadline - 5:
            break
        
        # For D > 1, start with simple configurations
        start_configs = [
            [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))],
            [(Fraction(0), Fraction(0)), (Fraction(D), Fraction(0))],
            [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(1))],
        ]
        
        for seed in start_configs:
            if time.time() > deadline - 3:
                break
            
            if not is_valid_set(seed, D):
                continue
            
            result = greedy_expand(seed, D, max_size=12, deadline=deadline)
            if result and len(result) > best_size:
                best_size = len(result)
                best_result = to_output_format(result, D)

    # Final attempt: try to improve the best result
    if time.time() < deadline - 10:
        D_best = best_result['D']
        pts_best = [(Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1])) 
                    for p in best_result['points']]
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            shuffled = list(pts_best)
            random.shuffle(shuffled)
            improved = greedy_expand(shuffled, D_best, max_size=12, deadline=deadline)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = to_output_format(improved, D_best)

    return best_result
