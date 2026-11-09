"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    def is_square_frac(f):
        """Check if a non-negative Fraction is a perfect square of a rational."""
        if f < 0:
            return False
        if f == 0:
            return True
        num = f.numerator
        den = f.denominator
        sn = isqrt(num)
        sd = isqrt(den)
        return sn * sn == num and sd * sd == den
    
    def dist_sq(p, q):
        """Squared distance between two points (as Fraction)."""
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + dy * dy
    
    def has_rational_dist(p, q):
        """Check if distance between p and q is rational."""
        d2 = dist_sq(p, q)
        return is_square_frac(d2)
    
    def check_all_rational(points):
        """Check all pairwise distances are rational."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if not has_rational_dist(points[i], points[j]):
                    return False
        return True
    
    def no_three_collinear(points):
        """Check no three points are collinear."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    cross = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                    if cross == 0:
                        return False
        return True
    
    def no_four_concyclic(points):
        """Check no four points are concyclic."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        r1 = x1*x1 + y1*y1
                        r2 = x2*x2 + y2*y2
                        r3 = x3*x3 + y3*y3
                        r4 = x4*x4 + y4*y4
                        
                        # 4x4 determinant
                        # |x1 y1 r1 1|
                        # |x2 y2 r2 1| = 0
                        # |x3 y3 r3 1|
                        # |x4 y4 r4 1|
                        
                        a = x1; b = y1; c = r1; d = 1
                        e = x2; f = y2; g = r2; h = 1
                        i_ = x3; j_ = y3; k_ = r3; l_ = 1
                        m = x4; n_ = y4; o_ = r4; p_ = 1
                        
                        det = (a*(f*(k_*p_-l_*o_) - g*(j_*p_-l_*n_) + h*(j_*o_-k_*n_))
                              - b*(e*(k_*p_-l_*o_) - g*(i_*p_-l_*m) + h*(i_*o_-k_*m))
                              + c*(e*(j_*p_-l_*n_) - f*(i_*p_-l_*m) + h*(i_*n_-j_*m))
                              - d*(e*(j_*o_-k_*n_) - f*(i_*o_-k_*m) + g*(i_*n_-j_*m)))
                        if det == 0:
                            return False
        return True
    
    def is_valid(points):
        """Check all conditions."""
        if len(points) < 3:
            return False
        if not check_all_rational(points):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True
    
    def to_output(points, D=1):
        """Convert points to output format."""
        return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]}
    
    # Start with known 5-point configuration
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]
    
    best_points = base_points
    
    # Try to add a 6th point by searching
    # Strategy: search over rational x,y in a bounded region
    # For efficiency, we can fix x and solve for y, or use parametric approaches
    
    # A better approach: use the fact that for points on a line, we can parametrize
    # Let's try adding points near the existing configuration
    
    # Search strategy: try various rational (x,y) pairs
    # Limit the search to reasonable bounds
    
    max_den = 200  # Maximum denominator for search
    max_num = 500  # Maximum absolute numerator for search
    
    # Try to find a 6th point
    found_6 = False
    for x_den in range(1, max_den+1):
        if time.time() > deadline - 30:
            break
        for x_num in range(-max_num, max_num+1):
            if time.time() > deadline - 30:
                break
            x = Fraction(x_num, x_den)
            for y_den in range(1, max_den+1):
                if time.time() > deadline - 30:
                    break
                for y_num in range(-max_num, max_num+1):
                    if time.time() > deadline - 30:
                        break
                    y = Fraction(y_num, y_den)
                    candidate = (x, y)
                    # Check distance to all base points
                    ok = True
                    for bp in base_points:
                        if not has_rational_dist(candidate, bp):
                            ok = False
                            break
                    if not ok:
                        continue
                    new_points = base_points + [candidate]
                    if is_valid(new_points):
                        best_points = new_points
                        found_6 = True
                        break
                if found_6:
                    break
            if found_6:
                break
        if found_6:
            break
    
    # If we found 6 points, try to find a 7th
    if len(best_points) >= 6:
        found_7 = False
        for x_den in range(1, max_den+1):
            if time.time() > deadline - 15:
                break
            for x_num in range(-max_num, max_num+1):
                if time.time() > deadline - 15:
                    break
                x = Fraction(x_num, x_den)
                for y_den in range(1, max_den+1):
                    if time.time() > deadline - 15:
                        break
                    for y_num in range(-max_num, max_num+1):
                        if time.time() > deadline - 15:
                            break
                        y = Fraction(y_num, y_den)
                        candidate = (x, y)
                        ok = True
                        for bp in best_points:
                            if not has_rational_dist(candidate, bp):
                                ok = False
                                break
                        if not ok:
                            continue
                        new_points = best_points + [candidate]
                        if is_valid(new_points):
                            best_points = new_points
                            found_7 = True
                            break
                    if found_7:
                        break
                if found_7:
                    break
            if found_7:
                break
        if found_7:
            break
    
    return to_output(best_points)
# EVOLVE-BLOCK-END
