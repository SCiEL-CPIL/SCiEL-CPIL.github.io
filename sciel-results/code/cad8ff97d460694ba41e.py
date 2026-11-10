import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    # Fallback: 5 integer points
    # These are known to be valid (rational distances, no 3 collinear, no 4 concyclic)
    fallback_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    fallback_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in fallback_points]}
    
    def is_rational_square(val):
        """Check if val (Fraction) is a square of a rational number."""
        if val < 0:
            return False
        num = val.numerator
        den = val.denominator
        if num == 0:
            return True
        s_num = int(math.isqrt(num))
        if s_num * s_num != num:
            return False
        s_den = int(math.isqrt(den))
        if s_den * s_den != den:
            return False
        return True

    def sqrt_frac(f):
        """Compute the square root of a Fraction if it is a perfect square."""
        if f < 0:
            return None
        if f == 0:
            return Fraction(0)
        num = f.numerator
        den = f.denominator
        s_num = int(math.isqrt(num))
        if s_num * s_num != num:
            return None
        s_den = int(math.isqrt(den))
        if s_den * s_den != den:
            return None
        return Fraction(s_num, s_den)

    def check_concyclic(D, p1, p2, p3, p4):
        """
        Check if 4 points are concyclic.
        Points are (x, y*sqrt(D)).
        Condition: Det of matrix [[x^2+D y^2, x, y*sqrt(D), 1]] is 0.
        Det = sqrt(D) * S, where S is rational.
        So we check if S == 0.
        S = sum_{i=1}^4 (-1)^{i+3} y_i * Det(M_i)
        where M_i is the 3x3 minor obtained by removing row i and column 3.
        """
        pts = [p1, p2, p3, p4]
        # Matrix rows: [K_i, x_i, 1] for the minor?
        # Original matrix columns: K, x, Y, 1.
        # Minor removes col 3 (Y). So columns are K, x, 1.
        # Wait, expansion is along col 3 (index 2).
        # Row i: [K_i, x_i, 1] is NOT correct.
        # The matrix is:
        # K1, x1, y1*sqrt(D), 1
        # K2, x2, y2*sqrt(D), 1
        # K3, x3, y3*sqrt(D), 1
        # K4, x4, y4*sqrt(D), 1
        # 
        # Determinant expansion along column 3 (index 2):
        # Term i: (-1)^{i+2} * y_i*sqrt(D) * Det(Minor_{i,2})
        # Minor_{i,2} is the 3x3 matrix formed by removing row i and col 2.
        # The remaining columns are K, x, 1.
        
        S = Fraction(0)
        for i in range(4):
            xi, yi = pts[i]
            Ki = xi*xi + D*yi*yi
            # Construct 3x3 matrix for minor
            # Rows j != i
            # Cols: K, x, 1
            m = []
            for j in range(4):
                if j == i: continue
                xj, yj = pts[j]
                Kj = xj*xj + D*yj*yj
                m.append([Kj, xj, Fraction(1)])
            
            # Calculate determinant of 3x3 matrix m
            # m[0][0]*m[1][1]*m[2][2] + m[0][1]*m[1][2]*m[2][0] + m[0][2]*m[1][0]*m[2][1]
            # - m[0][2]*m[1][1]*m[2][0] - m[0][1]*m[1][0]*m[2][2] - m[0][0]*m[1][2]*m[2][1]
            det_minor = (
                m[0][0]*m[1][1]*m[2][2] + m[0][1]*m[1][2]*m[2][0] + m[0][2]*m[1][0]*m[2][1]
                - m[0][2]*m[1][1]*m[2][0] - m[0][1]*m[1][0]*m[2][2] - m[0][0]*m[1][2]*m[2][1]
            )
            
            sign = 1 if (i + 2) % 2 == 0 else -1
            S += sign * yi * det_minor
            
        return S == 0

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0

    def generate_candidates(D, limit=200):
        """Generate candidate points with rational distances to (0,0) and (1,0)."""
        candidates = []
        # P1=(0,0), P2=(1,0)
        # x = (r1^2 - r2^2 + 1) / 2
        # D y^2 = r1^2 - x^2
        
        # We iterate over small integers for r1, r2 to keep numbers small.
        # r1, r2 are distances.
        # To ensure x is rational, r1^2 - r2^2 must be rational.
        # If we pick integer r1, r2, x is integer or half-integer.
        
        for r1_num in range(1, 30):
            for r2_num in range(1, 30):
                if time.time() > deadline:
                    break
                r1 = Fraction(r1_num)
                r2 = Fraction(r2_num)
                x = (r1*r1 - r2*r2 + 1) / 2
                # Check bounds for x to avoid huge numbers? 
                # The problem allows up to 10^40, so we are safe with small ints.
                
                rhs = r1*r1 - x*x
                if rhs < 0:
                    continue
                y_sq = rhs / D
                if not is_rational_square(y_sq):
                    continue
                y = sqrt_frac(y_sq)
                if y is None:
                    continue
                
                p = (x, y)
                # Check distinct from P1, P2
                if p == (Fraction(0), Fraction(0)) or p == (Fraction(1), Fraction(0)):
                    continue
                
                candidates.append(p)
                
                # Also consider negative y
                if y != 0:
                    candidates.append((x, -y))
                    
                if len(candidates) >= limit * 2:
                    break
            if len(candidates) >= limit * 2:
                break
        return candidates

    def find_max_set(D):
        """Find the largest set of points for given D."""
        candidates = generate_candidates(D, limit=100)
        if not candidates:
            return []
        
        # Start with P1, P2
        P1 = (Fraction(0), Fraction(0))
        P2 = (Fraction(1), Fraction(0))
        selected = [P1, P2]
        
        # Filter candidates that are not collinear with P1, P2 (which is y=0)
        # Since P1, P2 are on x-axis, any point with y != 0 is not collinear.
        # All our candidates have y != 0?
        # If y=0, then D y^2 = 0 => r1^2 = x^2.
        # x = (r1^2 - r2^2 + 1)/2.
        # If y=0, the point is on the axis. We should avoid these to ensure general position easily.
        filtered_candidates = [p for p in candidates if p[1] != 0]
        
        # Greedy selection
        for p in filtered_candidates:
            if time.time() > deadline:
                break
            
            # Check distinct
            if p in selected:
                continue
            
            # Check collinearity with all pairs in selected
            collinear = False
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if check_collinear(selected[i], selected[j], p):
                        collinear = True
                        break
                if collinear:
                    break
            
            if collinear:
                continue
            
            # Check rational distances to all selected
            dist_ok = True
            for s in selected:
                dx = p[0] - s[0]
                dy = p[1] - s[1]
                d2 = dx*dx + D*dy*dy
                if not is_rational_square(d2):
                    dist_ok = False
                    break
            
            if not dist_ok:
                continue
            
            # Check concyclicity with all triples in selected
            concyclic = False
            n_sel = len(selected)
            for i in range(n_sel):
                for j in range(i+1, n_sel):
                    for k in range(j+1, n_sel):
                        if check_concyclic(D, selected[i], selected[j], selected[k], p):
                            concyclic = True
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
            
            if concyclic:
                continue
            
            # If all checks pass, add point
            selected.append(p)
            
            # We want to maximize points. 
            # If we reach 12, we can stop.
            if len(selected) >= 12:
                break
                
        return selected

    best_points = []
    
    # Try D=1 first (integers)
    # D=1 candidates are integers or half-integers.
    # Known 5 points exist.
    
    for D in [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31]:
        if time.time() > deadline:
            break
        
        pts = find_max_set(D)
        
        if len(pts) > len(best_points):
            best_points = pts
            if len(pts) >= 12:
                break
        
        # If we found 8 or more, it's a good result, but let's try a bit longer if time permits
        # The search space is limited by the candidate generation.
        
    if not best_points:
        return fallback_result
        
    # Final verification of best_points
    # Convert to output format
    out_points = []
    for x, y in best_points:
        out_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        
    return {'D': D, 'points': out_points}

# EVOLVE-BLOCK-END
