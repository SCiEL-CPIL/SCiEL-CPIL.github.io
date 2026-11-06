"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import gcd
from math import isqrt

# EVOLVE-BLOCK-START

def _is_perfect_square(n):
    """Check if n is a perfect square using integer arithmetic."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_no_three_collinear(points):
    """Check that no three points are collinear using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                x_k, y_k = points[k]
                if (x_j - x_i) * (y_k - y_i) == (x_k - x_i) * (y_j - y_i):
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check that no four points are concyclic using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    a1, b1 = points[i]
                    a2, b2 = points[j]
                    a3, b3 = points[k]
                    a4, b4 = points[l]
                    
                    v1 = a1 * a1 + D * b1 * b1
                    v2 = a2 * a2 + D * b2 * b2
                    v3 = a3 * a3 + D * b3 * b3
                    v4 = a4 * a4 + D * b4 * b4
                    
                    # Determinant of [[v1, a1, b1, 1], [v2, a2, b2, 1], [v3, a3, b3, 1], [v4, a4, b4, 1]]
                    term1 = v1 * (a2 * (b3 - b4) - a3 * (b2 - b4) + a4 * (b2 - b3))
                    term2 = -a1 * (v2 * (b3 - b4) - v3 * (b2 - b4) + v4 * (b2 - b3))
                    term3 = b1 * (v2 * (a3 - a4) - v3 * (a2 - a4) + v4 * (a2 - a3))
                    term4 = -(v2 * (a3 * b4 - b3 * a4) - v3 * (a2 * b4 - b2 * a4) + v4 * (a2 * b3 - b2 * a3))
                    det = term1 + term2 + term3 + term4
                    if det == 0:
                        return False
    return True

def _check_all_pairwise_distances_rational(points, D):
    """Check that all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x_i, y_i = points[i]
            x_j, y_j = points[j]
            dx = x_i - x_j
            dy = y_i - y_j
            dist_sq = dx * dx + D * dy * dy
            if dist_sq <= 0:
                return False
            num = dist_sq.numerator
            den = dist_sq.denominator
            if not _is_perfect_square(num):
                return False
            if not _is_perfect_square(den):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s margin
    
    def format_result(points, D):
        result_points = []
        for x, y in points:
            result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        return {'D': D, 'points': result_points}
    
    # Known 5-point configuration with D=1
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    base_points = [(Fraction(0), Fraction(0)),
                   (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)),
                   (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]
    D_base = 1
    
    # Verify base points
    if not (_check_no_three_collinear(base_points) and
            _check_no_four_concyclic(base_points, D_base) and
            _check_all_pairwise_distances_rational(base_points, D_base)):
        # This should never happen, but fallback to a simpler set if it does
        base_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4)), (Fraction(3), Fraction(4))]
        D_base = 1
        # Re-verify
        if not (_check_no_three_collinear(base_points) and
                _check_no_four_concyclic(base_points, D_base) and
                _check_all_pairwise_distances_rational(base_points, D_base)):
            # Extreme fallback: 3 points
            base_points = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
            D_base = 1

    best_points = base_points
    best_D = D_base
    
    # Try to extend to 6 points
    # Search for a 6th point for the base configuration
    # We use a limited search to avoid timeout
    search_limit = 100 # Small range for 6th point search
    found_six = False
    
    # Try to add a point to the 5-point set
    # We scan a grid around the existing points
    # To speed up, we can pre-compute some properties
    # But simple grid scan is fine for small range
    
    # Generate candidates
    # The 5 points are within x: [-56, 16], y: [0, 63]
    # We search in a box slightly larger than this
    x_min = -60
    x_max = 20
    y_min = -10
    y_max = 80
    
    # To make it faster, we can iterate over integer coordinates
    # and check if they form rational distances
    # Note: The 6th point must have rational x, y.
    # We try integer x, y first.
    
    candidates = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            candidates.append((Fraction(x), Fraction(y)))
    
    # Sort candidates by distance to centroid? 
    # Or just iterate.
    
    # Time check
    if time.time() < deadline - 10:
        for candidate in candidates:
            if time.time() > deadline - 5:
                break
            # Skip if already in set
            if any(candidate[0] == p[0] and candidate[1] == p[1] for p in best_points):
                continue
            
            # Check distances to all existing points
            valid = True
            for p in best_points:
                dx = p[0] - candidate[0]
                dy = p[1] - candidate[1]
                dist_sq = dx * dx + D_base * dy * dy
                if dist_sq <= 0:
                    valid = False
                    break
                if not _is_perfect_square(dist_sq.numerator):
                    valid = False
                    break
                if not _is_perfect_square(dist_sq.denominator):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check collinearity with all pairs
            new_points = best_points + [candidate]
            if not _check_no_three_collinear(new_points):
                continue
            
            # Check concyclicity
            if not _check_no_four_concyclic(new_points, D_base):
                continue
            
            # If we get here, it's a valid 6-point set
            best_points = new_points
            found_six = True
            break
    
    # If 6 points not found, try other D values briefly?
    # Given the difficulty, it's unlikely to find 6 for other D in short time.
    # We stick with the best found.
    
    return format_result(best_points, best_D)

# EVOLVE-BLOCK-END
