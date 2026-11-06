"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import gcd, isqrt
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n < 1:
        return False
    if n == 1:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _is_perfect_square(n):
    """Check if non-negative integer n is a perfect square."""
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
                    term1 = v1 * (a2 * (b3 - b4) - a3 * (b2 - b4) + a4 * (b2 - b3))
                    term2 = -a1 * (v2 * (b3 - b4) - v3 * (b2 - b4) + v4 * (b2 - b3))
                    term3 = b1 * (v2 * (a3 - a4) - v3 * (a2 - a4) + v4 * (a2 - a3))
                    term4 = -(v2 * (a3 * b4 - b3 * a4) - v3 * (a2 * b4 - b2 * a4) + v4 * (a2 * b3 - b2 * a3))
                    det = term1 + term2 + term3 + term4
                    if det == 0:
                        return False
    return True

def _check_all_pairwise_distances_rational(points, D):
    """Check that all pairwise squared distances are rational squares."""
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

def _valid_config(points, D):
    """Full validation of a configuration."""
    if len(points) < 3:
        return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    if not _check_all_pairwise_distances_rational(points, D):
        return False
    return True

def _format_points(points):
    return [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]

def _generate_candidates_d1(max_radius=200):
    """Generate integer lattice points with rational distances from origin."""
    candidates = set()
    for x in range(-max_radius, max_radius + 1):
        for y in range(-max_radius, max_radius + 1):
            # Check distance from origin is rational
            d2 = x*x + y*y
            r = isqrt(d2)
            if r * r == d2:
                candidates.add((Fraction(x), Fraction(y)))
    return list(candidates)

def _try_add_point_fast(current_points, D, candidate):
    """Fast check for adding a single point."""
    x_new, y_new = candidate
    # Check distinctness
    for x, y in current_points:
        if x == x_new and y == y_new:
            return None
    
    # Check distances to all existing points
    for x, y in current_points:
        dx = x - x_new
        dy = y - y_new
        d2 = dx*dx + D*dy*dy
        if d2 <= 0:
            return None
        if not _is_perfect_square(d2.numerator) or not _is_perfect_square(d2.denominator):
            return None
    
    new_points = current_points + [candidate]
    
    # Check collinearity with new point
    n = len(current_points)
    for i in range(n):
        for j in range(i+1, n):
            x_i, y_i = current_points[i]
            x_j, y_j = current_points[j]
            if (x_j - x_i) * (y_new - y_i) == (x_new - x_i) * (y_j - y_i):
                return None
    
    # Check concyclicity with new point (only need to check triples from existing + new)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a1, b1 = current_points[i]
                a2, b2 = current_points[j]
                a3, b3 = current_points[k]
                a4, b4 = x_new, y_new
                v1 = a1*a1 + D*b1*b1
                v2 = a2*a2 + D*b2*b2
                v3 = a3*a3 + D*b3*b3
                v4 = a4*a4 + D*b4*b4
                term1 = v1*(a2*(b3-b4) - a3*(b2-b4) + a4*(b2-b3))
                term2 = -a1*(v2*(b3-b4) - v3*(b2-b4) + v4*(b2-b3))
                term3 = b1*(v2*(a3-a4) - v3*(a2-a4) + v4*(a2-a3))
                term4 = -(v2*(a3*b4 - b3*a4) - v3*(a2*b4 - b2*a4) + v4*(a2*b3 - b2*a3))
                if term1 + term2 + term3 + term4 == 0:
                    return None
    
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Known 5-point configuration with D=1
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63)),
    ]
    D = 1

    best_points = base_points[:]

    # Verify base
    if not _valid_config(best_points, D):
        best_points = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
        D = 1
        if not _valid_config(best_points, D):
            best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        return {'D': D, 'points': _format_points(best_points)}

    # Generate candidates: integer lattice points with rational distance from origin
    candidates = _generate_candidates_d1(max_radius=300)
    # Sort by distance from origin
    candidates.sort(key=lambda p: (p[0]*p[0] + p[1]*p[1], abs(p[0]) + abs(p[1])))

    # Greedy extension
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for candidate in candidates:
            if time.time() > deadline:
                break
            new_points = _try_add_point_fast(best_points, D, candidate)
            if new_points is not None:
                best_points = new_points
                improved = True
                break

    # If we found 6+ points, try to extend further
    if len(best_points) >= 6 and time.time() < deadline:
        # Try more aggressive search with rational coordinates
        max_num = 50
        max_den = 20
        candidates_rational = []
        seen = set()
        for den in range(1, max_den + 1):
            for num in range(-max_num * den, max_num * den + 1):
                x = Fraction(num, den)
                for den2 in range(1, max_den + 1):
                    for num2 in range(-max_num * den2, max_num * den2 + 1):
                        y = Fraction(num2, den2)
                        key = (x, y)
                        if key not in seen:
                            seen.add(key)
                            candidates_rational.append(key)
        
        candidates_rational.sort(key=lambda p: abs(p[0]) + abs(p[1]))
        
        improved2 = True
        while improved2 and time.time() < deadline:
            improved2 = False
            for candidate in candidates_rational:
                if time.time() > deadline:
                    break
                new_points = _try_add_point_fast(best_points, D, candidate)
                if new_points is not None:
                    best_points = new_points
                    improved2 = True
                    break

    # Try different D values if we haven't improved much
    if len(best_points) <= 5 and time.time() < deadline:
        for D_try in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
            if time.time() > deadline:
                break
            
            # Generate base points for this D
            base = [(Fraction(0), Fraction(0))]
            # Try to build up from origin
            candidates_d = []
            for x in range(-100, 101):
                for y in range(-100, 101):
                    d2 = x*x + D_try*y*y
                    if _is_perfect_square(d2):
                        candidates_d.append((Fraction(x), Fraction(y)))
            
            candidates_d.sort(key=lambda p: abs(p[0]) + abs(p[1]))
            
            trial = base[:]
            for candidate in candidates_d:
                if time.time() > deadline:
                    break
                new_trial = _try_add_point_fast(trial, D_try, candidate)
                if new_trial is not None:
                    trial = new_trial
            
            if len(trial) > len(best_points):
                best_points = trial
                D = D_try

    # Final validation
    if not _valid_config(best_points, D):
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        D = 1

    return {'D': D, 'points': _format_points(best_points)}

# EVOLVE-BLOCK-END
