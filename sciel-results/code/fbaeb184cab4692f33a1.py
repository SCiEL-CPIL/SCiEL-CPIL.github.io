"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_squarefree(n):
    if n == 1:
        return True
    r = isqrt(n)
    for i in range(2, r + 1):
        if n % (i * i) == 0:
            return False
    return True

def _check_valid(points, D, max_numer=10**40):
    n = len(points)
    if n < 3 or n > 12:
        return False
    if D < 1 or D > 10**12:
        return False
    if not _is_squarefree(D):
        return False
    
    # Check distinctness
    for i in range(n):
        for j in range(i+1, n):
            if points[i] == points[j]:
                return False

    # Check pairwise distances rational
    for i in range(n):
        for j in range(i+1, n):
            dx_num = points[i][0][0]*points[j][0][1] - points[j][0][0]*points[i][0][1]
            dx_den = points[i][0][1]*points[j][0][1]
            dy_num = points[i][1][0]*points[j][1][1] - points[j][1][0]*points[i][1][1]
            dy_den = points[i][1][1]*points[j][1][1]
            
            num = dx_num*dx_num * dy_den*dy_den + D * dy_num*dy_num * dx_den*dx_den
            den = dx_den*dx_den * dy_den*dy_den
            if den == 0:
                return False
            if not _is_square(num * den):
                return False

    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1n, x1d = points[i][0]
                y1n, y1d = points[i][1]
                x2n, x2d = points[j][0]
                y2n, y2d = points[j][1]
                x3n, x3d = points[k][0]
                y3n, y3d = points[k][1]
                
                dx12_n = x2n*x1d - x1n*x2d
                dx12_d = x1d*x2d
                dy12_n = y2n*y1d - y1n*y2d
                dy12_d = y1d*y2d
                dx13_n = x3n*x1d - x1n*x3d
                dx13_d = x1d*x3d
                dy13_n = y3n*y1d - y1n*y3d
                dy13_d = y1d*y3d
                
                num = dx12_n*dy13_n*dy12_d*dx13_d - dy12_n*dx13_n*dx12_d*dy13_d
                if num == 0:
                    return False

    # Check no four concyclic
    for i, j, k, l in combinations(range(n), 4):
        def get_row(pt):
            x = Fraction(pt[0][0], pt[0][1])
            y = Fraction(pt[1][0], pt[1][1])
            return [x*x + D*y*y, x, y, 1]
        
        M = [get_row(points[i]), get_row(points[j]), get_row(points[k]), get_row(points[l])]
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[2][1]*m[1][2]) -
                    m[0][1]*(m[1][0]*m[2][2]-m[2][0]*m[1][2]) +
                    m[0][2]*(m[1][0]*m[2][1]-m[2][0]*m[1][1]))
        
        det = 0
        for c in range(4):
            minor = [row[:c] + row[c+1:] for row in M[1:]]
            det += ((-1)**c) * M[0][c] * det3(minor)
        
        if det == 0:
            return False

    return True

def _precompute_constraints(base_points):
    """Precompute lines and circles for a set of base points (integers, D=1)."""
    n = len(base_points)
    bxs = [p[0] for p in base_points]
    bys = [p[1] for p in base_points]
    base_set = set(base_points)
    
    # Precompute lines: dy*x - dx*y + c = 0
    lines = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = base_points[i]
            x2, y2 = base_points[j]
            dx = x2 - x1
            dy = y2 - y1
            c = dx*y1 - dy*x1
            lines.append((dy, -dx, c))
    
    # Precompute circles: x^2 + y^2 + A*x + B*y + C = 0
    circles = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = base_points[i]
                x2, y2 = base_points[j]
                x3, y3 = base_points[k]
                
                dx21 = x2 - x1
                dy21 = y2 - y1
                dx31 = x3 - x1
                dy31 = y3 - y1
                
                rhs2 = -((x2*x2 - x1*x1) + (y2*y2 - y1*y1))
                rhs3 = -((x3*x3 - x1*x1) + (y3*y3 - y1*y1))
                
                det = dx21*dy31 - dx31*dy21
                if det == 0:
                    continue
                
                A = Fraction(rhs2*dy31 - rhs3*dy21, det)
                B = Fraction(dx21*rhs3 - dx31*rhs2, det)
                C = -Fraction(x1*x1 + y1*y1) - A*x1 - B*y1
                
                circles.append((A, B, C))
    
    return bxs, bys, base_set, lines, circles

def _check_point_fast(x, y, bxs, bys, lines, circles, n_base):
    """Fast check for a candidate point (x,y) against precomputed constraints."""
    y2 = y*y
    
    # Distance check
    for i in range(n_base):
        dx = x - bxs[i]
        dy = y - bys[i]
        d2 = dx*dx + dy*dy
        if not _is_square(d2):
            return False
    
    # Collinearity check
    for dy, dx, c in lines:
        if dy*x - dx*y - c == 0:
            return False
    
    # Concyclicity check
    for A, B, C in circles:
        val = x*x + y2 + A*x + B*y + C
        if val == 0:
            return False
    
    return True

def _search_for_point(base_points, search_range, deadline, time_check_interval=200):
    """Search for a new point that extends base_points to a valid set."""
    bxs, bys, base_set, lines, circles = _precompute_constraints(base_points)
    n_base = len(base_points)
    
    count = 0
    for x in range(-search_range, search_range + 1):
        x2 = x*x
        for y in range(-search_range, search_range + 1):
            count += 1
            if count % time_check_interval == 0:
                if time.time() > deadline:
                    return None
            
            if (x, y) in base_set:
                continue
            
            if _check_point_fast(x, y, bxs, bys, lines, circles, n_base):
                # Candidate found! Verify with full check
                new_point = [[x, 1], [y, 1]]
                test_points = [ [[p[0], 1], [p[1], 1]] for p in base_points ] + [new_point]
                if _check_valid(test_points, 1):
                    return (x, y)
    
    return None

def _generate_candidates_around(base_points, max_radius, deadline):
    """Generate candidate points by searching in a structured way."""
    bxs = [p[0] for p in base_points]
    bys = [p[1] for p in base_points]
    min_x = min(bxs) - max_radius
    max_x = max(bxs) + max_radius
    min_y = min(bys) - max_radius
    max_y = max(bys) + max_radius
    
    candidates = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if (x, y) not in set(base_points):
                candidates.append((x, y))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Known 5-point sets
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (0, 33), (16, 63), (-16, 63), (-56, 33)],
    ]
    
    best_points = None
    best_count = 0
    
    for base_points in base_sets:
        if time.time() > deadline - 15:
            break
        
        # Verify base set is valid
        formatted = [[[x, 1], [y, 1]] for x, y in base_points]
        if not _check_valid(formatted, 1):
            continue
        
        current_points = list(base_points)
        
        # Try to extend to 6 points
        if time.time() < deadline - 30:
            search_ranges = [80, 120, 160, 200]
            found_6th = None
            for sr in search_ranges:
                if time.time() > deadline - 25:
                    break
                result = _search_for_point(current_points, sr, deadline - 20)
                if result is not None:
                    found_6th = result
                    break
            
            if found_6th:
                current_points.append(found_6th)
                
                # Try to extend to 7 points
                if time.time() < deadline - 20:
                    search_ranges_7 = [60, 100, 140]
                    found_7th = None
                    for sr in search_ranges_7:
                        if time.time() > deadline - 15:
                            break
                        result = _search_for_point(current_points, sr, deadline - 10)
                        if result is not None:
                            found_7th = result
                            break
                    
                    if found_7th:
                        current_points.append(found_7th)
                        
                        # Try to extend to 8 points
                        if time.time() < deadline - 10:
                            search_ranges_8 = [50, 80]
                            found_8th = None
                            for sr in search_ranges_8:
                                if time.time() > deadline - 5:
                                    break
                                result = _search_for_point(current_points, sr, deadline - 3)
                                if result is not None:
                                    found_8th = result
                                    break
                            
                            if found_8th:
                                current_points.append(found_8th)
        
        if len(current_points) > best_count:
            best_count = len(current_points)
            best_points = current_points
    
    # If no extension found, use the best base set
    if best_points is None:
        best_points = base_sets[0]
    
    formatted_points = []
    for p in best_points:
        x, y = p
        formatted_points.append([[x, 1], [y, 1]])
    
    return {'D': 1, 'points': formatted_points}

# EVOLVE-BLOCK-END
