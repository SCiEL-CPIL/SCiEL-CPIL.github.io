import time
import math
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    a1, b1 = p1
    a2, b2 = p2
    a3, b3 = p3
    a4, b4 = p4
    
    if _are_collinear(p1, p2, p3):
        return False
    
    rhs1 = -(a2**2 - a1**2 + b2**2 - b1**2)
    rhs2 = -(a3**2 - a1**2 + b3**2 - b1**2)
    
    x1 = a2 - a1
    y1 = b2 - b1
    x2 = a3 - a1
    y2 = b3 - a1
    
    det = x1*y2 - x2*y1
    if det == 0:
        return False
    
    num_D = rhs1*y2 - rhs2*y1
    num_E = x1*rhs2 - x2*rhs1
    num_F = -((a1**2 + b1**2)*det + num_D*a1 + num_E*b1)
    
    check = (a4**2 + b4**2)*det + num_D*a4 + num_E*b4 + num_F
    return check == 0

def _validate_point_set(points):
    n = len(points)
    # Check all pairwise distances are perfect squares
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx*dx + dy*dy
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _try_known_6point_sets():
    """Try several known 6-point rational distance sets."""
    known_sets = [
        # Set 1: Based on Pythagorean triples
        [(0, 0), (3, 4), (0, 5), (4, 3), (5, 0), (4, 0)],
        # Set 2: Another configuration
        [(0, 0), (5, 0), (0, 5), (5, 5), (3, 4), (4, 3)],
        # Set 3: Symmetric configuration
        [(-5, 0), (5, 0), (0, 5), (0, -5), (3, 4), (-3, -4)],
        # Set 4: Extended from base5
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (56, 33)],
        # Set 5: Another attempt
        [(0, 0), (7, 24), (0, 25), (24, 7), (25, 0), (24, 24)],
        # Set 6: Using 8-15-17 triples
        [(0, 0), (8, 15), (0, 17), (15, 8), (17, 0), (15, 15)],
        # Set 7: Mixed approach
        [(0, 0), (5, 12), (0, 13), (12, 5), (13, 0), (12, 12)],
        # Set 8: Larger scale
        [(0, 0), (20, 21), (0, 29), (21, 20), (29, 0), (21, 21)],
    ]
    
    for pts in known_sets:
        if _validate_point_set(pts):
            return pts
    return None

def _try_add_point(base_points, x, y):
    new_point = (x, y)
    if new_point in base_points:
        return False
    for p in base_points:
        dx = x - p[0]
        dy = y - p[1]
        d2 = dx*dx + dy*dy
        if not _is_square(d2):
            return False
    n = len(base_points)
    for i in range(n):
        for j in range(i+1, n):
            if _are_collinear(base_points[i], base_points[j], new_point):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_concyclic(base_points[i], base_points[j], base_points[k], new_point):
                    return False
    return True

def _find_candidates_with_int_dist(base_points, max_coord):
    candidates = set()
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            valid = True
            for p in base_points:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx*dx + dy*dy
                if not _is_square(d2):
                    valid = False
                    break
            if valid:
                candidates.add((x, y))
    return list(candidates)

def _hilbert_param_points(center, radius, n_params):
    """Generate points on a circle using rational parameterization.
    Points are (center_x + radius*(1-t^2)/(1+t^2), center_y + radius*2t/(1+t^2))
    for rational t. These are rationally distant from center."""
    points = []
    cx, cy = center
    r = radius
    for t_num in range(-n_params, n_params+1):
        for t_den in range(1, n_params+1):
            if math.gcd(abs(t_num), t_den) != 1:
                continue
            t = Fraction(t_num, t_den)
            # Rational point on circle of radius r centered at (cx, cy)
            x = cx + r * (1 - t*t) / (1 + t*t)
            y = cy + r * 2*t / (1 + t*t)
            # Convert to integers if possible
            if x.denominator == 1 and y.denominator == 1:
                points.append((x.numerator, y.numerator))
    return points

def _generate_pythagorean_points(max_leg):
    """Generate points with integer coordinates at integer distance from origin
    using Pythagorean triples."""
    points = [(0, 0)]
    for a in range(1, max_leg + 1):
        for b in range(a + 1, max_leg + 1):
            c2 = a*a + b*b
            c = math.isqrt(c2)
            if c * c == c2:
                # Add all sign combinations
                for sa in [1, -1]:
                    for sb in [1, -1]:
                        points.append((sa * a, sb * b))
                        points.append((sa * b, sb * a))
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Known good 5-point base set
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _validate_point_set(base5):
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [5, 1]]]}
    
    best_set = base5.copy()
    
    # Strategy 0: Try known 6-point configurations first
    known_6 = _try_known_6point_sets()
    if known_6 is not None:
        best_set = known_6
        # Try to extend to 7 points
        if time.time() < deadline - 10:
            for max_coord in [50, 80, 100, 150, 200]:
                if time.time() > deadline - 5:
                    break
                candidates7 = _find_candidates_with_int_dist(best_set, max_coord)
                candidates7 = [p for p in candidates7 if p not in best_set]
                cx = sum(p[0] for p in best_set) / len(best_set)
                cy = sum(p[1] for p in best_set) / len(best_set)
                candidates7.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
                
                for cand in candidates7:
                    if time.time() > deadline - 2:
                        break
                    if _try_add_point(best_set, cand[0], cand[1]):
                        new_set = best_set + [cand]
                        if _validate_point_set(new_set):
                            best_set = new_set
                            break
                if len(best_set) >= 7:
                    break
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
    
    # Strategy 1: Search for 6th point in expanding grid
    for max_coord in [50, 80, 100, 150, 200, 300, 400, 500, 600, 800]:
        if time.time() > deadline - 5:
            break
        
        candidates = _find_candidates_with_int_dist(base5, max_coord)
        candidates = [p for p in candidates if p not in base5]
        
        # Sort by distance to centroid
        cx = sum(p[0] for p in base5) / 5
        cy = sum(p[1] for p in base5) / 5
        candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for cand in candidates:
            if time.time() > deadline - 2:
                break
            if _try_add_point(base5, cand[0], cand[1]):
                new_set = base5 + [cand]
                if _validate_point_set(new_set):
                    best_set = new_set
                    # Try to extend to 7 points
                    if time.time() < deadline - 10:
                        for max_coord7 in [50, 80, 100, 150, 200]:
                            if time.time() > deadline - 5:
                                break
                            candidates7 = _find_candidates_with_int_dist(best_set, max_coord7)
                            candidates7 = [p for p in candidates7 if p not in best_set]
                            cx7 = sum(p[0] for p in best_set) / len(best_set)
                            cy7 = sum(p[1] for p in best_set) / len(best_set)
                            candidates7.sort(key=lambda p: (p[0]-cx7)**2 + (p[1]-cy7)**2)
                            
                            for cand7 in candidates7:
                                if time.time() > deadline - 2:
                                    break
                                if _try_add_point(best_set, cand7[0], cand7[1]):
                                    new_set7 = best_set + [cand7]
                                    if _validate_point_set(new_set7):
                                        best_set = new_set7
                                        break
                            if len(best_set) >= 7:
                                break
                    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
    
    # Strategy 2: Use Hilbert parameterization to find points on circles
    if len(best_set) < 6 and time.time() < deadline - 10:
        # Try to find a 6th point using circle parameterization around existing points
        for base_pt in best_set:
            for radius in [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
                if time.time() > deadline - 3:
                    break
                circ_pts = _hilbert_param_points(base_pt, radius, 25)
                for cand in circ_pts:
                    if cand in best_set:
                        continue
                    if _try_add_point(best_set, cand[0], cand[1]):
                        new_set = best_set + [cand]
                        if _validate_point_set(new_set):
                            best_set = new_set
                            break
                if len(best_set) >= 6:
                    break
            if len(best_set) >= 6:
                break
    
    # Strategy 3: Use Pythagorean points
    if len(best_set) < 6 and time.time() < deadline - 10:
        pyth_pts = _generate_pythagorean_points(50)
        for cand in pyth_pts:
            if time.time() > deadline - 3:
                break
            if cand in best_set:
                continue
            if _try_add_point(best_set, cand[0], cand[1]):
                new_set = best_set + [cand]
                if _validate_point_set(new_set):
                    best_set = new_set
                    break
        if len(best_set) >= 6:
            # Try to extend to 7
            if time.time() < deadline - 5:
                for max_coord in [50, 80, 100]:
                    if time.time() > deadline - 2:
                        break
                    candidates7 = _find_candidates_with_int_dist(best_set, max_coord)
                    candidates7 = [p for p in candidates7 if p not in best_set]
                    for cand in candidates7:
                        if time.time() > deadline - 1:
                            break
                        if _try_add_point(best_set, cand[0], cand[1]):
                            new_set = best_set + [cand]
                            if _validate_point_set(new_set):
                                best_set = new_set
                                break
                    if len(best_set) >= 7:
                        break
    
    # If we found a 6th point, try to find a 7th
    if len(best_set) == 6 and time.time() < deadline - 10:
        for max_coord in [50, 80, 100, 150, 200]:
            if time.time() > deadline - 3:
                break
            candidates7 = _find_candidates_with_int_dist(best_set, max_coord)
            candidates7 = [p for p in candidates7 if p not in best_set]
            cx = sum(p[0] for p in best_set) / len(best_set)
            cy = sum(p[1] for p in best_set) / len(best_set)
            candidates7.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
            
            for cand in candidates7:
                if time.time() > deadline - 1:
                    break
                if _try_add_point(best_set, cand[0], cand[1]):
                    new_set = best_set + [cand]
                    if _validate_point_set(new_set):
                        best_set = new_set
                        break
            if len(best_set) == 7:
                break
        
        # Also try circle parameterization for 7th point
        if len(best_set) == 6 and time.time() < deadline - 5:
            for base_pt in best_set:
                for radius in [5, 10, 15, 20, 30, 40, 50, 60]:
                    if time.time() > deadline - 2:
                        break
                    circ_pts = _hilbert_param_points(base_pt, radius, 15)
                    for cand in circ_pts:
                        if cand in best_set:
                            continue
                        if _try_add_point(best_set, cand[0], cand[1]):
                            new_set = best_set + [cand]
                            if _validate_point_set(new_set):
                                best_set = new_set
                                break
                    if len(best_set) >= 7:
                        break
                if len(best_set) >= 7:
                    break
    
    # If we have 7 points, try for 8
    if len(best_set) == 7 and time.time() < deadline - 5:
        for max_coord in [50, 80, 100, 150]:
            if time.time() > deadline - 2:
                break
            candidates8 = _find_candidates_with_int_dist(best_set, max_coord)
            candidates8 = [p for p in candidates8 if p not in best_set]
            cx = sum(p[0] for p in best_set) / len(best_set)
            cy = sum(p[1] for p in best_set) / len(best_set)
            candidates8.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
            
            for cand in candidates8:
                if time.time() > deadline - 1:
                    break
                if _try_add_point(best_set, cand[0], cand[1]):
                    new_set = best_set + [cand]
                    if _validate_point_set(new_set):
                        best_set = new_set
                        break
            if len(best_set) == 8:
                break
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

# EVOLVE-BLOCK-END
