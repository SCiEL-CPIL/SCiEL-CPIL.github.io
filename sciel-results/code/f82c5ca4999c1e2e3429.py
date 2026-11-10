"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    a1 = x1*x1 + y1*y1
                    a2 = x2*x2 + y2*y2
                    a3 = x3*x3 + y3*y3
                    a4 = x4*x4 + y4*y4
                    b1 = a2 - a1
                    b2 = a3 - a1
                    b3 = a4 - a1
                    c1 = x2 - x1
                    c2 = x3 - x1
                    c3 = x4 - x1
                    d1 = y2 - y1
                    d2 = y3 - y1
                    d3 = y4 - y1
                    det = (b1 * (c2 * d3 - c3 * d2)
                           - c1 * (b2 * d3 - b3 * d2)
                           + d1 * (b2 * c3 - b3 * c2))
                    if det == 0:
                        return False
    return True

def _all_distances_rational(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            d2 = dx * dx + D * dy * dy
            if not _is_square(d2):
                return False
    return True

def _validate_set(pts, D=1):
    if len(pts) < 3:
        return False
    # Check distinctness
    seen = set()
    for p in pts:
        if p in seen:
            return False
        seen.add(p)
    return (_check_no_three_collinear(pts) and 
            _check_no_four_concyclic(pts) and 
            _all_distances_rational(pts, D))

def _generate_pythagorean_candidates(base_pts, max_coord=300):
    """Generate candidate points that have integer distance to at least one base point."""
    candidates = set()
    for bx, by in base_pts:
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                if (x, y) in base_pts:
                    continue
                dx = x - bx
                dy = y - by
                d2 = dx * dx + dy * dy
                if d2 > 0 and _is_square(d2):
                    candidates.add((x, y))
    return list(candidates)

def _greedy_expand(current, candidates, max_points, deadline):
    """Try to greedily add points to current set."""
    if not current:
        return current
    current = list(current)
    current_set = set(current)
    
    while len(current) < max_points:
        if time.time() > deadline:
            break
        
        # Score candidates: count integer distances to all current points
        best_cand = None
        best_score = -1
        best_collinear_ok = False
        
        for cand in candidates:
            if cand in current_set:
                continue
            
            # Check integer distance to all current points
            valid = True
            for cp in current:
                dx = cand[0] - cp[0]
                dy = cand[1] - cp[1]
                d2 = dx * dx + dy * dy
                if not _is_square(d2):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check no three collinear with current
            collinear_ok = True
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    x1, y1 = current[i]
                    x2, y2 = current[j]
                    x3, y3 = cand
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        collinear_ok = False
                        break
                if not collinear_ok:
                    break
            
            if not collinear_ok:
                continue
            
            # Score: prefer points that are "spread out" - maximize min distance to existing
            min_dist = float('inf')
            for cp in current:
                dx = cand[0] - cp[0]
                dy = cand[1] - cp[1]
                min_dist = min(min_dist, dx*dx + dy*dy)
            
            score = min_dist
            
            if score > best_score:
                best_score = score
                best_cand = cand
        
        if best_cand is None:
            break
        
        # Check concyclic constraint
        new_set = current + [best_cand]
        if _check_no_four_concyclic(new_set):
            current = new_set
            current_set.add(best_cand)
        else:
            # Try to find another candidate that works
            found = False
            for cand in candidates:
                if cand in current_set or cand == best_cand:
                    continue
                valid = True
                for cp in current:
                    dx = cand[0] - cp[0]
                    dy = cand[1] - cp[1]
                    d2 = dx * dx + dy * dy
                    if not _is_square(d2):
                        valid = False
                        break
                if not valid:
                    continue
                collinear_ok = True
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        x1, y1 = current[i]
                        x2, y2 = current[j]
                        x3, y3 = cand
                        if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                            collinear_ok = False
                            break
                    if not collinear_ok:
                        break
                if not collinear_ok:
                    continue
                new_set = current + [cand]
                if _check_no_four_concyclic(new_set):
                    current = new_set
                    current_set.add(cand)
                    found = True
                    break
            if not found:
                break
    
    return current

def _try_remove_and_replace(pts, candidates, max_points, deadline):
    """Try removing one point and replacing with a better one."""
    best = list(pts)
    best_len = len(pts)
    
    for remove_idx in range(len(pts)):
        if time.time() > deadline:
            break
        
        reduced = pts[:remove_idx] + pts[remove_idx+1:]
        if not _check_no_three_collinear(reduced) or not _check_no_four_concyclic(reduced):
            continue
        
        # Try to expand the reduced set
        expanded = _greedy_expand(reduced, candidates, max_points, deadline)
        if len(expanded) > best_len:
            best = expanded
            best_len = len(expanded)
    
    return best

def _search_integral_points(max_coord=200, max_points=8, deadline=135.0):
    """Search for integer point sets with integer pairwise distances."""
    start_time = time.time()
    
    # Multiple known base configurations
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (0, 5), (-3, 4), (5, 0)],
        [(0, 0), (5, 0), (0, 5), (-5, 0), (3, 4)],
        [(0, 0), (9, 12), (0, 15), (-9, 12), (12, 9)],
        [(0, 0), (7, 24), (0, 25), (-7, 24), (24, 7)],
        [(0, 0), (20, 21), (0, 29), (-20, 21), (21, 20)],
        [(0, 0), (12, 16), (0, 20), (-12, 16), (16, 12)],
        [(0, 0), (8, 15), (0, 17), (-8, 15), (15, 8)],
    ]
    
    best_set = None
    best_count = 0
    
    # Validate and use base sets
    for bs in base_sets:
        if _validate_set(bs, 1):
            if len(bs) > best_count:
                best_count = len(bs)
                best_set = list(bs)
    
    if best_set is None:
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_count = 5
    
    # Generate candidates from the best base set
    candidates = _generate_pythagorean_candidates(best_set, max_coord)
    
    # Also add some random-ish spread candidates
    for x in range(-max_coord, max_coord+1, 10):
        for y in range(-max_coord, max_coord+1, 10):
            candidates.append((x, y))
    
    # Try greedy expansion from best base
    expanded = _greedy_expand(best_set, candidates, max_points, deadline)
    if len(expanded) > best_count:
        best_count = len(expanded)
        best_set = expanded
    
    # Try from each valid base set
    for bs in base_sets:
        if time.time() > deadline - 5:
            break
        if not _validate_set(bs, 1):
            continue
        
        # Generate candidates specific to this base
        cands = _generate_pythagorean_candidates(bs, min(max_coord, 150))
        
        exp = _greedy_expand(bs, cands, max_points, deadline)
        if len(exp) > best_count:
            best_count = len(exp)
            best_set = exp
    
    # Try remove-and-replace on best set
    if best_count < max_points and time.time() < deadline - 10:
        candidates2 = _generate_pythagorean_candidates(best_set, max_coord)
        improved = _try_remove_and_replace(best_set, candidates2, max_points, deadline)
        if len(improved) > best_count:
            best_count = len(improved)
            best_set = improved
    
    return best_set if best_set else [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    best_set = _search_integral_points(max_coord=150, max_points=8, deadline=deadline)
    
    # Final verification
    if not _validate_set(best_set, 1):
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
# EVOLVE-BLOCK-END
