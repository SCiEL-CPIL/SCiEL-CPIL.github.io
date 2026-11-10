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

def _generate_pythagorean_candidates_v2(base_pts, max_coord=300):
    """Generate candidate points with integer distance to ALL base points."""
    candidates = set()
    n = len(base_pts)
    if n == 0:
        return list(candidates)
    
    # Start with points at integer distance to first base point
    bx0, by0 = base_pts[0]
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) in base_pts:
                continue
            dx = x - bx0
            dy = y - by0
            d2 = dx * dx + dy * dy
            if d2 > 0 and _is_square(d2):
                candidates.add((x, y))
    
    # Filter: keep only points at integer distance to ALL base points
    filtered = set()
    for (x, y) in candidates:
        valid = True
        for (bx, by) in base_pts:
            if (x, y) == (bx, by):
                continue
            dx = x - bx
            dy = y - by
            d2 = dx * dx + dy * dy
            if d2 == 0:
                continue
            if not _is_square(d2):
                valid = False
                break
        if valid:
            filtered.add((x, y))
    
    return list(filtered)


def _score_candidate_v2(cand, current, D=1):
    """Score a candidate: prefer spread-out points, penalize near-concyclic."""
    # Min distance to existing points (spread)
    min_dist = float('inf')
    for cp in current:
        dx = cand[0] - cp[0]
        dy = cand[1] - cp[1]
        min_dist = min(min_dist, dx*dx + D*dy*dy)
    
    # Check if adding would create concyclic quadruples (approximate penalty)
    concyclic_penalty = 0
    if len(current) >= 3:
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                for k in range(j+1, len(current)):
                    x1, y1 = current[i]
                    x2, y2 = current[j]
                    x3, y3 = current[k]
                    x4, y4 = cand
                    a1 = x1*x1 + D*y1*y1
                    a2 = x2*x2 + D*y2*y2
                    a3 = x3*x3 + D*y3*y3
                    a4 = x4*x4 + D*y4*y4
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
                        concyclic_penalty += 10000
                    elif abs(det) < 1000:
                        concyclic_penalty += abs(det)
    
    return min_dist - concyclic_penalty


def _greedy_expand_v2(current, candidates, max_points, deadline, D=1):
    """Greedy expansion with better scoring."""
    if not current:
        return current
    current = list(current)
    current_set = set(current)
    
    while len(current) < max_points:
        if time.time() > deadline:
            break
        
        best_cand = None
        best_score = -float('inf')
        
        for cand in candidates:
            if cand in current_set:
                continue
            
            # Check integer distance to all current points
            valid = True
            for cp in current:
                dx = cand[0] - cp[0]
                dy = cand[1] - cp[1]
                d2 = dx * dx + D * dy * dy
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
            
            # Score the candidate
            score = _score_candidate_v2(cand, current, D)
            
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
            # Try to find another candidate
            found = False
            for cand in sorted(candidates, key=lambda c: _score_candidate_v2(c, current, D), reverse=True):
                if cand in current_set or cand == best_cand:
                    continue
                valid = True
                for cp in current:
                    dx = cand[0] - cp[0]
                    dy = cand[1] - cp[1]
                    d2 = dx * dx + D * dy * dy
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


def _search_integral_points(max_coord=200, max_points=8, deadline=135.0):
    """Search for integer point sets with integer pairwise distances."""
    start_time = time.time()
    
    # Multiple known base configurations - include more diverse sets
    base_sets = [
        # 5-point sets
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (0, 5), (-3, 4), (5, 0)],
        [(0, 0), (5, 0), (0, 5), (-5, 0), (3, 4)],
        [(0, 0), (9, 12), (0, 15), (-9, 12), (12, 9)],
        [(0, 0), (7, 24), (0, 25), (-7, 24), (24, 7)],
        [(0, 0), (20, 21), (0, 29), (-20, 21), (21, 20)],
        [(0, 0), (12, 16), (0, 20), (-12, 16), (16, 12)],
        [(0, 0), (8, 15), (0, 17), (-8, 15), (15, 8)],
        # Additional diverse bases
        [(0, 0), (4, 3), (0, 5), (-4, 3), (5, 0)],
        [(0, 0), (12, 5), (0, 13), (-12, 5), (5, 12)],
        [(0, 0), (28, 45), (0, 53), (-28, 45), (45, 28)],
        [(0, 0), (16, 63), (-16, 63), (0, 63), (56, 33)],
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
    
    # Try greedy expansion from each valid base set with improved algorithm
    for bs in base_sets:
        if time.time() > deadline - 10:
            break
        if not _validate_set(bs, 1):
            continue
        
        # Generate candidates that have integer distance to ALL points in base
        cands = _generate_pythagorean_candidates_v2(bs, max_coord)
        
        # Also add some spread candidates
        for x in range(-max_coord, max_coord+1, 20):
            for y in range(-max_coord, max_coord+1, 20):
                if (x, y) not in bs:
                    cands.append((x, y))
        
        exp = _greedy_expand_v2(bs, cands, max_points, deadline, D=1)
        if len(exp) > best_count:
            best_count = len(exp)
            best_set = exp
    
    # Try remove-and-replace on best set
    if best_count < max_points and time.time() < deadline - 15:
        candidates2 = _generate_pythagorean_candidates_v2(best_set, max_coord)
        improved = _try_remove_and_replace(best_set, candidates2, max_points, deadline)
        if len(improved) > best_count:
            best_count = len(improved)
            best_set = improved
    
    # Final attempt: try to add points to best set with larger search
    if best_count < max_points and time.time() < deadline - 5:
        large_cands = _generate_pythagorean_candidates_v2(best_set, max_coord + 100)
        final_exp = _greedy_expand_v2(best_set, large_cands, max_points, deadline, D=1)
        if len(final_exp) > best_count:
            best_count = len(final_exp)
            best_set = final_exp
    
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
