"""Structural rewrite: rational parametrization + conic search for integral-distance sets."""
from fractions import Fraction
import math
import time
import random

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n <= 1:
        return True
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            return False
        p += 1 if p == 2 else 2
    return True


def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _dist2_is_rational_sq(p1, p2, D):
    """Check if squared distance between two points is a perfect square rational."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d2 = dx * dx + D * dy * dy
    if d2 < 0:
        return False
    num = d2.numerator
    den = d2.denominator
    return _is_perfect_square(num) and _is_perfect_square(den)


def _check_pairwise(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist2_is_rational_sq(pts[i], pts[j], D):
                return False
    return True


def _check_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) == \
                   (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]):
                    return False
    return True


def _check_concyclic(pts, D):
    """Check no four points lie on a conic x^2 + D*y^2 + Ax + By + C = 0."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    a1 = x1 * x1 + D * y1 * y1
                    a2 = x2 * x2 + D * y2 * y2
                    a3 = x3 * x3 + D * y3 * y3
                    a4 = x4 * x4 + D * y4 * y4
                    det = (x1 * (y2 * (a3 - a4) - y3 * (a2 - a4) + y4 * (a2 - a3)) -
                           y1 * (x2 * (a3 - a4) - x3 * (a2 - a4) + x4 * (a2 - a3)) +
                           a1 * (x2 * y3 - x3 * y2 - x2 * y4 + x4 * y2 + x3 * y4 - x4 * y3) -
                           (x2 * y3 * a4 - x2 * y4 * a3 - x3 * y2 * a4 + x3 * y4 * a2 +
                            x4 * y2 * a3 - x4 * y3 * a2))
                    if det == 0:
                        return False
    return True


def _check_valid(pts, D):
    n = len(pts)
    if n < 3:
        return False
    seen = set()
    for p in pts:
        if p in seen:
            return False
        seen.add(p)
    if not _check_pairwise(pts, D):
        return False
    if not _check_collinear(pts):
        return False
    if n >= 4 and not _check_concyclic(pts, D):
        return False
    return True


def _check_partial(pts, D):
    """Fast partial check: pairwise + collinear (no concyclic)."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist2_is_rational_sq(pts[i], pts[j], D):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) == \
                   (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]):
                    return False
    return True


def _to_output(pts):
    return [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts]


def _search_integer_grid(D, limit, target_n, deadline, max_nodes=200000):
    """Backtracking search on integer grid for target_n points."""
    candidates = [(Fraction(x), Fraction(y)) for x in range(-limit, limit + 1)
                  for y in range(-limit, limit + 1)]
    random.seed(D * 1000 + limit)
    random.shuffle(candidates)
    
    result_box = [None]
    node_count = [0]
    
    def bt(current, idx):
        if time.time() > deadline or node_count[0] > max_nodes:
            return False
        if len(current) >= target_n:
            if _check_concyclic(current, D):
                result_box[0] = list(current)
                return True
            return False
        for i in range(idx, len(candidates)):
            node_count[0] += 1
            if time.time() > deadline or node_count[0] > max_nodes:
                return False
            cand = candidates[i]
            new_cur = current + [cand]
            if _check_partial(new_cur, D):
                if bt(new_cur, i + 1):
                    return True
        return False
    
    if bt([], 0):
        return result_box[0]
    return None


def _search_conic_param(D, target_n, deadline):
    """
    Generate points via rational parametrization of a conic, then search subsets.
    Points on x^2 + D*y^2 = R^2 have rational distances from origin.
    We parametrize and look for subsets with all pairwise rational distances.
    """
    # Use the parametrization of the circle x^2 + y^2 = 1:
    # x = (1-t^2)/(1+t^2), y = 2t/(1+t^2) for rational t
    # For x^2 + D*y^2 = R^2, scale appropriately.
    # 
    # Alternative: use the fact that if P is on the conic, then the line through
    # P with rational slope hits the conic at another rational point.
    #
    # Simpler: generate candidate points as (a, b) where a^2 + D*b^2 is a square.
    # This is equivalent to finding rational points on the cone.
    
    # Generate points where x^2 + D*y^2 = z^2 for some rational z
    # Parametrize: x = u*(1-v^2), y = 2*u*v/sqrt(D) ... but y must be rational.
    # For D=1: x = (1-t^2)*s, y = 2*t*s, z = (1+t^2)*s
    # For general D: x^2 + D*y^2 = z^2
    # Let x = D*(m^2 - n^2)*k, y = 2*m*n*k, z = D*(m^2 + n^2)*k  -- no, that's for x^2+y^2=z^2
    # For x^2 + D*y^2 = z^2: let x = (p^2 - D*q^2)*r, y = 2*p*q*r, z = (p^2 + D*q^2)*r
    
    candidates = []
    seen = set()
    
    for p in range(1, 30):
        for q in range(1, 30):
            if math.gcd(p, q) != 1:
                continue
            for r in range(1, 10):
                x = (p * p - D * q * q) * r
                y = 2 * p * q * r
                z = (p * p + D * q * q) * r
                # Verify: x^2 + D*y^2 = z^2
                if x * x + D * y * y == z * z:
                    pt = (Fraction(x), Fraction(y))
                    if pt not in seen:
                        seen.add(pt)
                        candidates.append(pt)
                    # Also try negative variants
                    pt2 = (Fraction(-x), Fraction(y))
                    if pt2 not in seen:
                        seen.add(pt2)
                        candidates.append(pt2)
                    pt3 = (Fraction(x), Fraction(-y))
                    if pt3 not in seen:
                        seen.add(pt3)
                        candidates.append(pt3)
                    pt4 = (Fraction(-x), Fraction(-y))
                    if pt4 not in seen:
                        seen.add(pt4)
                        candidates.append(pt4)
    
    if len(candidates) < target_n:
        return None
    
    # Now search for target_n points with all pairwise rational distances
    # and no 3 collinear, no 4 concyclic
    random.seed(D * 777)
    random.shuffle(candidates)
    
    # Limit candidates for tractability
    if len(candidates) > 200:
        candidates = candidates[:200]
    
    result_box = [None]
    node_count = [0]
    max_nodes = 100000
    
    def bt(current, idx):
        if time.time() > deadline or node_count[0] > max_nodes:
            return False
        if len(current) >= target_n:
            if _check_concyclic(current, D):
                result_box[0] = list(current)
                return True
            return False
        for i in range(idx, len(candidates)):
            node_count[0] += 1
            if time.time() > deadline or node_count[0] > max_nodes:
                return False
            cand = candidates[i]
            new_cur = current + [cand]
            if _check_partial(new_cur, D):
                if bt(new_cur, i + 1):
                    return True
        return False
    
    if bt([], 0):
        return result_box[0]
    return None


def _try_bend_transform(pts, D, deadline):
    """
    Apply a rational bend to break concyclicity while preserving rational distances.
    Idea: if points are on a conic, a rational Möbius-like transform can move them off.
    We try small rational perturbations.
    """
    if not pts or len(pts) < 3:
        return None
    
    n = len(pts)
    # Try small rational perturbations of y-coordinates
    # This is heuristic: perturb y by small rational amounts
    for dy_num in range(-3, 4):
        for dy_den in [1, 2, 3]:
            if dy_num == 0:
                continue
            dy = Fraction(dy_num, dy_den)
            new_pts = [(x, y + dy) for x, y in pts]
            if _check_valid(new_pts, D):
                return new_pts
    return None


def _greedy_extend(pts, D, candidates, deadline, max_size=12):
    """Greedy: try to add candidates one by one."""
    best = list(pts)
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in best:
            continue
        new_pts = best + [cand]
        if len(new_pts) <= max_size and _check_valid(new_pts, D):
            best = new_pts
    return best


def _local_search(pts, D, candidates, deadline, max_iter=5000):
    """Local search: add if possible, else try swaps."""
    best = list(pts)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        added = False
        for cand in candidates:
            if cand in best:
                continue
            new_pts = best + [cand]
            if _check_valid(new_pts, D):
                best = new_pts
                added = True
                break
        if not added:
            break
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    # Fallback: known 5-point set
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D_5 = 1
    fracs_5 = [(Fraction(x), Fraction(y)) for x, y in points_5]
    best_result = {'D': D_5, 'points': _to_output(fracs_5), 'n': 5}
    
    def remaining():
        return deadline - time.time()
    
    # Strategy 1: Integer grid backtracking for 6+ points
    for D in [1, 2, 3, 5, 7, 10, 13]:
        if remaining() < 15:
            break
        if not _is_squarefree(D):
            continue
        
        for limit in [4, 5, 6, 7]:
            if remaining() < 10:
                break
            if limit * limit > 80:
                continue
            max_nodes = 100000 if limit <= 5 else 50000
            result = _search_integer_grid(D, limit, 6, deadline, max_nodes=max_nodes)
            if result:
                best_result = {'D': D, 'points': _to_output(result), 'n': len(result)}
                # Try to extend
                if remaining() > 10:
                    candidates = [(Fraction(x), Fraction(y)) for x in range(-limit-2, limit+3)
                                  for y in range(-limit-2, limit+3)]
                    extended = _local_search(result, D, candidates, deadline, max_iter=3000)
                    if len(extended) > best_result['n']:
                        best_result = {'D': D, 'points': _to_output(extended), 'n': len(extended)}
                if best_result['n'] >= 6:
                    break
        
        if best_result['n'] >= 7:
            break
    
    # Strategy 2: Conic parametrization search
    if remaining() > 15:
        for D in [1, 2, 3, 5, 7]:
            if remaining() < 10:
                break
            if not _is_squarefree(D):
                continue
            result = _search_conic_param(D, 6, deadline)
            if result and len(result) > best_result['n']:
                best_result = {'D': D, 'points': _to_output(result), 'n': len(result)}
                # Try to extend
                if remaining() > 10:
                    # Generate more candidates from the same conic family
                    extra_candidates = []
                    for p in range(1, 50):
                        for q in range(1, 50):
                            if math.gcd(p, q) != 1:
                                continue
                            for r in range(1, 15):
                                x = (p * p - D * q * q) * r
                                y = 2 * p * q * r
                                pt = (Fraction(x), Fraction(y))
                                extra_candidates.append(pt)
                                extra_candidates.append((Fraction(-x), Fraction(y)))
                                extra_candidates.append((Fraction(x), Fraction(-y)))
                                extra_candidates.append((Fraction(-x), Fraction(-y)))
                    extended = _local_search(result, D, extra_candidates, deadline, max_iter=5000)
                    if len(extended) > best_result['n']:
                        best_result = {'D': D, 'points': _to_output(extended), 'n': len(extended)}
    
    # Strategy 3: Try known small configurations
    if remaining() > 5:
        known_configs = [
            (1, [(0,0), (3,0), (0,4), (4,3), (-3,4), (0,-5)]),
            (1, [(0,0), (5,0), (0,12), (12,5), (-5,12), (0,-13)]),
            (1, [(0,0), (7,0), (0,24), (24,7), (-7,24), (0,-25)]),
            (1, [(0,0), (9,0), (0,40), (40,9), (-9,40), (0,-41)]),
            (2, [(0,0), (1,0), (0,1), (1,1), (-1,1), (1,-1)]),
            (2, [(0,0), (2,0), (0,1), (2,1), (-2,1), (2,-1)]),
            (3, [(0,0), (1,0), (0,1), (1,1), (-1,1), (1,-1)]),
            (1, [(0,0), (1,0), (0,1), (1,1), (-1,0), (0,-1)]),
            (1, [(0,0), (4,0), (0,3), (4,3), (-4,3), (0,-5)]),
            (1, [(0,0), (5,0), (0,4), (5,4), (-5,4), (0,-3)]),
        ]
        for D, pts in known_configs:
            if not _is_squarefree(D):
                continue
            fracs = [(Fraction(x), Fraction(y)) for x, y in pts]
            if _check_valid(fracs, D) and len(fracs) > best_result['n']:
                best_result = {'D': D, 'points': _to_output(fracs), 'n': len(fracs)}
    
    # Strategy 4: Try to extend best found with bend + greedy
    if remaining() > 10 and best_result['n'] < 8:
        D_cur = best_result['D']
        pts_cur = [(Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1])) for p in best_result['points']]
        
        # Try bend
        bent = _try_bend_transform(pts_cur, D_cur, deadline)
        if bent and len(bent) > best_result['n']:
            best_result = {'D': D_cur, 'points': _to_output(bent), 'n': len(bent)}
        
        # Generate broader candidate set for extension
        limit_ext = 15
        candidates_ext = [(Fraction(x), Fraction(y)) for x in range(-limit_ext, limit_ext+1)
                          for y in range(-limit_ext, limit_ext+1)]
        extended = _local_search(pts_cur, D_cur, candidates_ext, deadline, max_iter=10000)
        if len(extended) > best_result['n']:
            best_result = {'D': D_cur, 'points': _to_output(extended), 'n': len(extended)}
    
    del best_result['n']
    return best_result

# EVOLVE-BLOCK-END
