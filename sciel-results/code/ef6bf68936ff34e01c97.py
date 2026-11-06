"""Rewritten program: smarter search for 6+ points in the (x, y*sqrt(D)) plane."""
from fractions import Fraction
import math
import time
import random

# EVOLVE-BLOCK-START

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _dist2_rational(pt1, pt2, D):
    """Return the squared distance as a Fraction, or None if not a perfect square rational."""
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    d2 = dx * dx + D * dy * dy
    num = d2.numerator
    den = d2.denominator
    if _is_perfect_square(num) and _is_perfect_square(den):
        return d2
    return None


def _check_pairwise(pts, D):
    """Check all pairwise distances are rational. Returns True if valid."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if _dist2_rational(pts[i], pts[j], D) is None:
                return False
    return True


def _check_collinear(pts, D):
    """Check no three points are collinear."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) == \
                   (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]):
                    return False
    return True


def _check_concyclic(pts, D):
    """Check no four points are concyclic (on conic x^2+D y^2+Ax+By+C=0)."""
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
    """Full validity check."""
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
    if not _check_collinear(pts, D):
        return False
    if n >= 4 and not _check_concyclic(pts, D):
        return False
    return True


def _generate_candidates(D, limit):
    """Generate candidate points (x, y) with rational coords in [-limit, limit]."""
    pts = []
    for x in range(-limit, limit + 1):
        for y in range(-limit, limit + 1):
            pts.append((Fraction(x), Fraction(y)))
    return pts


def _search_fixed_size(D, target_n, limit, max_backtracks=200000):
    """
    Search for target_n points from candidates in [-limit,limit]^2 for given D.
    Uses backtracking with smart ordering.
    """
    candidates = _generate_candidates(D, limit)
    
    # Pre-filter: keep only points that are distinct (all are) 
    # We'll do greedy + backtracking.
    
    best = None
    count = [0]
    
    def check_partial(pts):
        """Quick check for a partial set: pairwise distances + collinearity."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if _dist2_rational(pts[i], pts[j], D) is None:
                    return False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) == \
                       (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]):
                        return False
        return True
    
    def backtrack(current, idx):
        if count[0] > max_backtracks:
            return False
        if len(current) >= target_n:
            # Full check including concyclic
            if _check_concyclic(current, D):
                nonlocal_best = current
                return True
            return False
        
        if candidates_idx - idx > target_n - len(current):
            pass  # enough candidates left
        
        for i in range(idx, len(candidates)):
            count[0] += 1
            if count[0] > max_backtracks:
                return False
            
            cand = candidates[i]
            new_current = current + [cand]
            if check_partial(new_current):
                if len(current) == target_n - 1:
                    if _check_concyclic(new_current, D):
                        nonlocal_best = new_current
                        return True
                else:
                    if backtrack(new_current, i + 1):
                        return True
        return False
    
    # We need nonlocal, so use a list
    result_box = [None]
    
    def backtrack2(current, idx):
        if count[0] > max_backtracks:
            return False
        if len(current) >= target_n:
            if _check_concyclic(current, D):
                result_box[0] = list(current)
                return True
            return False
        
        for i in range(idx, len(candidates)):
            count[0] += 1
            if count[0] > max_backtracks:
                return False
            cand = candidates[i]
            new_current = current + [cand]
            if check_partial(new_current):
                if backtrack2(new_current, i + 1):
                    return True
        return False
    
    if backtrack2([], 0):
        return result_box[0]
    return None


def _greedy_build(D, candidates, max_size=8):
    """
    Greedy construction: iteratively add the best candidate.
    Returns a valid set.
    """
    pts = []
    for cand in candidates:
        new_pts = pts + [cand]
        if _check_valid(new_pts, D):
            pts = new_pts
            if len(pts) >= max_size:
                break
    return pts


def _find_good_start(D, candidates):
    """Find a good starting triple that has many compatible extensions."""
    best_triple = None
    best_score = -1
    
    # Sample some triples
    sampled = candidates[:]
    random.seed(42)
    if len(sampled) > 200:
        random.shuffle(sampled)
        sampled = sampled[:200]
    
    n = len(sampled)
    triples_checked = 0
    for i in range(n):
        for j in range(i + 1, n):
            triples_checked += 1
            if triples_checked > 20000:
                break
            for k in range(j + 1, n):
                if (pts_valid := _is_valid_triple(sampled[i], sampled[j], sampled[k], D)):
                    # Score: how many other candidates are compatible?
                    comp = _count_compatible(sampled, [sampled[i], sampled[j], sampled[k]], D)
                    if comp > best_score:
                        best_score = comp
                        best_triple = [sampled[i], sampled[j], sampled[k]]
        if triples_checked > 20000:
            break
    
    return best_triple


def _is_valid_triple(p1, p2, p3, D):
    """Check three points form a valid partial set."""
    if _dist2_rational(p1, p2, D) is None:
        return False
    if _dist2_rational(p2, p3, D) is None:
        return False
    if _dist2_rational(p1, p3, D) is None:
        return False
    # Collinearity
    if (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1]):
        return False
    return True


def _count_compatible(candidates, pts, D):
    """Count how many candidates can extend the given set (pairwise + collinearity)."""
    count = 0
    for cand in candidates:
        new_pts = pts + [cand]
        if check_partial(new_pts):
            count += 1
    return count


def _local_search(pts, D, candidates, max_iter=5000):
    """
    Local search: try to extend the set by adding one point, and if stuck,
    try replacing the worst point with a better one.
    """
    best = list(pts)
    for iteration in range(max_iter):
        # Try to add any compatible candidate
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
    deadline = start_time + 110.0  # 10s safety margin before 150s kill
    
    def remaining():
        return deadline - time.time()
    
    # Fallback: known 5-point set with D=1
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D_5 = 1
    fracs_5 = [(Fraction(x), Fraction(y)) for x, y in points_5]
    fallback_5 = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in fracs_5]
    
    best_result = {'D': D_5, 'points': fallback_5, 'n': 5}
    
    # Strategy:
    # 1. Try known small configurations that are pre-verified.
    # 2. Greedy construction with different D and candidate orderings.
    # 3. Backtracking search for 6 points with limited search space.
    # 4. Local search to extend.
    
    # Pre-verified 5-point set with D=2
    # Try to find better 6-point sets
    
    for D in [1, 2, 3, 5, 7]:
        if time.time() > deadline - 10:
            break
        
        if not _is_squarefree(D):
            continue
        
        # Greedy with different orderings
        for limit in [4, 6, 8, 10, 12]:
            if time.time() > deadline - 10:
                break
            if limit * limit > 400:  # too many candidates for reasonable greedy
                continue
                
            candidates = _generate_candidates(D, limit)
            
            # Try different random orderings
            for trial in range(3):
                if time.time() > deadline - 10:
                    break
                random.seed(D * 100 + limit * 10 + trial)
                shuffled = candidates[:]
                random.shuffle(shuffled)
                
                pts = _greedy_build(D, shuffled, max_size=12)
                if len(pts) > best_result['n']:
                    pts_out = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts]
                    best_result = {'D': D, 'points': pts_out, 'n': len(pts)}
                    
                    # Local search to try to extend further
                    if time.time() < deadline - 15:
                        extended = _local_search(pts, D, candidates, max_iter=2000)
                        if len(extended) > best_result['n']:
                            pts_out2 = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in extended]
                            best_result = {'D': D, 'points': pts_out2, 'n': len(extended)}
        
        # Backtracking search for 6 points with small limits
        for limit in [3, 4, 5]:
            if time.time() > deadline - 10:
                break
            if limit * limit > 100:
                continue
            candidates = _generate_candidates(D, limit)
            max_bt = 50000 if limit <= 4 else 100000
            
            result = _search_fixed_size(D, 6, limit, max_backtracks=max_bt)
            if result:
                pts_out = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in result]
                best_result = {'D': D, 'points': pts_out, 'n': 6}
                
                # Try to extend further with local search
                if time.time() < deadline - 20:
                    extended = _local_search(result, D, candidates, max_iter=5000)
                    if len(extended) > best_result['n']:
                        pts_out2 = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in extended]
                        best_result = {'D': D, 'points': pts_out2, 'n': len(extended)}
                
                if best_result['n'] >= 6:
                    break
        
        if best_result['n'] >= 6:
            # Try to push further
            if time.time() < deadline - 30:
                # Local search on the best found so far
                D_cur = best_result['D']
                pts_cur = [ (Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1])) for p in best_result['points'] ]
                limit_cur = 15
                candidates = _generate_candidates(D_cur, limit_cur)
                extended = _local_search(pts_cur, D_cur, candidates, max_iter=10000)
                if len(extended) > best_result['n']:
                    pts_out = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in extended]
                    best_result = {'D': D, 'points': pts_out, 'n': len(extended)}
    
    # Final: also try a few hand-crafted configurations
    # These are known or easy to verify
    
    # Try specific 6-point patterns
    known_patterns = [
        # 6 points forming a regular-ish hexagon, D=1
        ((0,0), (1,0), (0,1), (1,1), (-1,0), (0,-1)),
        # Square + 2 points
        ((0,0), (1,0), (0,1), (1,1), (2,0), (0,2)),
        # Try with D=2
    ]
    
    for pattern in known_patterns:
        if time.time() > deadline - 5:
            break
        for D in [1, 2]:
            pts = [(Fraction(x), Fraction(y)) for x, y in pattern]
            if _check_valid(pts, D):
                pts_out = [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts]
                if len(pts) > best_result['n']:
                    best_result = {'D': D, 'points': pts_out, 'n': len(pts)}
    
    # Return best found
    del best_result['n']
    return best_result


def _is_squarefree(n):
    if n <= 1:
        return True
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            return False
        p += 1 if p == 2 else 2
    return True

# EVOLVE-BLOCK-END
