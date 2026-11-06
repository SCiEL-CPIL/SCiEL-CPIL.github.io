"""Integral point sets in general position — fresh design.
Strategy: precomputed 5-point base + fast local search to extend.
All checks use exact integer arithmetic (no floating point).
"""
import time
import random
import math
import sys

_DEADLINE = time.time() + 110.0  # safety margin under 120s
_MAX_COORD = 5000

# ---------- exact integer helpers ----------

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _sq_dist_is_square(p, q):
    return _is_square(_sq_dist(p, q))

def _collinear(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0

def _concyclic(p, q, r, s):
    # Determinant of 4x4 matrix; zero iff concyclic
    def det3(a, b, c):
        return a * (b * c[2] - c[1] * b[2]) - b * (a[2] * c[1] - c[0] * a[1]) + c * (a[1] * c[0] - b[0] * a[1])
    
    pts = [p, q, r, s]
    signs = [1, -1, 1, -1]
    total = 0
    for i in range(4):
        rows = [pts[j] for j in range(4) if j != i]
        a = (rows[0][0], rows[0][1], rows[0][0]*rows[0][0] + rows[0][1]*rows[0][1])
        b = (rows[1][0], rows[1][1], rows[1][0]*rows[1][0] + rows[1][1]*rows[1][1])
        c = (rows[2][0], rows[2][1], rows[2][0]*rows[2][0] + rows[2][1]*rows[2][1])
        d = (a[0]*(b[1]*c[2] - b[2]*c[1]) - a[1]*(b[0]*c[2] - b[2]*c[0]) + a[2]*(b[0]*c[1] - b[1]*c[0]))
        total += signs[i] * d
    return total == 0

def _check_pairwise_rational(pts):
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _sq_dist_is_square(pts[i], pts[j]):
                return False
    return True

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic(pts[i], pts[j], pts[k], pts[l]):
                        return False
    return True

def _is_valid_set(pts):
    if len(pts) < 3:
        return False
    if len(set(tuple(p) for p in pts)) != len(pts):
        return False
    if not _check_pairwise_rational(pts):
        return False
    if not _check_no_three_collinear(pts):
        return False
    if not _check_no_four_concyclic(pts):
        return False
    return True

# ---------- fast incremental checks ----------

def _can_add_point(pts, cand):
    """Check if cand can be added to pts maintaining all constraints."""
    if any(cand == p for p in pts):
        return False
    # rational distances to all existing
    for p in pts:
        if not _sq_dist_is_square(cand, p):
            return False
    # no three collinear: for each pair in pts, check with cand
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear(pts[i], pts[j], cand):
                return False
    # no four concyclic: for each triple in pts, check with cand
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic(pts[i], pts[j], pts[k], cand):
                    return False
    return True

def _can_replace(pts, idx, new_pt):
    """Check if replacing pts[idx] with new_pt maintains all constraints."""
    if any(new_pt == p for i, p in enumerate(pts) if i != idx):
        return False
    # rational distances to all other points
    for i, p in enumerate(pts):
        if i != idx and not _sq_dist_is_square(new_pt, p):
            return False
    # no three collinear
    others = [p for i, p in enumerate(pts) if i != idx]
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            if _collinear(others[i], others[j], new_pt):
                return False
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            for k in range(j + 1, len(others)):
                if _collinear(others[i], others[j], others[k]):
                    return False
    # no four concyclic
    all_pts = [new_pt] + others
    n = len(all_pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic(all_pts[i], all_pts[j], all_pts[k], all_pts[l]):
                        return False
    return True

# ---------- precomputed base sets ----------

_BASE_SETS = [
    [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
    [(0, 0), (3, 4), (0, 5), (5, 3), (4, 12)],
    [(0, 0), (7, 24), (15, 8), (8, 15), (24, 7)],
    [(0, 0), (5, 12), (12, 5), (9, 12), (12, 9)],
]

# Verify base sets at import time
_VALID_BASES = []
for bs in _BASE_SETS:
    if _is_valid_set(bs):
        _VALID_BASES.append(bs)

if not _VALID_BASES:
    _VALID_BASES = [[(0, 0), (3, 4), (0, 5)]]

# ---------- search ----------

def _generate_candidates(pts, rng, limit=5000):
    """Generate candidate lattice points near existing points with better coverage."""
    candidates = []
    seen = set()
    
    # Compute centroid and spread
    if pts:
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        spread = max(max(abs(p[0] - cx) for p in pts), max(abs(p[1] - cy) for p in pts)) + 50
    else:
        cx, cy, spread = 0, 0, 100
    
    # Add points in a grid around centroid with multiple densities
    for radius in [spread // 2, spread, spread * 2, spread * 4]:
        step = max(1, radius // 20)
        for dx in range(-radius, radius + 1, step):
            for dy in range(-radius, radius + 1, step):
                if dx == 0 and dy == 0:
                    continue
                cand = (cx + dx, cy + dy)
                if -_MAX_COORD <= cand[0] <= _MAX_COORD and -_MAX_COORD <= cand[1] <= _MAX_COORD:
                    if cand not in seen:
                        seen.add(cand)
                        candidates.append(cand)
    
    # Add points near each existing point (local refinement)
    for p in pts:
        for radius in [10, 30, 80]:
            for dx in range(-radius, radius + 1, max(1, radius // 8)):
                for dy in range(-radius, radius + 1, max(1, radius // 8)):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (p[0] + dx, p[1] + dy)
                    if -_MAX_COORD <= cand[0] <= _MAX_COORD and -_MAX_COORD <= cand[1] <= _MAX_COORD:
                        if cand not in seen:
                            seen.add(cand)
                            candidates.append(cand)
    
    # Add random candidates in a larger area
    for _ in range(1000):
        cand = (rng.randint(-_MAX_COORD, _MAX_COORD), rng.randint(-_MAX_COORD, _MAX_COORD))
        if cand not in seen:
            seen.add(cand)
            candidates.append(cand)
    
    rng.shuffle(candidates)
    return candidates[:limit]

def _search_greedy(pts, rng, time_limit):
    """Try to add points greedily with multiple restarts."""
    pts = list(pts)
    best = list(pts)
    best_len = len(pts)
    
    while time.time() < time_limit:
        improved = False
        candidates = _generate_candidates(pts, rng, limit=5000)
        for cand in candidates:
            if time.time() >= time_limit:
                break
            if _can_add_point(pts, cand):
                pts.append(cand)
                improved = True
                if len(pts) > best_len:
                    best = list(pts)
                    best_len = len(pts)
                break
        if not improved:
            # Try a few random perturbations before giving up
            for _ in range(50):
                if time.time() >= time_limit:
                    break
                idx = rng.randrange(len(pts))
                old = pts[idx]
                new_pt = (old[0] + rng.randint(-30, 30), old[1] + rng.randint(-30, 30))
                if -_MAX_COORD <= new_pt[0] <= _MAX_COORD and -_MAX_COORD <= new_pt[1] <= _MAX_COORD:
                    if _can_replace(pts, idx, new_pt):
                        pts[idx] = new_pt
                        improved = True
                        break
            if not improved:
                break
    return best if best_len >= len(pts) else pts

def _search_replace(pts, rng, time_limit):
    """Try to improve by replacing one point with a better one (may enable future additions)."""
    pts = list(pts)
    n = len(pts)
    iterations = 0
    while time.time() < time_limit and iterations < 500:
        iterations += 1
        idx = rng.randrange(n)
        old = pts[idx]
        found = False
        # Try local moves first
        for radius in [20, 50, 100]:
            for _ in range(100):
                if time.time() >= time_limit:
                    break
                dx = rng.randint(-radius, radius)
                dy = rng.randint(-radius, radius)
                new_pt = (old[0] + dx, old[1] + dy)
                if -_MAX_COORD <= new_pt[0] <= _MAX_COORD and -_MAX_COORD <= new_pt[1] <= _MAX_COORD:
                    if _can_replace(pts, idx, new_pt):
                        pts[idx] = new_pt
                        found = True
                        break
            if found:
                break
        # Try random global moves
        if not found:
            for _ in range(100):
                if time.time() >= time_limit:
                    break
                new_pt = (rng.randint(-_MAX_COORD, _MAX_COORD), rng.randint(-_MAX_COORD, _MAX_COORD))
                if _can_replace(pts, idx, new_pt):
                    pts[idx] = new_pt
                    found = True
                    break
    return pts

def _search_combined(pts, rng, time_limit):
    """Combined search: alternate between greedy addition and replacement."""
    pts = list(pts)
    best = list(pts)
    best_len = len(pts)
    no_improve_count = 0
    
    while time.time() < time_limit and no_improve_count < 100:
        candidates = _generate_candidates(pts, rng, limit=3000)
        added = False
        for cand in candidates:
            if time.time() >= time_limit:
                break
            if _can_add_point(pts, cand):
                pts.append(cand)
                added = True
                if len(pts) > best_len:
                    best = list(pts)
                    best_len = len(pts)
                no_improve_count = 0
                break
        if added:
            continue
        no_improve_count += 1
        n = len(pts)
        if n < 3:
            break
        idx = rng.randrange(n)
        old = pts[idx]
        replaced = False
        for _ in range(150):
            if time.time() >= time_limit:
                break
            dx = rng.randint(-80, 80)
            dy = rng.randint(-80, 80)
            new_pt = (old[0] + dx, old[1] + dy)
            if -_MAX_COORD <= new_pt[0] <= _MAX_COORD and -_MAX_COORD <= new_pt[1] <= _MAX_COORD:
                if _can_replace(pts, idx, new_pt):
                    pts[idx] = new_pt
                    replaced = True
                    no_improve_count = 0
                    break
        if not replaced:
            # Try a completely random point replacement
            for _ in range(50):
                if time.time() >= time_limit:
                    break
                new_pt = (rng.randint(-_MAX_COORD, _MAX_COORD), rng.randint(-_MAX_COORD, _MAX_COORD))
                if _can_replace(pts, idx, new_pt):
                    pts[idx] = new_pt
                    replaced = True
                    no_improve_count = 0
                    break
            if not replaced:
                break
    return best if best_len >= len(pts) else pts

def _try_base_set(base_pts, rng, time_limit):
    """Try to extend a base set using multiple search strategies."""
    best = list(base_pts)
    best_len = len(base_pts)
    
    # Phase 1: Greedy addition (more time)
    t1 = min(time.time() + 25, time_limit)
    pts1 = _search_greedy(base_pts, rng, t1)
    if len(pts1) > best_len:
        best = list(pts1)
        best_len = len(pts1)
    
    # Phase 2: Combined search (more time)
    if time.time() < time_limit - 15:
        pts2 = _search_combined(best, rng, time_limit - 10)
        if len(pts2) > best_len:
            best = list(pts2)
            best_len = len(pts2)
    
    # Phase 3: Replacement search
    if time.time() < time_limit - 5 and best_len < 12:
        pts4 = _search_replace(best, rng, time_limit - 3)
        if len(pts4) > best_len:
            best = list(pts4)
            best_len = len(pts4)
    
    # Phase 4: Final greedy push
    if time.time() < time_limit - 2 and best_len < 12:
        pts3 = _search_greedy(best, rng, time_limit - 1)
        if len(pts3) > best_len:
            best = list(pts3)
            best_len = len(pts3)
    
    return best

def construct():
    global _DEADLINE
    rng = random.Random(42)
    
    best_overall = None
    best_len = 0
    
    # Try each valid base set
    for i, base_pts in enumerate(_VALID_BASES):
        remaining_time = _DEADLINE - time.time()
        if remaining_time < 10:
            break
        
        # Allocate time proportionally, with more time for earlier bases
        time_per_base = min(remaining_time / (len(_VALID_BASES) - i), 50)
        base_time_limit = time.time() + time_per_base
        
        result = _try_base_set(base_pts, rng, base_time_limit)
        
        if len(result) > best_len:
            best_len = len(result)
            best_overall = list(result)
        
        # Early exit if we found a large set
        if best_len >= 10:
            break
    
    # If no valid set found, use the first base set
    if best_overall is None:
        best_overall = list(_VALID_BASES[0])
    
    # Final verification
    if not _is_valid_set(best_overall):
        best_overall = list(_VALID_BASES[0])
    
    # Format output
    result = {
        'D': 1,
        'points': [[[x, 1], [y, 1]] for x, y in best_overall]
    }
    return result
