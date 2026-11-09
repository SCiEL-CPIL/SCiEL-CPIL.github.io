"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import sys

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _is_no_isosceles(pts):
    """Check if no three points form an isosceles triangle."""
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            for k in range(j+1, len(pts)):
                if _is_isosceles(pts[i], pts[j], pts[k]):
                    return False
    return True

def _is_maximal(pts, n):
    """Check if pts is a maximal no-isosceles set in [0,n)^2."""
    pt_set = set(pts)
    for x in range(n):
        for y in range(n):
            p = (x, y)
            if p in pt_set:
                continue
            # Check if p is blocked: exists a,b in pts such that {p,a,b} is isosceles
            blocked = False
            pts_list = list(pt_set)
            for i in range(len(pts_list)):
                for j in range(i+1, len(pts_list)):
                    if _is_isosceles(p, pts_list[i], pts_list[j]):
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return False
    return True

def _greedy_construct(n):
    """Greedy construction in row-major order."""
    chosen = []
    for p in itertools.product(range(n), repeat=2):
        if all(not _is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    return chosen

def _try_remove_point(pts, n):
    """Try to remove a point from pts if the result is still no-isosceles and maximal."""
    for idx in range(len(pts)):
        candidate = pts[:idx] + pts[idx+1:]
        if not _is_no_isosceles(candidate):
            continue
        if _is_maximal(candidate, n):
            return candidate
    return None

def _local_search(pts, n, max_iters=1000, seed=42):
    """Local search to minimize the set size."""
    rng = random.Random(seed)
    current = list(pts)
    best = list(current)
    
    for it in range(max_iters):
        # Try to remove a random point
        if len(current) > 1:
            idx = rng.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if _is_no_isosceles(candidate) and _is_maximal(candidate, n):
                current = candidate
                if len(current) < len(best):
                    best = list(current)
                continue
        
        # Try a swap: remove one point, add another
        if len(current) > 0:
            remove_idx = rng.randint(0, len(current) - 1)
            # Find a candidate point to add
            pt_set = set(current)
            candidates = [(x, y) for x in range(n) for y in range(n) if (x, y) not in pt_set]
            if candidates:
                add_pt = rng.choice(candidates)
                candidate = current[:remove_idx] + current[remove_idx+1:] + [add_pt]
                # Shuffle to avoid order bias
                rng.shuffle(candidate)
                if _is_no_isosceles(candidate) and _is_maximal(candidate, n):
                    if len(candidate) < len(current):
                        current = candidate
                        if len(current) < len(best):
                            best = list(current)
                    elif rng.random() < 0.3:
                        # Accept equal-size swaps with some probability to escape local optima
                        current = candidate
        
        # Occasionally try to remove multiple points
        if len(current) > 2 and rng.random() < 0.1:
            for _ in range(3):
                new_pts = _try_remove_point(current, n)
                if new_pts is not None:
                    current = new_pts
                    if len(current) < len(best):
                        best = list(current)
                else:
                    break
    
    return best

def _random_valid_set(n, seed=0):
    """Generate a random valid no-isosceles set."""
    rng = random.Random(seed)
    all_pts = [(x, y) for x in range(n) for y in range(n)]
    rng.shuffle(all_pts)
    chosen = []
    for p in all_pts:
        if all(not _is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    return chosen

def construct():
    n = N
    # Start with greedy
    greedy = _greedy_construct(n)
    
    # Try multiple random starts and local searches
    best = list(greedy)
    best_size = len(greedy)
    
    # Multiple restarts with different seeds
    for seed in range(50):
        if seed == 0:
            start = list(greedy)
        else:
            start = _random_valid_set(n, seed=seed)
        
        result = _local_search(start, n, max_iters=200, seed=seed)
        if len(result) < best_size:
            best = result
            best_size = len(result)
    
    # Additional pass: try to remove points from best
    for _ in range(100):
        new_best = _try_remove_point(best, n)
        if new_best is not None:
            best = new_best
        else:
            break
    
    return best
# EVOLVE-BLOCK-END
