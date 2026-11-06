"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
TOTAL = N * N

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE_MARGIN = 30.0
_START_TIME = None


def _init_time():
    global _START_TIME
    _START_TIME = time.time()


def _time_remaining():
    return 330.0 - (time.time() - _START_TIME) - _DEADLINE_MARGIN


def _precompute():
    """Precompute all isosceles relationships using bitsets for speed."""
    pts = [(x, y) for x in range(N) for y in range(N)]
    
    # For each point i, precompute which pairs (j,k) would make {i,j,k} isosceles
    # iso_conflicts[i] = bitmask of pairs? Too large.
    # Instead: for each pair (i,j), precompute set of k such that {i,j,k} is isosceles
    # pair_iso[i][j] = set of k
    
    # Use a 2D array of sets
    pair_iso = [[set() for _ in range(TOTAL)] for _ in range(TOTAL)]
    
    for i in range(TOTAL):
        xi, yi = pts[i]
        for j in range(i + 1, TOTAL):
            xj, yj = pts[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                xk, yk = pts[k]
                d1 = (xi - xk) ** 2 + (yi - yk) ** 2
                d3 = (xj - xk) ** 2 + (yj - yk) ** 2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    pair_iso[i][j].add(k)
                    pair_iso[j][i].add(k)
    
    return pts, pair_iso


def _can_add(chosen, p, pair_iso):
    """Check if p can be added to chosen without creating isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            ii, jj = chosen[i], chosen[j]
            if p in pair_iso[ii][jj]:
                return False
    return True


def _is_maximal(chosen, pair_iso):
    """Check if chosen is a maximal no-isosceles set."""
    chosen_set = set(chosen)
    for p in range(TOTAL):
        if p in chosen_set:
            continue
        # Check if p is blocked by some pair in chosen
        is_blocked = False
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                ii, jj = chosen[i], chosen[j]
                if p in pair_iso[ii][jj]:
                    is_blocked = True
                    break
            if is_blocked:
                break
        if not is_blocked:
            return False
    return True


def _greedy_construct(pair_iso, order=None):
    """Greedy construction with given order."""
    if order is None:
        order = list(range(TOTAL))
    chosen = []
    for p in order:
        if _can_add(chosen, p, pair_iso):
            chosen.append(p)
    return chosen


def _make_maximal(chosen, pair_iso):
    """Add points to make chosen maximal (greedily)."""
    chosen = list(chosen)
    changed = True
    while changed:
        changed = False
        for p in range(TOTAL):
            if p not in chosen and _can_add(chosen, p, pair_iso):
                chosen.append(p)
                changed = True
    return chosen


def _local_search(chosen, pair_iso, max_iters=200):
    """
    Local search: try to remove points while maintaining no-isosceles,
    then re-maximize. This can reduce size.
    """
    chosen = list(chosen)
    best = list(chosen)
    best_size = len(chosen)
    
    for _ in range(max_iters):
        if _time_remaining() < 15.0:
            break
        
        improved = False
        # Try removing each point
        for p in chosen[:]:
            if _time_remaining() < 15.0:
                return best
            
            # Try removing p
            new_chosen = [x for x in chosen if x != p]
            
            # No-isosceles is preserved after removal
            # Now make it maximal
            maximalized = _make_maximal(new_chosen, pair_iso)
            
            if len(maximalized) < best_size:
                best = maximalized
                best_size = len(maximalized)
                chosen = maximalized
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add another
            for p in chosen[:]:
                if _time_remaining() < 15.0:
                    return best
                
                new_chosen = [x for x in chosen if x != p]
                # Try adding a different point
                for q in range(TOTAL):
                    if q in new_chosen:
                        continue
                    if _can_add(new_chosen, q, pair_iso):
                        candidate = new_chosen + [q]
                        maximalized = _make_maximal(candidate, pair_iso)
                        if len(maximalized) < best_size:
                            best = maximalized
                            best_size = len(maximalized)
                            chosen = maximalized
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return best


def construct():
    _init_time()
    pts, pair_iso = _precompute()
    
    # Generate multiple initial solutions with different orders
    candidates = []
    
    # Natural order
    candidates.append(_greedy_construct(pair_iso))
    
    # Random orders
    random.seed(42)
    for seed in range(10):
        random.seed(seed)
        order = list(range(TOTAL))
        random.shuffle(order)
        candidates.append(_greedy_construct(pair_iso, order))
    
    # Make all maximal
    maximal_candidates = []
    for c in candidates:
        mc = _make_maximal(c, pair_iso)
        maximal_candidates.append(mc)
    
    # Find best initial
    best = min(maximal_candidates, key=len)
    best_size = len(best)
    
    # Local search from multiple starting points
    for i, start in enumerate(maximal_candidates):
        if _time_remaining() < 20.0:
            break
        
        result = _local_search(start, pair_iso, max_iters=100)
        if len(result) < best_size:
            best_size = len(result)
            best = result
    
    # Additional random restarts
    for _ in range(5):
        if _time_remaining() < 20.0:
            break
        
        # Random subset as seed
        random.shuffle(range(TOTAL))
        seed_set = list(random.sample(range(TOTAL), min(10, TOTAL)))
        # Make maximal
        mc = _make_maximal(seed_set, pair_iso)
        result = _local_search(mc, pair_iso, max_iters=50)
        if len(result) < best_size:
            best_size = len(result)
            best = result
    
    # Convert to (x, y) tuples
    return [(p // N, p % N) for p in best]
# EVOLVE-BLOCK-END
