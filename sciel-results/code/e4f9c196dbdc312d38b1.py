"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill


def _time_remaining():
    return 330.0 - (time.time() - _start_time) - _DEADLINE_MARGIN


def _is_iso(a, b, c):
    """Check if triangle with vertices a, b, c is isosceles (including degenerate)."""
    d_ab = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d_ac = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d_bc = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d_ab == d_ac or d_ab == d_bc or d_ac == d_bc


def _can_add(point, chosen):
    """Check if point can be added to chosen without creating an isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_iso(point, chosen[i], chosen[j]):
                return False
    return True


def _is_maximal(chosen, all_points):
    """Check if no further grid point can be added to chosen."""
    chosen_set = set(chosen)
    for p in all_points:
        if p not in chosen_set:
            if _can_add(p, chosen):
                return False
    return True


def _greedy_construct(order):
    """Greedy construction following a given point order."""
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _greedy_then_optimize(chosen, all_points, time_limit):
    """Try to remove points while maintaining maximality and no-isosceles property."""
    chosen = list(chosen)
    all_points_set = set(all_points)
    
    while _time_remaining() > 0:
        improved = False
        # Try removing each point and see if we can restore maximality
        for idx in range(len(chosen)):
            if _time_remaining() <= 0:
                break
            candidate = chosen[:idx] + chosen[idx+1:]
            if not candidate:
                continue
            
            # Check if candidate is still no-isosceles (it should be, since we're removing)
            # Check maximality
            if _is_maximal(candidate, all_points):
                chosen = candidate
                improved = True
                break
        
        if not improved:
            break
    
    return chosen


def _local_search(chosen, all_points, time_limit):
    """Local search: try swaps (remove one, add another) to reduce size."""
    chosen = list(chosen)
    all_points_set = set(all_points)
    
    while _time_remaining() > 0:
        improved = False
        for idx in range(len(chosen)):
            if _time_remaining() <= 0:
                break
            # Try removing chosen[idx]
            base = chosen[:idx] + chosen[idx+1:]
            for p in all_points:
                if p in base:
                    continue
                candidate = base + [p]
                # Check no-isosceles
                valid = True
                for i in range(len(candidate)):
                    for j in range(i+1, len(candidate)):
                        for k in range(j+1, len(candidate)):
                            if _is_iso(candidate[i], candidate[j], candidate[k]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid and _is_maximal(candidate, all_points):
                    chosen = candidate
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return chosen


def construct():
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Strategy 1: Greedy in different orders, then optimize
    best = None
    
    # Row-major greedy
    order1 = [(x, y) for y in range(N) for x in range(N)]
    chosen1 = _greedy_construct(order1)
    best = _greedy_then_optimize(chosen1, all_points, _time_remaining())
    
    if _time_remaining() <= 0:
        return best
    
    # Column-major greedy
    order2 = [(x, y) for x in range(N) for y in range(N)]
    chosen2 = _greedy_construct(order2)
    cand2 = _greedy_then_optimize(chosen2, all_points, _time_remaining())
    if len(cand2) < len(best):
        best = cand2
    
    if _time_remaining() <= 0:
        return best
    
    # Random orders
    random.seed(42)
    for _ in range(50):
        if _time_remaining() <= 0:
            break
        order = list(all_points)
        random.shuffle(order)
        chosen = _greedy_construct(order)
        cand = _greedy_then_optimize(chosen, all_points, _time_remaining())
        if len(cand) < len(best):
            best = cand
    
    if _time_remaining() <= 0:
        return best
    
    # Local search on best
    best = _local_search(best, all_points, _time_remaining())
    
    return best
# EVOLVE-BLOCK-END
