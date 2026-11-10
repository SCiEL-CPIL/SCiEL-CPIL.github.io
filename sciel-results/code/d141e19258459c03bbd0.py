"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools


def _isosceles(a, b, c):
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1 == d2 or d1 == d3 or d2 == d3


def _greedy_random(points, rng):
    """Greedy construction with random point ordering."""
    order = list(points)
    rng.shuffle(order)
    chosen = []
    for p in order:
        # Check if p can be added without creating an isosceles triangle
        can_add = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if _isosceles(p, chosen[i], chosen[j]):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            chosen.append(p)
    return chosen


def _is_maximal(chosen, all_points):
    """Check if chosen is a maximal no-isosceles set."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        # Check if p can be added
        can_add = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if _isosceles(p, chosen[i], chosen[j]):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True


def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_size = float('inf')
    
    # Run multiple randomized trials
    for trial in range(500):
        rng = random.Random(trial)
        chosen = _greedy_random(all_points, rng)
        
        # Verify maximality (should be true by construction, but let's be safe)
        if not _is_maximal(chosen, all_points):
            # This shouldn't happen, but if it does, skip
            continue
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen
    
    # Also try the original deterministic greedy as a fallback
    chosen_det = []
    for p in all_points:
        can_add = True
        for i in range(len(chosen_det)):
            for j in range(i + 1, len(chosen_det)):
                if _isosceles(p, chosen_det[i], chosen_det[j]):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            chosen_det.append(p)
    
    if len(chosen_det) < best_size:
        best = chosen_det
        best_size = len(chosen_det)
    
    if best is None:
        best = chosen_det
    
    return best
# EVOLVE-BLOCK-END
