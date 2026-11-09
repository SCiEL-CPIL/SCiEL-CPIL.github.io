"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools

_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _isosceles(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(p, chosen):
    """Check if point p can be added to chosen without creating isosceles triangle."""
    for a, b in itertools.combinations(chosen, 2):
        if _isosceles(p, a, b):
            return False
    return True

def _is_maximal(chosen, all_points):
    """Check if chosen is maximal (no point outside can be added)."""
    chosen_set = set(chosen)
    for p in all_points:
        if p not in chosen_set and _can_add(p, chosen):
            return False
    return True

def _make_maximal(chosen, all_points, order):
    """Given a no-isosceles set, greedily add points (in given order) to make it maximal."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    for p in order:
        if p not in chosen_set:
            if _can_add(p, chosen):
                chosen.append(p)
                chosen_set.add(p)
    return chosen

def _greedy_random(all_points, seed):
    """Randomized greedy construction of a maximal no-isosceles set."""
    rng = random.Random(seed)
    points = list(all_points)
    rng.shuffle(points)
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    best = None
    best_size = float('inf')
    
    # Try multiple random greedy starts with local search
    for trial in range(200):
        if time.time() > _DEADLINE - 30:
            break
        
        seed = trial * 7919 + 13
        chosen = _greedy_random(all_points, seed)
        
        # Local search: try removing each point and see if we can get a smaller maximal set
        improved = True
        while improved and time.time() < _DEADLINE - 30:
            improved = False
            for i in range(len(chosen)):
                if time.time() > _DEADLINE - 30:
                    break
                # Try removing chosen[i]
                new_chosen = [chosen[j] for j in range(len(chosen)) if j != i]
                if len(new_chosen) >= best_size:
                    continue
                # Make it maximal again
                rng = random.Random(seed + i)
                order = list(all_points)
                rng.shuffle(order)
                made_maximal = _make_maximal(new_chosen, all_points, order)
                if len(made_maximal) < len(chosen):
                    chosen = made_maximal
                    improved = True
                    if len(chosen) < best_size:
                        best_size = len(chosen)
                        best = list(chosen)
                    break  # restart the for loop with new chosen
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = list(chosen)
    
    if best is None:
        # Fallback: simple greedy
        best = []
        for p in all_points:
            if _can_add(p, best):
                best.append(p)
    
    return best
# EVOLVE-BLOCK-END
