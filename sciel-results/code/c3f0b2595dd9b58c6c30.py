import random
import time
import itertools
import sys

N = 16
DEADLINE = time.time() + 290  # Safety margin before 300s kill

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(point, chosen):
    """Check if point can be added to chosen without creating isosceles triangle."""
    for a, b in itertools.combinations(chosen, 2):
        if _is_isosceles(point, a, b):
            return False
    return True

def _is_maximal(chosen):
    """Check if no grid point can be added to chosen."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _is_valid(chosen):
    """Check if chosen is isosceles-free."""
    for a, b, c in itertools.combinations(chosen, 3):
        if _is_isosceles(a, b, c):
            return False
    return True

def _get_addable_points(chosen):
    """Return list of points that can be added to chosen."""
    chosen_set = set(chosen)
    addable = []
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            addable.append(p)
    return addable

def _conflict_count(point, addable_points, chosen):
    """Count how many addable points conflict with 'point'."""
    # A point q conflicts with p if p and q cannot coexist in a valid set.
    # This is not simply _can_add(q, chosen + [p]) because of transitivity?
    # Actually, if we add p, we check if q can be added to chosen + [p].
    # If _can_add(q, chosen + [p]) is False, then p and q conflict (given chosen).
    count = 0
    for q in addable_points:
        if q == point:
            continue
        # Check if q can be added if p is already in
        # We need to check if (chosen, p, q) is valid.
        # Since q is addable to chosen, and p is addable to chosen,
        # the only new constraint is between p and q relative to chosen.
        # _can_add(q, chosen + [p]) checks all pairs in chosen+[p] with q.
        # Since q is compatible with chosen, we only need to check pairs involving p.
        # i.e., is _is_isosceles(q, p, a) for any a in chosen?
        # _can_add does this.
        if not _can_add(q, chosen + [point]):
            count += 1
    return count

def _minimally_complete(chosen, rng):
    """
    Greedily add points to 'chosen' until it is maximal.
    Heuristic: Pick the point that blocks the most other addable points.
    """
    current_set = list(chosen)
    current_set_set = set(current_set)
    
    while True:
        addable = _get_addable_points(current_set)
        if not addable:
            break
        
        # If only one option, take it
        if len(addable) == 1:
            current_set.append(addable[0])
            current_set_set.add(addable[0])
            continue
            
        # Heuristic: Pick point that conflicts with the most other addable points
        # This reduces the search space for future additions, potentially leading to a smaller set.
        best_point = None
        best_conflicts = -1
        
        # Limit evaluation for speed if addable is large
        # But N=16, max 256 points. Addable is usually small.
        for p in addable:
            c = _conflict_count(p, addable, current_set)
            if c > best_conflicts:
                best_conflicts = c
                best_point = p
        
        if best_point is None:
            best_point = rng.choice(addable)
            
        current_set.append(best_point)
        current_set_set.add(best_point)
        
    return current_set

def _greedy_random(seed_val):
    """Randomized greedy construction."""
    rng = random.Random(seed_val)
    points = list(itertools.product(range(N), repeat=2))
    rng.shuffle(points)
    
    chosen = []
    for p in points:
        if _can_add(p, chosen):
            chosen.append(p)
    
    return chosen

def construct():
    best_set = None
    best_size = float('inf')
    
    # Baseline: Deterministic greedy
    baseline = []
    for p in itertools.product(range(N), repeat=2):
        if _can_add(p, baseline):
            baseline.append(p)
    if _is_valid(baseline) and _is_maximal(baseline):
        best_set = baseline
        best_size = len(baseline)
        
    start_time = time.time()
    attempt = 0
    
    # Local Search with Remove-and-Complete
    while time.time() < DEADLINE:
        attempt += 1
        
        # Initialize current set from best found or random
        if best_set is not None and attempt % 10 != 0:
            current_set = list(best_set)
        else:
            current_set = _greedy_random(attempt)
            if not _is_valid(current_set):
                continue
            # Complete to maximal
            current_set = _minimally_complete(current_set, random)
            
        if not _is_valid(current_set):
            continue
            
        if not _is_maximal(current_set):
            current_set = _minimally_complete(current_set, random)
            if not _is_valid(current_set):
                continue

        current_size = len(current_set)
        
        if current_size < best_size:
            best_set = list(current_set)
            best_size = current_size
            print(f"New best: {best_size} at attempt {attempt}")
            if best_size <= 10:
                break
                
        # Local Search Step: Try removing a point and completing
        improved = False
        points_to_try = list(current_set)
        random.shuffle(points_to_try)
        
        for p in points_to_try:
            if time.time() > DEADLINE - 10:
                break
                
            # Remove p
            new_set = [x for x in current_set if x != p]
            
            # Minimally complete
            completed_set = _minimally_complete(new_set, random)
            
            if not _is_valid(completed_set):
                continue
                
            new_size = len(completed_set)
            
            if new_size < len(current_set):
                current_set = completed_set
                improved = True
                # If we improved, restart the loop over points? 
                # Or just continue to try more removals on the new set.
                # Restarting is safer to avoid using stale info.
                break
                
        if not improved and attempt % 5 == 0:
            # Perturbation: Remove 2 random points and complete
            if len(current_set) > 2:
                removals = random.sample(current_set, 2)
                new_set = [x for x in current_set if x not in removals]
                completed_set = _minimally_complete(new_set, random)
                if _is_valid(completed_set):
                    current_set = completed_set

    return best_set if best_set is not None else baseline
