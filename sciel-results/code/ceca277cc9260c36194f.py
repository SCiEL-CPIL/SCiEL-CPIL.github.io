"""Rewritten program: Minimum maximal no-isosceles set via randomized greedy + local pruning."""
import time
import random
from itertools import combinations

N = 16
DEADLINE = 320.0  # safety margin before 330s kill


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _isosceles(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _is_valid_addition(p, chosen):
    """Check if point p can be added to chosen without creating an isosceles triangle."""
    for a, b in combinations(chosen, 2):
        if _isosceles(p, a, b):
            return False
    return True


def _precompute_all_points():
    return [(x, y) for x in range(N) for y in range(N)]


def _greedy_construct(order, chosen_set):
    """Greedy: add points in given order if they don't create isosceles triangles."""
    chosen = list(chosen_set)
    for p in order:
        if p in chosen_set:
            continue
        if _is_valid_addition(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen


def _local_prune(chosen, deadline):
    """
    Try to remove points that are 'redundant' for maximality.
    A point is redundant if removing it doesn't allow any new point to be added.
    This is expensive, so we do a heuristic version:
    For each point, check if it's the only point preventing some other point from being added.
    """
    chosen_set = set(chosen)
    all_points = _precompute_all_points()
    
    # For each candidate point not in chosen, find which chosen points block it
    # If a chosen point is the sole blocker for many candidates, it might be essential
    # If a chosen point is the sole blocker for few/no candidates, it might be removable
    
    # Heuristic: try removing each point and see if the set is still maximal
    # This is O(n^3) per removal attempt, so limit attempts
    improved = True
    while improved:
        improved = False
        for i in range(len(chosen)):
            if time.time() > deadline:
                return chosen
            # Try removing chosen[i]
            trial = chosen[:i] + chosen[i+1:]
            trial_set = set(trial)
            
            # Check if trial is still maximal: can we add any point?
            is_maximal = True
            for p in all_points:
                if p in trial_set:
                    continue
                if _is_valid_addition(p, trial):
                    is_maximal = False
                    break
            
            if is_maximal:
                # We can remove this point and still have a maximal set
                chosen = trial
                chosen_set = trial_set
                improved = True
                break  # restart the loop since indices changed
    
    return chosen


def _randomized_greedy(seed, deadline):
    """
    Randomized greedy with maximum coverage heuristic:
    1. Start with empty set
    2. At each step, among all addable points, pick one that blocks the most other points
       (i.e., creates the most conflicts with remaining candidates)
    3. Use randomization to escape local optima
    """
    random.seed(seed)
    all_points = _precompute_all_points()
    chosen = []
    chosen_set = set()
    
    # Precompute distance pairs for speed
    # For each pair of points, store the distance squared
    # Actually, let's just do it on the fly but cache the valid additions
    
    while time.time() < deadline:
        # Find all points that can be added
        addable = []
        for p in all_points:
            if p in chosen_set:
                continue
            if _is_valid_addition(p, chosen):
                addable.append(p)
        
        if not addable:
            break
        
        # Score each addable point by how many other points it would block
        # A point p blocks q if adding p makes q not addable (due to isosceles with some pair)
        # Approximate: count how many other addable points form isosceles with p and some existing point
        scores = []
        for p in addable:
            # Count conflicts: for each other addable point q, check if p and q can't both be added
            # This is expensive, so use a simpler heuristic
            # Count how many existing points are at the same distance from p
            conflict_count = 0
            for q in addable:
                if q == p:
                    continue
                # Check if p and q are mutually exclusive
                # They are exclusive if there exists a pair (a,b) in chosen such that
                # _isosceles(p,a,b) or _isosceles(q,a,b) is true
                # But both are already addable, so neither creates isosceles with any pair
                # So they're only mutually exclusive if adding both together creates an isosceles
                # i.e., _isosceles(p, q, a) for some a in chosen
                for a in chosen:
                    if _isosceles(p, q, a):
                        conflict_count += 1
                        break
            scores.append(conflict_count)
        
        # Pick the point with highest score (most blocking), with some randomization
        max_score = max(scores)
        best_candidates = [p for p, s in zip(addable, scores) if s == max_score]
        
        # Add randomization: pick randomly among top performers
        p = random.choice(best_candidates)
        chosen.append(p)
        chosen_set.add(p)
    
    return chosen


def _hybrid_construct(seed, deadline):
    """
    Hybrid approach:
    1. Use randomized greedy to build a set
    2. Apply local pruning to remove redundant points
    3. Try to add back points in a different order to potentially get a smaller set
    """
    # Phase 1: Randomized greedy
    chosen = _randomized_greedy(seed, deadline)
    
    if time.time() > deadline:
        return chosen
    
    # Phase 2: Local pruning
    chosen = _local_prune(chosen, deadline)
    
    return chosen


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best = None
    best_size = float('inf')
    
    # Try multiple random seeds
    seed = 0
    while time.time() < deadline - 5:  # leave 5s margin
        result = _hybrid_construct(seed, deadline)
        if len(result) < best_size:
            best_size = len(result)
            best = result
        
        seed += 1
    
    if best is None:
        # Fallback: simple greedy
        all_points = _precompute_all_points()
        chosen = []
        chosen_set = set()
        for p in all_points:
            if _is_valid_addition(p, chosen):
                chosen.append(p)
                chosen_set.add(p)
        best = chosen
    
    return best
