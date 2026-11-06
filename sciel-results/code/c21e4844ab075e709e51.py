"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import itertools
from collections import defaultdict

def _dist2(a, b):
    """Squared distance between two points."""
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _is_isosceles(p, chosen):
    """Check if adding point p to chosen creates any isosceles triangle with two points in chosen."""
    if len(chosen) < 2:
        return False
    
    k = len(chosen)
    # Compute squared distances from p to each chosen point
    d2_from_p = np.array([_dist2(p, c) for c in chosen])
    
    # Check all pairs in chosen
    for i in range(k):
        for j in range(i + 1, k):
            dij2 = _dist2(chosen[i], chosen[j])
            # Three conditions for isosceles:
            # 1. p is apex: d(p,i) == d(p,j)
            if d2_from_p[i] == d2_from_p[j]:
                return True
            # 2. chosen[i] is apex: d(p,i) == d(i,j)
            if d2_from_p[i] == dij2:
                return True
            # 3. chosen[j] is apex: d(p,j) == d(i,j)
            if d2_from_p[j] == dij2:
                return True
    return False

def _can_add(p, chosen):
    """Check if point p can be added without creating an isosceles triangle."""
    return not _is_isosceles(p, chosen)

def _find_addable_points(chosen, all_points):
    """Find all points that can be added to the set."""
    addable = []
    for p in all_points:
        if p not in chosen_set:
            if _can_add(p, chosen):
                addable.append(p)
    return addable

def _is_maximal(chosen, all_points):
    """Check if the set is maximal - no point can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p not in chosen_set:
            if _can_add(p, chosen):
                return False
    return True

def _greedy_construct(order):
    """Greedy construction given a specific point ordering."""
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen

def _make_maximal(chosen, all_points, add_order=None):
    """Make the set maximal by adding points in the given order (or random order if None)."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    
    if add_order is None:
        add_order = [p for p in all_points if p not in chosen_set]
        random.shuffle(add_order)
    else:
        add_order = [p for p in add_order if p not in chosen_set]
    
    for p in add_order:
        if p not in chosen_set:
            if _can_add(p, chosen):
                chosen.append(p)
                chosen_set.add(p)
    
    return chosen

def _try_remove_and_minimize(chosen, all_points, remove_order=None):
    """Try to remove points and find a smaller maximal set."""
    chosen = list(chosen)
    best = list(chosen)
    best_size = len(chosen)
    
    if remove_order is None:
        remove_order = list(range(len(chosen)))
        random.shuffle(remove_order)
    
    # Try removing points in the given order
    for idx in remove_order:
        if idx >= len(chosen):
            continue
        
        # Temporarily remove this point
        removed = chosen[idx]
        new_chosen = chosen[:idx] + chosen[idx+1:]
        
        # Check if the set is still maximal
        if _is_maximal(new_chosen, all_points):
            # It's still maximal with fewer points - great!
            chosen = new_chosen
            if len(chosen) < best_size:
                best = list(chosen)
                best_size = len(chosen)
        else:
            # Not maximal, so we need to add back some points
            # Try to add back the minimum number of points to make it maximal
            # Strategy: try adding points one at a time in a smart order
            
            # First, find which points can be added
            addable = []
            new_chosen_set = set(new_chosen)
            for p in all_points:
                if p not in new_chosen_set:
                    if _can_add(p, new_chosen):
                        addable.append(p)
            
            if not addable:
                # Can't add anything back, so this removal doesn't help
                continue
            
            # Try to find a minimal subset of addable points that makes it maximal
            # Greedy: add points that "block" the most other addable points
            # For simplicity, just try adding back in a few different orders
            
            # Order 1: random
            trial = list(new_chosen)
            trial_set = set(trial)
            addable_copy = addable[:]
            random.shuffle(addable_copy)
            for p in addable_copy:
                if p not in trial_set:
                    if _can_add(p, trial):
                        trial.append(p)
                        trial_set.add(p)
            if len(trial) < best_size and _is_maximal(trial, all_points):
                best = list(trial)
                best_size = len(trial)
            
            # Order 2: priority by how many other addable points it conflicts with
            # (skip for now, keep it simple)
    
    return best, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 280 second deadline (safety margin)
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best_solution = None
    best_size = float('inf')
    
    # First, get a baseline with row-major greedy
    baseline = _greedy_construct(all_points)
    best_solution = baseline
    best_size = len(baseline)
    
    elapsed = time.time() - start_time
    if elapsed > 250:
        return best_solution
    
    random.seed(42)
    
    # Multiple restarts with different strategies
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create a random ordering for initial greedy
        order = all_points[:]
        random.shuffle(order)
        
        # Greedy construction
        current = _greedy_construct(order)
        
        # Try to minimize by removing points
        current, current_size = _try_remove_and_minimize(current, all_points)
        
        if current_size < best_size:
            best_solution = current
            best_size = current_size
        
        # Also try a different approach: start from baseline and try removals
        if restart % 3 == 0 and best_solution:
            trial = list(best_solution)
            trial, trial_size = _try_remove_and_minimize(trial, all_points)
            if trial_size < best_size:
                best_solution = trial
                best_size = trial_size
    
    return best_solution
# EVOLVE-BLOCK-END
