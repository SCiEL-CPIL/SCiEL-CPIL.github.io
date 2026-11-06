"""Rewritten program using a randomized greedy heuristic with backtracking-lite."""
# EVOLVE-BLOCK-START
import random
import itertools

def _pairwise_intersections_equal(a, b, c):
    """Check if three distinct 4-sets have all three pairwise intersections equal."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _is_sunflower_free(chosen, new_set):
    """Check if adding new_set to chosen creates a sunflower (3 sets with equal pairwise intersections)."""
    # For each pair in chosen, check if (chosen[i], chosen[j], new_set) forms a sunflower
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        for j in range(i + 1, n):
            b = chosen[j]
            if _pairwise_intersections_equal(a, b, new_set):
                return False
    return True

def _greedy_construct(seed=None, max_iter=500):
    """Randomized greedy construction."""
    if seed is not None:
        random.seed(seed)
    
    all_4sets = list(itertools.combinations(range(1, 17), 4))
    random.shuffle(all_4sets)
    
    chosen = []
    for s in all_4sets:
        s_set = set(s)
        if _is_sunflower_free(chosen, s_set):
            chosen.append(s_set)
    
    return [sorted(s) for s in chosen]

def _local_improvement(initial_family, max_rounds=100, seed=None):
    """Try to improve the family by removing and re-adding sets."""
    if seed is not None:
        random.seed(seed)
    
    chosen = list(initial_family)
    chosen_sets = [set(s) for s in chosen]
    
    best_size = len(chosen_sets)
    best_family = list(chosen_sets)
    
    for round_num in range(max_rounds):
        improved = False
        # Try to add a random set that isn't already in the family
        all_4sets = list(itertools.combinations(range(1, 17), 4))
        random.shuffle(all_4sets)
        
        for s in all_4sets:
            s_set = set(s)
            if s_set in chosen_sets:
                continue
            if _is_sunflower_free(chosen_sets, s_set):
                chosen_sets.append(s_set)
                improved = True
                if len(chosen_sets) > best_size:
                    best_size = len(chosen_sets)
                    best_family = list(chosen_sets)
        
        if not improved:
            break
    
    return [sorted(s) for s in best_family]

def construct():
    """Construct a large family of 4-element subsets of [1,16] with no three sunflowers."""
    best_family = []
    best_size = 0
    
    # Try multiple random seeds to find a good family
    for seed in range(50):
        family = _greedy_construct(seed=seed)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
        
        # Try local improvement
        improved = _local_improvement(family, max_rounds=50, seed=seed + 1000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family = improved
    
    return best_family
# EVOLVE-BLOCK-END
