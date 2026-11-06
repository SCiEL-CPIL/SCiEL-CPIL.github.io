"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import time
import random

def _is_sunflower_triple(a, b, c):
    """Check if three 4-element sets form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(candidate, chosen):
    """Check if adding candidate to chosen creates no sunflower triple."""
    for i, a in enumerate(chosen):
        for b in chosen[i+1:]:
            if _is_sunflower_triple(a, b, candidate):
                return False
    return True

def _greedy_build():
    """Greedy construction with random ordering."""
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    random.shuffle(all_sets)
    
    chosen = []
    for s in all_sets:
        if _can_add(s, chosen):
            chosen.append(s)
    return chosen

def _local_improve(chosen, deadline):
    """Local search: try to swap out one set for a better one."""
    chosen = list(chosen)
    chosen_sets = set(tuple(sorted(s)) for s in chosen)
    
    # Precompute all possible 4-sets
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    all_set_tuples = [tuple(sorted(s)) for s in all_sets]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(all_sets)
        
        for i in range(len(chosen)):
            if time.time() >= deadline:
                break
            removed = chosen[i]
            remaining = [chosen[j] for j in range(len(chosen)) if j != i]
            
            # Try to find a replacement
            for candidate in all_sets:
                if candidate in chosen_sets:
                    continue
                # Check if we can swap: remove 'removed', add 'candidate'
                # Need to check no sunflower in remaining + candidate
                ok = True
                for j, a in enumerate(remaining):
                    if _is_sunflower_triple(a, candidate, removed):
                        pass  # removed is gone, so this doesn't matter
                    for b in remaining[j+1:]:
                        if _is_sunflower_triple(a, b, candidate):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    # Also check pairs in remaining that might form sunflower with candidate
                    # Actually we already checked all pairs in remaining with candidate
                    chosen[i] = candidate
                    chosen_sets.discard(tuple(sorted(removed)))
                    chosen_sets.add(tuple(sorted(candidate)))
                    improved = True
                    break
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best = _greedy_build()
    best_size = len(best)
    
    # Multiple restarts with local improvement
    for restart in range(20):
        if time.time() >= deadline:
            break
        
        candidate = _greedy_build()
        candidate = _local_improve(candidate, deadline)
        
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    return [sorted(s) for s in best]
# EVOLVE-BLOCK-END
