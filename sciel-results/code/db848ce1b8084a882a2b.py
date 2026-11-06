import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three distinct 4-sets form a sunflower: all pairwise intersections equal."""
    ab = len(a & b)
    ac = len(a & c)
    bc = len(b & c)
    return ab == ac == bc

def _can_add(family, new_set):
    """Check if adding new_set to family creates a sunflower triple."""
    for i in range(len(family)):
        for j in range(i + 1, len(family)):
            if _is_sunflower_triple(family[i], family[j], new_set):
                return False
    return True

def _greedy_construct(deadline):
    """Greedy construction with proper sunflower check."""
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    random.shuffle(all_sets)
    family = []
    for s in all_sets:
        if time.time() > deadline:
            break
        if _can_add(family, s):
            family.append(s)
    return family

def _local_search(family, deadline):
    """Try to improve by swapping or adding sets."""
    family = list(family)
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    in_family = set(frozenset(s) for s in family)
    
    best = len(family)
    improvement_found = True
    
    while time.time() < deadline and improvement_found:
        improvement_found = False
        # Try adding random sets not in family
        candidates = [s for s in all_sets if frozenset(s) not in in_family]
        random.shuffle(candidates)
        for new_s in candidates:
            if time.time() > deadline:
                break
            if _can_add(family, new_s):
                family.append(new_s)
                in_family.add(frozenset(new_s))
                improvement_found = True
                best = len(family)
                break
        # Try swaps: remove one, add one
        if time.time() > deadline:
            break
        for i in range(len(family)):
            if time.time() > deadline:
                break
            saved = family[i]
            family.pop(i)
            in_family.discard(frozenset(saved))
            for new_s in candidates:
                if time.time() > deadline:
                    break
                if frozenset(new_s) not in in_family and _can_add(family, new_s):
                    family.append(new_s)
                    in_family.add(frozenset(new_s))
                    best = len(family)
                    improvement_found = True
                    break
            if not improvement_found:
                family.insert(i, saved)
                in_family.add(frozenset(saved))
            else:
                break
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    random.seed(42)
    
    # Initial greedy construction
    family = _greedy_construct(deadline)
    
    # Local search to improve
    if time.time() < deadline:
        family = _local_search(family, deadline)
    
    # Verify no sunflower triples
    for i in range(len(family)):
        for j in range(i + 1, len(family)):
            for k in range(j + 1, len(family)):
                if _is_sunflower_triple(family[i], family[j], family[k]):
                    # Remove the last one to fix
                    family.pop()
                    break
    
    return [sorted(s) for s in family]
