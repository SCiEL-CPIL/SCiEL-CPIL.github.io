"""Structural redesign: bitmask-based greedy with time-bounded local search."""
import random
import time
from itertools import combinations

def _all_4sets():
    """Precompute all C(16,4)=1820 four-element subsets as bitmasks."""
    out = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        out.append(mask)
    return out

def _sunflower_violation(a, b, c):
    """Check if three 4-sets form a sunflower (all pairwise intersections equal).
    Two 4-sets can intersect in 0,1,2,3,4 elements. If they share 4 they're identical,
    so for distinct sets intersection is in {0,1,2,3}.
    Sunflower: a&b == a&c == b&c."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _check_family(family_masks):
    """Verify no three distinct sets form a sunflower."""
    n = len(family_masks)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _sunflower_violation(family_masks[i], family_masks[j], family_masks[k]):
                    return False
    return True

def _greedy_construct(order, deadline):
    """Greedy: add sets in given order if they don't create a sunflower."""
    chosen = []
    conflict_count = {}  # mask -> count of conflicts with chosen
    
    for mask in order:
        if time.time() > deadline:
            break
        # Check against all pairs of chosen
        valid = True
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                if _sunflower_violation(chosen[i], chosen[j], mask):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            chosen.append(mask)
    return chosen

def _local_search(family, all_masks, deadline, max_rounds=50):
    """Try to improve family by removing one set and adding up to two new ones."""
    best = list(family)
    best_size = len(best)
    best_set = set(best)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        # Try removing one set
        for i in range(len(best)):
            if time.time() > deadline:
                break
            candidate = best[:i] + best[i+1:]
            cand_set = set(candidate)
            
            # Try adding one new set
            for mask in all_masks:
                if mask in cand_set:
                    continue
                valid = True
                for a in candidate:
                    for b in candidate:
                        if a is b:
                            continue
                        if _sunflower_violation(a, b, mask):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_family = candidate + [mask]
                    if len(new_family) > best_size:
                        best = new_family
                        best_size = len(best)
                        best_set = set(best)
                        improved = True
                        break
            if improved:
                break
        if not improved and rnd > 0:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin for 90s limit
    
    all_masks = _all_4sets()
    n_total = len(all_masks)
    
    best_family = []
    best_size = 0
    
    # Multiple greedy passes with different random orderings
    rng = random.Random(42)
    
    num_greedy = 20
    for attempt in range(num_greedy):
        if time.time() > deadline - 10.0:
            break
        
        # Different ordering strategies
        if attempt == 0:
            order = list(all_masks)
        elif attempt == 1:
            order = sorted(all_masks, key=lambda m: bin(m).count('1'))
        elif attempt == 2:
            order = sorted(all_masks, key=lambda m: -bin(m).count('1'))
        else:
            order = list(all_masks)
            rng.shuffle(order)
        
        family = _greedy_construct(order, deadline)
        size = len(family)
        
        if size > best_size:
            best_size = size
            best_family = family
            best_set = set(best_family)
        
        # Local search improvement
        if time.time() < deadline - 5.0:
            improved = _local_search(family, all_masks, deadline - 2.0, max_rounds=10)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = improved
                best_set = set(best_family)
    
    # Final local search on best family
    if time.time() < deadline - 2.0 and best_family:
        final = _local_search(best_family, all_masks, deadline - 0.5, max_rounds=20)
        if len(final) > best_size:
            best_size = len(final)
            best_family = final
    
    # Convert bitmasks to sorted lists
    result = []
    for mask in best_family:
        elems = [i+1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elems))
    
    return result
