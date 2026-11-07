"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 85.0

N = 12
M = 40
TOTAL_MASKS = 1 << N

# Precompute comparability using bitsets
_comp = [0] * TOTAL_MASKS
for a in range(TOTAL_MASKS):
    bits = 0
    for b in range(TOTAL_MASKS):
        if a != b and ((a & b) == a or (a & b) == b):
            bits |= (1 << b)
    _comp[a] = bits

def _is_comparable(a, b):
    return (_comp[a] >> b) & 1

def _score_family(family):
    count = 0
    m = len(family)
    for i in range(m):
        ai = family[i]
        for j in range(i + 1, m):
            if _is_comparable(ai, family[j]):
                count += 1
    return count

def _delta_score(family, i, new_mask):
    old_mask = family[i]
    if old_mask == new_mask:
        return 0
    m = len(family)
    delta = 0
    for j in range(m):
        if j == i:
            continue
        other = family[j]
        old_rel = 1 if _is_comparable(old_mask, other) else 0
        new_rel = 1 if _is_comparable(new_mask, other) else 0
        delta += new_rel - old_rel
    return delta

def _build_multi_chain_family():
    """Build initial family with multiple overlapping chains."""
    family = []
    used = set()
    
    # Chain 1: {0}, {0,1}, {0,1,2}, ..., {0,1,...,11}
    for i in range(1, N + 1):
        mask = (1 << i) - 1
        if mask not in used:
            family.append(mask)
            used.add(mask)
    
    # Chain 2: {11}, {11,10}, {11,10,9}, ..., sharing the top
    for i in range(1, N + 1):
        mask = ((1 << N) - 1) ^ ((1 << (N - i)) - 1)
        if mask not in used:
            family.append(mask)
            used.add(mask)
    
    # Chain 3: empty set and some mid-size sets
    if 0 not in used:
        family.append(0)
        used.add(0)
    
    # Fill remaining with greedy selection
    all_masks = list(range(TOTAL_MASKS))
    random.shuffle(all_masks)
    
    for mask in all_masks:
        if len(family) >= M:
            break
        if mask in used:
            continue
        family.append(mask)
        used.add(mask)
    
    return family[:M]

def _hill_climb(family, time_limit=20.0):
    """Hill climbing with delta computation."""
    current_family = family[:]
    current_score = _score_family(current_family)
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    used_set = set(current_family)
    deadline = time.time() + time_limit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(m):
            if time.time() > deadline:
                break
            old_mask = current_family[i]
            used_set.discard(old_mask)
            
            best_new_mask = None
            best_delta = 0
            
            for new_mask in all_masks:
                if new_mask in used_set:
                    continue
                delta = _delta_score(current_family, i, new_mask)
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_new_mask is not None:
                used_set.add(best_new_mask)
                current_family[i] = best_new_mask
                current_score += best_delta
                improved = True
            else:
                used_set.add(old_mask)
    
    return current_family, current_score

def _simulated_annealing(family, time_limit=15.0):
    """Simulated annealing to escape local optima."""
    current_family = family[:]
    current_score = _score_family(current_family)
    best_family = current_family[:]
    best_score = current_score
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    used_set = set(current_family)
    deadline = time.time() + time_limit
    
    T = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        i = random.randrange(m)
        old_mask = current_family[i]
        new_mask = random.choice(all_masks)
        
        if new_mask == old_mask or new_mask in used_set:
            continue
        
        used_set.discard(old_mask)
        used_set.add(new_mask)
        
        delta = _delta_score(current_family, i, new_mask)
        current_family[i] = new_mask
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_family = current_family[:]
        else:
            # Revert
            current_family[i] = old_mask
            used_set.discard(new_mask)
            used_set.add(old_mask)
        
        T *= cooling
    
    return best_family, best_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    best_family = None
    best_score = -1
    
    # Multiple restarts with different initial families
    for restart in range(5):
        if time.time() > _DEADLINE - 20.0:
            break
        
        if restart == 0:
            family = _build_multi_chain_family()
        else:
            # Perturb the best found so far
            family = best_family[:] if best_family else _build_multi_chain_family()
            # Random swaps
            for _ in range(10):
                i = random.randrange(M)
                new_mask = random.randrange(TOTAL_MASKS)
                # Ensure distinctness
                attempts = 0
                while new_mask in family and attempts < 100:
                    new_mask = random.randrange(TOTAL_MASKS)
                    attempts += 1
                if new_mask not in family:
                    family[i] = new_mask
        
        # Phase 1: Hill climbing
        family, score = _hill_climb(family, time_limit=10.0)
        
        # Phase 2: Simulated annealing if time permits
        if time.time() < _DEADLINE - 15.0:
            family2, score2 = _simulated_annealing(family, time_limit=8.0)
            if score2 > score:
                family, score = family2, score2
        
        # Phase 3: Final hill climb polish
        if time.time() < _DEADLINE - 5.0:
            family, score = _hill_climb(family, time_limit=5.0)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
    
    # Convert to list of lists of ints
    result = []
    for mask in best_family:
        subset = []
        for b in range(N):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
    
    # Ensure types are correct
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]
    
    return result

# EVOLVE-BLOCK-END
