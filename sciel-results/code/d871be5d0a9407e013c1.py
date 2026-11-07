"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

_DEADLINE = time.time() + 82.0

N = 12
M = 40
TOTAL_MASKS = 1 << N

def _is_strict_subset(a, b):
    """Check if a is a strict subset of b using bit operations."""
    return a != b and (a & b) == a

def _comparable(a, b):
    """Check if a and b are comparable (one is strict subset of the other)."""
    if a == b:
        return False
    if (a & b) == a:
        return True
    if (a & b) == b:
        return True
    return False

def _score_family(family):
    """Count unordered pairs (i, j) with i < j such that one is a strict subset of the other."""
    count = 0
    m = len(family)
    for i in range(m):
        ai = family[i]
        for j in range(i + 1, m):
            aj = family[j]
            if _comparable(ai, aj):
                count += 1
    return count

def _greedy_build():
    """Build family greedily: at each step, add the mask that maximizes new comparable pairs."""
    family = []
    used = set()
    
    # Start with empty set and full set as anchors
    family.append(0)
    used.add(0)
    family.append(TOTAL_MASKS - 1)
    used.add(TOTAL_MASKS - 1)
    
    all_masks = list(range(TOTAL_MASKS))
    
    while len(family) < M:
        best_mask = -1
        best_gain = -1
        for mask in all_masks:
            if mask in used:
                continue
            gain = 0
            for f in family:
                if _comparable(mask, f):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
                if gain == len(family):
                    break  # Can't do better than this
        if best_mask >= 0:
            family.append(best_mask)
            used.add(best_mask)
        else:
            break
    
    return family

def _local_search(family, max_time=None):
    """Local search: try replacing one element with another to improve score."""
    if max_time is None:
        max_time = _DEADLINE - 3.0
    
    current_family = family[:]
    current_score = _score_family(current_family)
    best_family = current_family[:]
    best_score = current_score
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    random.shuffle(all_masks)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < max_time:
        i = random.randrange(m)
        new_mask = random.choice(all_masks)
        if new_mask == current_family[i]:
            continue
        # Check distinctness
        is_dup = False
        for k in range(m):
            if k != i and current_family[k] == new_mask:
                is_dup = True
                break
        if is_dup:
            continue
        
        old_mask = current_family[i]
        # Compute delta
        delta = 0
        for j in range(m):
            if j == i:
                continue
            other = current_family[j]
            old_rel = _comparable(old_mask, other)
            new_rel = _comparable(new_mask, other)
            if new_rel and not old_rel:
                delta += 1
            elif old_rel and not new_rel:
                delta -= 1
        
        if delta > 0:
            current_family[i] = new_mask
            current_score += delta
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_family = current_family[:]
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_family, best_score

def _systematic_hill_climb(family, max_time=None):
    """Systematic hill climbing: try all possible replacements for each position."""
    if max_time is None:
        max_time = _DEADLINE - 3.0
    
    current_family = family[:]
    current_score = _score_family(current_family)
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    
    improved = True
    while improved and time.time() < max_time:
        improved = False
        for i in range(m):
            if time.time() > max_time:
                break
            old_mask = current_family[i]
            for new_mask in all_masks:
                if new_mask == old_mask:
                    continue
                # Check distinctness
                is_dup = False
                for k in range(m):
                    if k != i and current_family[k] == new_mask:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                delta = 0
                for j in range(m):
                    if j == i:
                        continue
                    other = current_family[j]
                    old_rel = _comparable(old_mask, other)
                    new_rel = _comparable(new_mask, other)
                    if new_rel and not old_rel:
                        delta += 1
                    elif old_rel and not new_rel:
                        delta -= 1
                
                if delta > 0:
                    current_family[i] = new_mask
                    current_score += delta
                    improved = True
                    break
    
    return current_family, current_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    # Phase 1: Greedy construction
    family = _greedy_build()
    score = _score_family(family)
    
    best_family = family[:]
    best_score = score
    
    # Phase 2: Local search with multiple restarts
    remaining_time = _DEADLINE - time.time()
    num_restarts = max(1, int(remaining_time / 5.0))
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Perturb the best family
        perturbed = best_family[:]
        for _ in range(random.randint(2, 5)):
            idx = random.randrange(M)
            new_mask = random.randrange(TOTAL_MASKS)
            # Ensure distinctness
            attempts = 0
            while new_mask in perturbed and attempts < 20:
                new_mask = random.randrange(TOTAL_MASKS)
                attempts += 1
            if new_mask not in perturbed:
                perturbed[idx] = new_mask
        
        # Run local search on perturbed family
        t_end = min(_DEADLINE - 5.0, time.time() + max(1.0, (remaining_time / num_restarts) * 0.8))
        f2, s2 = _local_search(perturbed, max_time=t_end)
        if s2 > best_score:
            best_score = s2
            best_family = f2[:]
    
    # Phase 3: Final systematic hill climb if time permits
    if time.time() < _DEADLINE - 5.0:
        f3, s3 = _systematic_hill_climb(best_family, max_time=_DEADLINE - 2.0)
        if s3 > best_score:
            best_score = s3
            best_family = f3
    
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
