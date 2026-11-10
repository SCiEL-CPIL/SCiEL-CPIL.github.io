"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from functools import lru_cache

N = 12
M = 40
DEADLINE = 85.0  # safety margin before 90s kill

def _mask_to_list(mask):
    return [i for i in range(N) if (mask >> i) & 1]

def _count_comparable(family):
    """Count unordered pairs where one is a strict subset of the other."""
    masks = []
    for s in family:
        m = 0
        for v in s:
            m |= (1 << v)
        masks.append(m)
    
    count = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if (masks[i] & masks[j]) == masks[i] and masks[i] != masks[j]:
                count += 1
            elif (masks[i] & masks[j]) == masks[j] and masks[i] != masks[j]:
                count += 1
    return count

def _evaluate(family):
    return _count_comparable(family)

def _random_family():
    """Generate a random family of 40 distinct subsets."""
    all_masks = list(range(1 << N))
    random.shuffle(all_masks)
    selected = all_masks[:M]
    return [_mask_to_list(m) for m in selected]

def _greedy_construct():
    """
    Greedy construction: start with empty family, repeatedly add the set
    that maximizes the marginal gain in comparable pairs.
    """
    family = []
    family_masks = []
    
    # Precompute subset relationships for efficiency
    # For a candidate mask c, gain = number of existing sets that are comparable with c
    # comparable if (c & s) == c (c subset of s) or (c & s) == s (s subset of c)
    
    all_masks = list(range(1 << N))
    random.shuffle(all_masks)
    
    # To make it deterministic and faster, use a structured approach
    # Start with small sets and grow
    
    # Actually, let's use a smarter greedy: 
    # 1. Start with the empty set
    # 2. At each step, add the mask that gives max marginal comparable pairs
    
    # This is O(M * 2^N) which is 40 * 4096 = 163840 checks, each O(M) = 40
    # Total: ~6.5M operations, should be fast
    
    used = set()
    family = []
    family_masks = []
    
    # Start with empty set (mask 0)
    family.append([])
    family_masks.append(0)
    used.add(0)
    
    for _ in range(M - 1):
        best_gain = -1
        best_mask = -1
        
        for m in all_masks:
            if m in used:
                continue
            # Compute gain
            gain = 0
            for fm in family_masks:
                if (m & fm) == m:  # m subset of fm
                    gain += 1
                elif (m & fm) == fm:  # fm subset of m
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = m
        
        if best_mask < 0:
            break
        
        used.add(best_mask)
        family.append(_mask_to_list(best_mask))
        family_masks.append(best_mask)
    
    return family

def _hill_climb(family, time_limit=5.0):
    """
    Local search: try swapping one set for another to improve the count.
    """
    start_time = time.time()
    family = list(family)
    
    # Convert to masks for speed
    masks = []
    for s in family:
        m = 0
        for v in s:
            m |= (1 << v)
        masks.append(m)
    
    current_score = _count_comparable(family)
    
    all_masks = list(range(1 << N))
    random.shuffle(all_masks)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        random.shuffle(all_masks)
        
        for i in range(M):
            for new_m in all_masks:
                if new_m == masks[i]:
                    continue
                # Check if new_m is already in the family (at another position)
                if new_m in masks:
                    continue
                
                # Try replacing masks[i] with new_m
                old_m = masks[i]
                masks[i] = new_m
                
                new_score = _count_comparable(family)
                
                if new_score > current_score:
                    current_score = new_score
                    # Update family list
                    family[i] = _mask_to_list(new_m)
                    improved = True
                else:
                    masks[i] = old_m
                
                if (time.time() - start_time) > time_limit:
                    break
            if (time.time() - start_time) > time_limit:
                break
    
    return family, current_score

def construct():
    start_time = time.time()
    
    # Try multiple strategies and keep the best
    best_family = None
    best_score = -1
    
    # Strategy 1: Greedy construction
    family1 = _greedy_construct()
    score1 = _evaluate(family1)
    if score1 > best_score:
        best_score = score1
        best_family = family1
    
    if time.time() - start_time > DEADLINE:
        return best_family
    
    # Strategy 2: Hill climb from greedy
    family2, score2 = _hill_climb(family1, time_limit=min(10.0, DEADLINE - (time.time() - start_time) - 5.0))
    if score2 > best_score:
        best_score = score2
        best_family = family2
    
    if time.time() - start_time > DEADLINE:
        return best_family
    
    # Strategy 3: Random restarts with hill climbing
    remaining = DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 2.0:
        n_restarts = int(remaining / 2.0)
        n_restarts = max(1, min(n_restarts, 20))
        
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 5.0:
                break
            
            # Random family
            fam = _random_family()
            fam, sc = _hill_climb(fam, time_limit=min(1.5, (DEADLINE - (time.time() - start_time) - 5.0) / max(1, n_restarts - r)))
            if sc > best_score:
                best_score = sc
                best_family = fam
    
    # Ensure valid return
    if best_family is None:
        best_family = _greedy_construct()
    
    # Final validation and type conversion
    result = []
    seen = set()
    for s in best_family:
        # Ensure it's a list of Python ints
        clean = [int(v) for v in s]
        key = tuple(sorted(clean))
        if key not in seen:
            seen.add(key)
            result.append(clean)
    
    # If we lost some due to duplicates (shouldn't happen), fill with random
    if len(result) < M:
        all_masks = list(range(1 << N))
        random.shuffle(all_masks)
        for m in all_masks:
            if len(result) >= M:
                break
            lst = _mask_to_list(m)
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                result.append(lst)
    
    return result[:M]

# EVOLVE-BLOCK-END
