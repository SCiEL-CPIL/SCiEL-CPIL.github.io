"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 12
M = 40
FULL_MASK = (1 << N) - 1
ALL_MASKS = list(range(1 << N))

def _count_comparable_pairs_masks(masks):
    """Count unordered pairs where one mask is strict subset of the other."""
    m = len(masks)
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if mi == mj:
                continue
            if (mi & mj) == mi or (mi & mj) == mj:
                count += 1
    return count

def _masks_to_sets(masks):
    """Convert list of bitmasks to list of lists of ints."""
    result = []
    for mask in masks:
        s = []
        for bit in range(N):
            if mask & (1 << bit):
                s.append(bit)
        result.append(s)
    return result

def _count_pairs(family):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = family[i], family[j]
            if a < b or b < a:
                count += 1
    return count

def _construct():
    n = N
    m = M
    
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    best_score = -1
    best_family = None
    
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline - 5.0:
            break
        
        # Greedy construction using bitmasks
        family = [0, FULL_MASK]
        used = set(family)
        
        while len(family) < m:
            best_set = None
            best_gain = -1
            
            # Sample a subset of remaining to speed up
            candidates = [s for s in ALL_MASKS if s not in used]
            random.shuffle(candidates)
            sample = candidates[:min(300, len(candidates))]
            
            for s in sample:
                gain = 0
                for t in family:
                    if s != t and ((s & t) == s or (s & t) == t):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_set = s
            
            if best_set is not None:
                family.append(best_set)
                used.add(best_set)
            else:
                s = random.choice([x for x in ALL_MASKS if x not in used])
                family.append(s)
                used.add(s)
        
        score = _count_comparable_pairs_masks(family)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
        
        # Local search refinement with exhaustive best-replacement
        for iteration in range(200):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for i in range(m):
                if time.time() > deadline - 5.0:
                    break
                
                current_mask = family[i]
                best_candidate = None
                best_new_score = score
                
                # Try all possible replacements (exhaustive)
                for candidate in ALL_MASKS:
                    if candidate in used:
                        continue
                    
                    # Quick check: does this candidate improve anything?
                    new_family = family[:]
                    new_family[i] = candidate
                    new_score = _count_comparable_pairs_masks(new_family)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    used.discard(current_mask)
                    family[i] = best_candidate
                    used.add(best_candidate)
                    score = best_new_score
                    improved = True
                    break
            
            if not improved:
                break
        
        if score > best_score:
            best_score = score
            best_family = family[:]
    
    # Final polish: try to improve best_family further
    if best_family is not None:
        family = best_family[:]
        used = set(family)
        score = _count_comparable_pairs_masks(family)
        
        for iteration in range(500):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            for i in range(m):
                if time.time() > deadline - 2.0:
                    break
                
                current_mask = family[i]
                best_candidate = None
                best_new_score = score
                
                for candidate in ALL_MASKS:
                    if candidate in used:
                        continue
                    
                    new_family = family[:]
                    new_family[i] = candidate
                    new_score = _count_comparable_pairs_masks(new_family)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    used.discard(current_mask)
                    family[i] = best_candidate
                    used.add(best_candidate)
                    score = best_new_score
                    improved = True
                    break
            
            if not improved:
                break
        
        best_family = family
    
    return _masks_to_sets(best_family)

def construct():
    result = _construct()
    # Ensure all elements are Python ints and sequences are lists
    cleaned = []
    for s in result:
        cleaned.append([int(x) for x in s])
    return cleaned
# EVOLVE-BLOCK-END
