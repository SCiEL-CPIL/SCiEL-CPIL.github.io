"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(masks):
    """
    Count the number of unordered pairs (i, j) with i < j such that
    masks[i] is a strict subset of masks[j] or masks[j] is a strict subset of masks[i].
    """
    m = len(masks)
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            # mi is strict subset of mj: (mi & mj) == mi and mi != mj
            # mj is strict subset of mi: (mi & mj) == mj and mi != mj
            if mi == mj:
                continue
            if (mi & mj) == mi:
                count += 1
            elif (mi & mj) == mj:
                count += 1
    return count

def _masks_to_lists(masks):
    """Convert list of masks to list of lists of integers."""
    result = []
    for mask in masks:
        lst = [b for b in range(P['n']) if (mask >> b) & 1]
        result.append(lst)
    return result

def _lists_to_masks(lists):
    """Convert list of lists to list of masks."""
    result = []
    for lst in lists:
        mask = 0
        for v in lst:
            if 0 <= v < P['n']:
                mask |= (1 << v)
        result.append(mask)
    return result

def _initial_solution():
    """
    Generate an initial solution using a chain-based approach.
    We want 40 distinct subsets of {0,...,11}.
    A good starting point is to create several chains.
    For example, 10 chains of length 4:
    Chain k: {k}, {k, k+1}, {k, k+1, k+2}, {k, k+1, k+2, k+3} mod 12?
    Let's just use random initialization with some structure.
    """
    random.seed(42)
    masks = []
    # Try to build a few chains
    # Chain 1: sizes 1,2,3,4
    # Chain 2: sizes 1,2,3,4
    # ...
    # Total 40 sets.
    
    # Let's use a simple heuristic: start with empty set, then add random elements.
    # Or just random distinct masks.
    # But we want comparability.
    # Let's try:
    # 10 chains of 4 sets each.
    # Chain i:
    #   S1 = {i}
    #   S2 = {i, (i+1)%12}
    #   S3 = {i, (i+1)%12, (i+2)%12}
    #   S4 = {i, (i+1)%12, (i+2)%12, (i+3)%12}
    
    used = set()
    for i in range(10):
        for j in range(4):
            mask = 0
            for k in range(j+1):
                mask |= (1 << ((i+k) % 12))
            if mask not in used:
                used.add(mask)
                masks.append(mask)
    
    # We have 40 masks now? 10*4 = 40.
    # Check distinctness.
    if len(masks) != 40:
        # Fill with random
        while len(masks) < 40:
            m = random.randint(0, (1<<12)-1)
            if m not in used:
                used.add(m)
                masks.append(m)
    
    return masks

def _local_search(masks, time_limit=50.0):
    """
    Local search to improve comparability.
    Moves:
    1. Replace a mask with a random subset/superset of another mask.
    2. Swap elements between two masks.
    3. Random replacement.
    """
    start_time = time.time()
    best_score = _count_comparable_pairs(masks)
    best_masks = masks[:]
    current_masks = masks[:]
    current_score = best_score
    
    n = P['n']
    m = P['m']
    
    # Precompute all possible masks? No, 4096 is small.
    all_masks = list(range(1 << n))
    
    # For each mask, precompute its subsets and supersets?
    # subsets[mask] = list of masks that are subsets of mask
    # supersets[mask] = list of masks that are supersets of mask
    # This might be expensive to precompute for all 4096 masks.
    # But 4096 * 4096 = 16M operations, which is fast in Python? Maybe not.
    # Let's just compute on the fly or use bit tricks.
    
    # Let's use a simpler move:
    # Pick two indices i, j.
    # Try to change masks[i] to be a subset or superset of masks[j].
    # Or pick i, and a random mask k, and try to make masks[i] a subset/superset of masks[k].
    
    # Let's try:
    # For each iteration:
    #   Pick random i.
    #   Pick random j != i.
    #   Generate a candidate mask for i that is a subset or superset of masks[j].
    #   If candidate is distinct from all other masks, accept if it improves score.
    
    # How to generate a subset of masks[j]?
    # Pick a random submask.
    # How to generate a superset of masks[j]?
    # Pick a random supermask.
    
    # But we also want to explore the space more broadly.
    # Let's mix moves:
    # 1. Subset/Superset move (70%)
    # 2. Random replacement (30%)
    
    max_iters = 100000
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Subset/Superset move
            i = random.randint(0, m-1)
            j = random.randint(0, m-1)
            while j == i:
                j = random.randint(0, m-1)
            
            base_mask = current_masks[j]
            if random.random() < 0.5:
                # Subset
                # Pick a random submask of base_mask
                # Number of bits in base_mask is k. Number of submasks is 2^k.
                # We can pick a random integer r in [0, 2^k) and map it to a submask.
                # Or simpler: for each bit in base_mask, keep it with prob 0.5.
                candidate = 0
                for b in range(n):
                    if (base_mask >> b) & 1:
                        if random.random() < 0.5:
                            candidate |= (1 << b)
            else:
                # Superset
                # Start with base_mask, add random bits not in base_mask.
                candidate = base_mask
                for b in range(n):
                    if not ((base_mask >> b) & 1):
                        if random.random() < 0.5:
                            candidate |= (1 << b)
            
            # Check distinctness
            valid = True
            for k in range(m):
                if k != i and current_masks[k] == candidate:
                    valid = False
                    break
            
            if valid:
                # Calculate new score
                # We can do this efficiently by only checking pairs involving i.
                old_pairs = 0
                for k in range(m):
                    if k == i:
                        continue
                    mi = current_masks[i]
                    mk = current_masks[k]
                    if mi == mk:
                        continue
                    if (mi & mk) == mi or (mi & mk) == mk:
                        old_pairs += 1
                
                new_pairs = 0
                for k in range(m):
                    if k == i:
                        continue
                    mi = candidate
                    mk = current_masks[k]
                    if mi == mk:
                        continue
                    if (mi & mk) == mi or (mi & mk) == mk:
                        new_pairs += 1
                
                delta = new_pairs - old_pairs
                if delta > 0:
                    current_masks[i] = candidate
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_masks = current_masks[:]
        
        else:
            # Random replacement
            i = random.randint(0, m-1)
            candidate = random.randint(0, (1<<n)-1)
            
            # Check distinctness
            valid = True
            for k in range(m):
                if k != i and current_masks[k] == candidate:
                    valid = False
                    break
            
            if valid:
                # Calculate new score
                old_pairs = 0
                for k in range(m):
                    if k == i:
                        continue
                    mi = current_masks[i]
                    mk = current_masks[k]
                    if mi == mk:
                        continue
                    if (mi & mk) == mi or (mi & mk) == mk:
                        old_pairs += 1
                
                new_pairs = 0
                for k in range(m):
                    if k == i:
                        continue
                    mi = candidate
                    mk = current_masks[k]
                    if mi == mk:
                        continue
                    if (mi & mk) == mi or (mi & mk) == mk:
                        new_pairs += 1
                
                delta = new_pairs - old_pairs
                if delta > 0:
                    current_masks[i] = candidate
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_masks = current_masks[:]
    
    return best_masks, best_score

def construct():
    """
    Return a list of exactly 40 lists, each a list of distinct integers from 0..11.
    Maximize the number of unordered pairs in which one member is a strict subset of the other.
    """
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    # Initial solution
    masks = _initial_solution()
    
    # Local search
    masks, score = _local_search(masks, time_limit=deadline - (time.time() - start_time))
    
    # Convert to lists
    result = _masks_to_lists(masks)
    
    # Ensure types are correct
    final_result = []
    for lst in result:
        final_result.append([int(x) for x in lst])
    
    return final_result
# EVOLVE-BLOCK-END
