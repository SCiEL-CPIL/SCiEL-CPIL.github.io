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

def _count_comparable(masks):
    """Count unordered pairs where one is a strict subset of the other."""
    count = 0
    n = len(masks)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = masks[i], masks[j]
            if a != b:
                if (a & b) == a or (a & b) == b:
                    count += 1
    return count

def _delta_score(masks, idx, new_mask):
    """Compute the change in comparable pairs when replacing masks[idx] with new_mask."""
    old_mask = masks[idx]
    if old_mask == new_mask:
        return 0
    
    delta = 0
    for j in range(len(masks)):
        if j == idx:
            continue
        other = masks[j]
        
        # Old relationship
        old_comp = 0
        if old_mask != other:
            if (old_mask & other) == old_mask or (old_mask & other) == other:
                old_comp = 1
        
        # New relationship
        new_comp = 0
        if new_mask != other:
            if (new_mask & other) == new_mask or (new_mask & other) == other:
                new_comp = 1
        
        delta += new_comp - old_comp
    
    return delta

def _chain_initialization():
    """
    Initialize with a long chain structure: pick a random ordering of elements,
    then take prefixes of that ordering to form a chain. Fill remaining slots
    with sets that are comparable to many chain members.
    """
    random.seed()
    order = list(range(N))
    random.shuffle(order)
    
    masks = []
    # Build a chain: empty set, {order[0]}, {order[0],order[1]}, ..., all 12
    for k in range(N + 1):
        m = 0
        for i in range(k):
            m |= (1 << order[i])
        masks.append(m)
    
    # We have N+1 = 13 masks in the chain. Need M - 13 = 27 more.
    # Add sets that extend or are subsets of chain members
    used = set(masks)
    
    # Try to add sets that are comparable to many existing masks
    all_masks = list(range(1 << N))
    random.shuffle(all_masks)
    
    candidates = []
    for m in all_masks:
        if m in used:
            continue
        # Count how many existing masks are comparable with m
        comp_count = 0
        for fm in masks:
            if m != fm:
                if (m & fm) == m or (m & fm) == fm:
                    comp_count += 1
        candidates.append((comp_count, m))
    
    candidates.sort(key=lambda x: -x[0])
    
    for comp_count, m in candidates:
        if len(masks) >= M:
            break
        masks.append(m)
        used.add(m)
    
    # If still not enough, add random
    for m in all_masks:
        if len(masks) >= M:
            break
        if m not in used:
            masks.append(m)
            used.add(m)
    
    return [_mask_to_list(m) for m in masks]

def _greedy_construct():
    """
    Greedy construction: start with empty family, repeatedly add the set
    that maximizes the marginal gain in comparable pairs.
    """
    used = set()
    family_masks = []
    
    # Start with empty set (mask 0)
    family_masks.append(0)
    used.add(0)
    
    all_masks = list(range(1, 1 << N))
    
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
        family_masks.append(best_mask)
    
    return [_mask_to_list(m) for m in family_masks]

def _local_search(family, time_limit=5.0):
    """
    Local search with incremental delta computation.
    Try single swaps and accept improvements.
    """
    start_time = time.time()
    masks = []
    for s in family:
        m = 0
        for v in s:
            m |= (1 << v)
        masks.append(m)
    
    current_score = _count_comparable(masks)
    
    all_masks = list(range(1 << N))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        random.shuffle(all_masks)
        
        for i in range(M):
            for new_m in all_masks:
                if new_m == masks[i]:
                    continue
                # Check if new_m is already in the family (at another position)
                found = False
                for j in range(M):
                    if j != i and masks[j] == new_m:
                        found = True
                        break
                if found:
                    continue
                
                # Try replacing masks[i] with new_m
                delta = _delta_score(masks, i, new_m)
                
                if delta > 0:
                    masks[i] = new_m
                    current_score += delta
                    improved = True
                elif delta == 0 and random.random() < 0.1:
                    # Accept neutral moves with small probability to escape plateaus
                    masks[i] = new_m
                    improved = True
                
                if (time.time() - start_time) > time_limit:
                    break
            if (time.time() - start_time) > time_limit:
                break
    
    return [_mask_to_list(m) for m in masks], current_score

def _double_swap_search(family, time_limit=5.0):
    """
    Local search that tries pairs of swaps simultaneously.
    """
    start_time = time.time()
    masks = []
    for s in family:
        m = 0
        for v in s:
            m |= (1 << v)
        masks.append(m)
    
    current_score = _count_comparable(masks)
    
    all_masks = list(range(1 << N))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        random.shuffle(all_masks)
        
        for i in range(M):
            for new_m1 in all_masks:
                if new_m1 == masks[i]:
                    continue
                # Check if new_m1 is already in the family (at another position)
                found = False
                for j in range(M):
                    if j != i and masks[j] == new_m1:
                        found = True
                        break
                if found:
                    continue
                
                for j in range(i + 1, M):
                    for new_m2 in all_masks:
                        if new_m2 == masks[j] or new_m2 == new_m1:
                            continue
                        # Check if new_m2 is already in the family (at another position)
                        found2 = False
                        for k in range(M):
                            if k != i and k != j and masks[k] == new_m2:
                                found2 = True
                                break
                        if found2:
                            continue
                        
                        # Compute delta for double swap
                        old_i = masks[i]
                        old_j = masks[j]
                        
                        delta = 0
                        for k in range(M):
                            if k == i or k == j:
                                continue
                            other = masks[k]
                            
                            # Old relationships
                            old_comp_i = 0
                            if old_i != other:
                                if (old_i & other) == old_i or (old_i & other) == other:
                                    old_comp_i = 1
                            
                            old_comp_j = 0
                            if old_j != other:
                                if (old_j & other) == old_j or (old_j & other) == other:
                                    old_comp_j = 1
                            
                            # New relationships
                            new_comp_i = 0
                            if new_m1 != other:
                                if (new_m1 & other) == new_m1 or (new_m1 & other) == other:
                                    new_comp_i = 1
                            
                            new_comp_j = 0
                            if new_m2 != other:
                                if (new_m2 & other) == new_m2 or (new_m2 & other) == other:
                                    new_comp_j = 1
                            
                            delta += (new_comp_i + new_comp_j) - (old_comp_i + old_comp_j)
                        
                        # Also account for the i-j pair itself
                        old_ij = 0
                        if old_i != old_j:
                            if (old_i & old_j) == old_i or (old_i & old_j) == old_j:
                                old_ij = 1
                        
                        new_ij = 0
                        if new_m1 != new_m2:
                            if (new_m1 & new_m2) == new_m1 or (new_m1 & new_m2) == new_m2:
                                new_ij = 1
                        
                        delta += new_ij - old_ij
                        
                        if delta > 0:
                            masks[i] = new_m1
                            masks[j] = new_m2
                            current_score += delta
                            improved = True
                            break
                        
                        if (time.time() - start_time) > time_limit:
                            break
                    
                    if (time.time() - start_time) > time_limit:
                        break
                
                if (time.time() - start_time) > time_limit:
                    break
            
            if (time.time() - start_time) > time_limit:
                break
    
    return [_mask_to_list(m) for m in masks], current_score

def construct():
    start_time = time.time()
    
    # Try multiple strategies and keep the best
    best_family = None
    best_score = -1
    
    # Strategy 1: Chain initialization + local search
    family1 = _chain_initialization()
    score1 = _count_comparable([sum(1 << v for v in s) for s in family1])
    if score1 > best_score:
        best_score = score1
        best_family = family1
    
    if time.time() - start_time > DEADLINE:
        return best_family
    
    # Strategy 2: Local search from chain
    family2, score2 = _local_search(family1, time_limit=min(8.0, DEADLINE - (time.time() - start_time) - 5.0))
    if score2 > best_score:
        best_score = score2
        best_family = family2
    
    if time.time() - start_time > DEADLINE:
        return best_family
    
    # Strategy 3: Greedy construction + local search
    family3 = _greedy_construct()
    score3 = _count_comparable([sum(1 << v for v in s) for s in family3])
    if score3 > best_score:
        best_score = score3
        best_family = family3
    
    if time.time() - start_time > DEADLINE:
        return best_family
    
    family3, score3 = _local_search(family3, time_limit=min(8.0, DEADLINE - (time.time() - start_time) - 5.0))
    if score3 > best_score:
        best_score = score3
        best_family = family3
    
    if time.time() - start_time > DEADLINE:
        return best_family
    
    # Strategy 4: Double swap search on best family
    remaining = DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 3.0:
        family4, score4 = _double_swap_search(best_family, time_limit=min(10.0, remaining))
        if score4 > best_score:
            best_score = score4
            best_family = family4
    
    if time.time() - start_time > DEADLINE:
        return best_family
    
    # Strategy 5: More local search on best
    remaining = DEADLINE - (time.time() - start_time) - 3.0
    if remaining > 2.0:
        family5, score5 = _local_search(best_family, time_limit=min(8.0, remaining))
        if score5 > best_score:
            best_score = score5
            best_family = family5
    
    # Ensure valid return
    if best_family is None:
        best_family = _chain_initialization()
    
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
