import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # Safety margin before 90s limit


def _precompute_4ap_triples():
    """For each position p, precompute the set of (x,y) pairs such that
    {p, x, y} completes a 4-AP (i.e., p plus x and y form a 4-AP).
    Actually, we want: if p is added, which pairs (a,c) already in the set
    would create a 4-AP? No -- we need: adding p creates a 4-AP if there exist
    a,d such that {a, a+d, a+2d, a+3d} contains p and the other three are in the set.
    
    Better: precompute for each p, the list of "blocking pairs": pairs (i,j) 
    such that {p, i, j} is a subset of some 4-AP. If both i and j are in the set,
    adding p creates a 4-AP.
    
    Even better for O(1) check: maintain a 'blocked' count. For each p, 
    blocked[p] = number of 4-APs that contain p and have at least 2 other members in the set.
    Adding p is safe iff blocked[p] == 0.
    
    Let's precompute all 4-APs as tuples (a, a+d, a+2d, a+3d).
    For each 4-AP, for each element p in it, the other 3 elements form a "witness triple".
    When we add p, we need to check if any of the other 3 are all in the set.
    That's still O(number of 4-APs containing p) per check.
    
    Alternative: for each p, precompute list of "forbidden pairs": 
    For each 4-AP containing p, the other 3 elements -- if any pair of those 3 is in the set,
    and the third is also in the set... no, we need all 3.
    
    Let me use a different approach: for each p, maintain a count of how many 4-APs 
    containing p have exactly 2 of the other 3 elements in the set. 
    Actually, the cleanest is: for each p, blocked[p] counts the number of 4-APs 
    containing p where the other 3 are ALL in the set. If blocked[p] > 0, adding p is bad.
    
    When we add element q, for each 4-AP containing q, we need to update blocked for the 
    other 3 elements in that 4-AP. This is O(#4APs containing q) per add.
    """
    # Generate all 4-APs: (a, a+d, a+2d, a+3d) where all in [0, N)
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, list of 4-AP indices that contain it
    pos_to_4aps = [[] for _ in range(N)]
    for idx, ap in enumerate(all_4aps):
        for p in ap:
            pos_to_4aps[p].append(idx)
    
    return all_4aps, pos_to_4aps


def _count_3ap_fast(mask):
    """Count 3-APs using difference-based vectorization."""
    n = len(mask)
    count = 0
    for d in range(1, n // 2 + 1):
        a_part = mask[:n - 2 * d]
        b_part = mask[d:n - d]
        c_part = mask[2 * d:]
        count += np.sum(a_part & b_part & c_part)
    return int(count)


def _marginal_3ap_gain(mask, pos):
    """Compute how many new 3-APs are created by adding pos."""
    n = len(mask)
    gain = 0
    # pos as first (a): b = pos+d, c = pos+2d
    max_d = (n - 1 - pos) // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        gain += np.sum(mask[pos + d] & mask[pos + 2 * d])
    # pos as middle (b): a = pos-d, c = pos+d
    max_d = min(pos, n - 1 - pos)
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        gain += np.sum(mask[pos - d] & mask[pos + d])
    # pos as last (c): b = pos-d, a = pos-2d
    max_d = pos // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        gain += np.sum(mask[pos - d] & mask[pos - 2 * d])
    return int(gain)


def _greedy_build(all_4aps, pos_to_4aps, start_time, deadline, seed=42):
    """Build a 4-AP-free set using greedy with 3-AP potential bias."""
    rng = random.Random(seed)
    mask = np.zeros(N, dtype=bool)
    
    # blocked[p] = number of 4-APs containing p where all other 3 are in the set
    blocked = np.zeros(N, dtype=np.int32)
    
    remaining = list(range(N))
    rng.shuffle(remaining)
    
    # First pass: add elements that don't create 4-APs, prioritizing high 3-AP gain
    # Use a priority queue approach: repeatedly pick the element with max gain that's not blocked
    
    while remaining:
        if time.time() > deadline - 10:
            break
        
        best_pos = -1
        best_gain = -1
        
        # Sample a subset to speed up (or do full scan if fast enough)
        for pos in remaining:
            if blocked[pos] > 0:
                continue
            gain = _marginal_3ap_gain(mask, pos)
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
        
        if best_pos == -1:
            break
        
        # Add best_pos
        mask[best_pos] = True
        remaining.remove(best_pos)
        
        # Update blocked for elements in 4-APs containing best_pos
        for ap_idx in pos_to_4aps[best_pos]:
            ap = all_4aps[ap_idx]
            # For each other element in this 4-AP, check if the other 2 (besides this element and best_pos) are in set
            for other in ap:
                if other == best_pos:
                    continue
                # The other two elements in the 4-AP (besides other and best_pos)
                others = [x for x in ap if x != other and x != best_pos]
                if len(others) == 2:
                    if mask[others[0]] and mask[others[1]]:
                        blocked[other] += 1
    
    return mask


def _local_search_1flip(mask, all_4aps, pos_to_4aps, start_time, deadline):
    """Local search: try all single additions and removals."""
    current = mask.copy()
    current_score = _count_3ap_fast(current)
    best = current.copy()
    best_score = current_score
    
    # Compute blocked
    blocked = np.zeros(N, dtype=np.int32)
    for p in np.where(current)[0]:
        for ap_idx in pos_to_4aps[p]:
            ap = all_4aps[ap_idx]
            for other in ap:
                if other == p:
                    continue
                others = [x for x in ap if x != other and x != p]
                if len(others) == 2:
                    if current[others[0]] and current[others[1]]:
                        blocked[other] += 1
    
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        
        # Try adding each unblocked element
        for pos in range(N):
            if time.time() > deadline - 5:
                break
            if current[pos]:
                continue
            if blocked[pos] > 0:
                continue
            gain = _marginal_3ap_gain(current, pos)
            if gain > 0:
                current[pos] = True
                current_score += gain
                # Update blocked
                for ap_idx in pos_to_4aps[pos]:
                    ap = all_4aps[ap_idx]
                    for other in ap:
                        if other == pos:
                            continue
                        others = [x for x in ap if x != other and x != pos]
                        if len(others) == 2:
                            if current[others[0]] and current[others[1]]:
                                blocked[other] += 1
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing each element
        for pos in range(N):
            if time.time() > deadline - 5:
                break
            if not current[pos]:
                continue
            # Compute loss from removing pos
            loss = _marginal_3ap_gain(current, pos)  # This is gain from adding, which equals loss from removing
            if loss < 0:
                # Removing pos increases score (shouldn't happen with non-negative gains, but check)
                current[pos] = False
                current_score -= loss  # loss is negative, so subtracting negative = adding
                # Update blocked (decrease)
                for ap_idx in pos_to_4aps[pos]:
                    ap = all_4aps[ap_idx]
                    for other in ap:
                        if other == pos:
                            continue
                        others = [x for x in ap if x != other and x != pos]
                        if len(others) == 2:
                            if current[others[0]] and current[others[1]]:
                                blocked[other] -= 1
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
                improved = True
                break
    
    return best, best_score


def _swap_search(mask, all_4aps, pos_to_4aps, start_time, deadline, max_rounds=50):
    """Try swaps: remove one, add another."""
    current = mask.copy()
    current_score = _count_3ap_fast(current)
    best = current.copy()
    best_score = current_score
    
    # Compute blocked
    blocked = np.zeros(N, dtype=np.int32)
    for p in np.where(current)[0]:
        for ap_idx in pos_to_4aps[p]:
            ap = all_4aps[ap_idx]
            for other in ap:
                if other == p:
                    continue
                others = [x for x in ap if x != other and x != p]
                if len(others) == 2:
                    if current[others[0]] and current[others[1]]:
                        blocked[other] += 1
    
    selected = list(np.where(current)[0])
    unselected = list(np.where(~current)[0])
    
    rng = random.Random(123)
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try random swaps
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            if not selected or not unselected:
                break
            
            pos_remove = rng.choice(selected)
            pos_add = rng.choice(unselected)
            
            if blocked[pos_add] > 0:
                continue
            
            # Compute net gain
            gain_add = _marginal_3ap_gain(current, pos_add)
            loss_remove = _marginal_3ap_gain(current, pos_remove)
            
            # But removing pos_remove might affect gain_add if they interact
            # For simplicity, approximate: net = gain_add - loss_remove
            # More accurate: compute actual score change
            new_mask = current.copy()
            new_mask[pos_remove] = False
            new_mask[pos_add] = True
            new_score = _count_3ap_fast(new_mask)
            delta = new_score - current_score
            
            if delta > 0:
                current = new_mask
                current_score = new_score
                selected.remove(pos_remove)
                selected.append(pos_add)
                unselected.remove(pos_add)
                unselected.append(pos_remove)
                
                # Recompute blocked (expensive but safe)
                blocked = np.zeros(N, dtype=np.int32)
                for p in np.where(current)[0]:
                    for ap_idx in pos_to_4aps[p]:
                        ap = all_4aps[ap_idx]
                        for other in ap:
                            if other == p:
                                continue
                            others = [x for x in ap if x != other and x != p]
                            if len(others) == 2:
                                if current[others[0]] and current[others[1]]:
                                    blocked[other] += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
                improved = True
                break
        
        if not improved:
            break
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute 4-AP structure
    all_4aps, pos_to_4aps = _precompute_4ap_triples()
    
    best_mask = None
    best_score = -1
    
    # Multi-start greedy
    for seed in range(10):
        if time.time() > deadline - 15:
            break
        
        mask = _greedy_build(all_4aps, pos_to_4aps, start_time, deadline, seed=seed + 1)
        score = _count_3ap_fast(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask.copy()
    
    if best_mask is None:
        return [1]
    
    # Local search
    if time.time() < deadline - 10:
        best_mask, best_score = _local_search_1flip(best_mask, all_4aps, pos_to_4aps, start_time, deadline)
    
    # Swap search
    if time.time() < deadline - 5:
        best_mask, best_score = _swap_search(best_mask, all_4aps, pos_to_4aps, start_time, deadline)
    
    # Final local search
    if time.time() < deadline - 3:
        best_mask, best_score = _local_search_1flip(best_mask, all_4aps, pos_to_4aps, start_time, deadline)
    
    result = sorted([int(i) + 1 for i in np.where(best_mask)[0]])
    
    if len(result) == 0:
        return [1]
    
    return result
