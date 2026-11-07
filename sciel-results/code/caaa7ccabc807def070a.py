import time

def _count_comparable_bitmasks(masks):
    """Count unordered pairs where one mask is a strict subset of the other."""
    n = len(masks)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = masks[i], masks[j]
            if a != b and (a & b) == a:
                count += 1
            elif a != b and (a & b) == b:
                count += 1
    return count

def _delta_score(masks, i, old_mask, new_mask):
    """Compute change in comparable count when replacing masks[i] from old_mask to new_mask."""
    delta = 0
    for j in range(len(masks)):
        if j == i:
            continue
        m = masks[j]
        # Old contribution: pair (old_mask, m)
        if old_mask != m:
            if (old_mask & m) == old_mask or (old_mask & m) == m:
                delta -= 1
        # New contribution: pair (new_mask, m)
        if new_mask != m:
            if (new_mask & m) == new_mask or (new_mask & m) == m:
                delta += 1
    return delta

def _construct_base():
    """Construct a strong initial solution: chain + greedy additions."""
    n = 12
    m = 40
    
    # Full chain: mask for size i is (1<<i) - 1
    chain = [(1 << i) - 1 for i in range(n + 1)]  # 13 sets
    masks = list(chain)
    
    all_masks = list(range(1 << n))
    remaining = set(all_masks) - set(masks)
    
    current_score = _count_comparable_bitmasks(masks)
    
    for _ in range(m - len(masks)):
        best_mask = -1
        best_score = -1
        for cand in remaining:
            delta = _delta_score(masks, len(masks), -1, cand)  # Adding, not replacing
            # For adding, delta is just the number of comparable pairs with existing
            count = 0
            for existing in masks:
                if cand != existing:
                    if (cand & existing) == cand or (cand & existing) == existing:
                        count += 1
            if count > best_score:
                best_score = count
                best_mask = cand
        masks.append(best_mask)
        remaining.discard(best_mask)
    
    return masks

def _local_search(masks, time_limit):
    """Hill-climbing with incremental score updates."""
    n = 12
    m = len(masks)
    all_masks = list(range(1 << n))
    
    current_score = _count_comparable_bitmasks(masks)
    start_time = time.time()
    mask_set = set(masks)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(m):
            if (time.time() - start_time) > time_limit:
                break
            old = masks[i]
            for cand in all_masks:
                if cand == old or cand in mask_set:
                    continue
                delta = _delta_score(masks, i, old, cand)
                if delta > 0:
                    masks[i] = cand
                    current_score += delta
                    mask_set.discard(old)
                    mask_set.add(cand)
                    improved = True
                    break  # Restart scan after improvement
    
    return masks, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    masks = _construct_base()
    
    if time.time() > deadline - 5:
        return [[int(b) for b in range(12) if (mask >> b) & 1] for mask in masks]
    
    masks, score = _local_search(masks, deadline - time.time() - 5)
    
    return [[int(b) for b in range(12) if (mask >> b) & 1] for mask in masks]
