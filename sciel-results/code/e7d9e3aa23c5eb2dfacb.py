import time
import itertools
import random

def _count_comparable(sets):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(sets)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = sets[i], sets[j]
            if len(si) < len(sj):
                if all(x in sj for x in si):
                    count += 1
            elif len(sj) < len(si):
                if all(x in si for x in sj):
                    count += 1
    return count

def _count_comparable_fast(sets, n=12):
    """Count using bitmask representation for speed."""
    m = len(sets)
    masks = [0] * m
    for i, s in enumerate(sets):
        mask = 0
        for x in s:
            mask |= (1 << x)
        masks[i] = mask
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            mi, mj = masks[i], masks[j]
            if mi & mj == mi:  # i is subset of j
                if mi != mj:
                    count += 1
            elif mi & mj == mj:  # j is subset of i
                count += 1
    return count

def _construct_initial():
    """Construct a strong initial solution: chain + additional sets."""
    n = 12
    # Chain: {0}, {0,1}, {0,1,2}, ..., {0,1,...,11} and empty set
    chain = [list(range(i)) for i in range(13)]  # sizes 0 to 12
    sets = list(chain)
    
    # Add 27 more sets to reach 40
    # Strategy: add sets that are subsets of chain elements or supersets
    # to maximize comparability
    existing = set(tuple(s) for s in sets)
    
    # Add sets that extend the chain structure
    # Try adding sets with various patterns
    candidates = []
    
    # All subsets of size 1-11 (excluding those already in chain)
    for size in range(1, 12):
        for combo in itertools.combinations(range(n), size):
            c = list(combo)
            if tuple(c) not in existing:
                candidates.append(c)
    
    # Sort candidates by potential comparability
    # Prefer sets that are subsets of or supersets of many existing sets
    def potential(s):
        s_set = set(s)
        score = 0
        for t in sets:
            t_set = set(t)
            if s_set < t_set or t_set < s_set:
                score += 1
        return score
    
    candidates.sort(key=potential, reverse=True)
    
    for c in candidates:
        if len(sets) >= 40:
            break
        sets.append(c)
        existing.add(tuple(c))
    
    # If we still need more, add random sets
    while len(sets) < 40:
        size = random.randint(0, n)
        c = random.sample(range(n), size)
        if tuple(c) not in existing:
            sets.append(c)
            existing.add(tuple(c))
    
    return sets[:40]

def _local_search(sets, max_time=50.0):
    """Local search with swap moves."""
    start = time.time()
    n = len(sets)
    best_score = _count_comparable_fast(sets)
    
    # Precompute masks
    def get_masks(s):
        return [0 if x not in s else (1 << x) for x in range(12)]
    
    all_masks = []
    for s in sets:
        mask = 0
        for x in s:
            mask |= (1 << x)
        all_masks.append(mask)
    
    def count_with_masks(masks):
        m = len(masks)
        cnt = 0
        for i in range(m):
            for j in range(i + 1, m):
                mi, mj = masks[i], masks[j]
                if mi != mj and (mi & mj == mi or mi & mj == mj):
                    cnt += 1
        return cnt
    
    current_score = best_score
    
    # Generate all possible masks
    all_possible_masks = list(range(1 << 12))
    used = set(all_masks)
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start < max_time and no_improve < max_no_improve:
        improved = False
        # Try swapping each set with a better one
        for i in range(n):
            if time.time() - start > max_time - 5:
                break
            current_mask = all_masks[i]
            # Try removing and replacing with another mask
            for new_mask in all_possible_masks:
                if new_mask in used or new_mask == current_mask:
                    continue
                # Quick check: would this improve?
                # Count comparability of new_mask with all others
                gain = 0
                for j in range(n):
                    if j == i:
                        continue
                    mj = all_masks[j]
                    # Old contribution
                    if current_mask != mj and (current_mask & mj == current_mask or current_mask & mj == mj):
                        gain -= 1
                    # New contribution
                    if new_mask != mj and (new_mask & mj == new_mask or new_mask & mj == mj):
                        gain += 1
                if gain > 0:
                    # Accept
                    used.remove(current_mask)
                    used.add(new_mask)
                    all_masks[i] = new_mask
                    current_score += gain
                    improved = True
                    no_improve = 0
                    break
            if improved:
                continue
            no_improve += 1
        
        if not improved:
            no_improve += 100
    
    # Convert back to lists
    result = []
    for mask in all_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result

def construct():
    start = time.time()
    
    # Initial construction
    sets = _construct_initial()
    
    # Local search with time budget
    if time.time() - start < 50:
        sets = _local_search(sets, max_time=55.0)
    
    # Final safety check
    if time.time() - start > 58:
        # Return what we have
        pass
    
    # Ensure exactly 40 distinct sets
    seen = set()
    final = []
    for s in sets:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            final.append(list(s))
    
    # If we lost some due to duplicates, add more
    if len(final) < 40:
        all_masks = set(tuple(sorted(s)) for s in final)
        for mask in range(1 << 12):
            if len(final) >= 40:
                break
            s = [b for b in range(12) if (mask >> b) & 1]
            t = tuple(sorted(s))
            if t not in all_masks:
                final.append(s)
                all_masks.add(t)
    
    return final[:40]
