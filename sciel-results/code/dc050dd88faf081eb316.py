"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(masks):
    """Count unordered pairs where one mask is a strict subset of the other."""
    m = len(masks)
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if (mi & mj) == mi and mi != mj:
                count += 1
            elif (mi & mj) == mj and mi != mj:
                count += 1
    return count

def _delta_score(masks, idx, new_mask):
    """Calculate change in comparable pair count when replacing masks[idx] with new_mask."""
    old_mask = masks[idx]
    if old_mask == new_mask:
        return 0
    m = len(masks)
    delta = 0
    for j in range(m):
        if j == idx:
            continue
        mj = masks[j]
        # Old contribution
        if (old_mask & mj) == old_mask and old_mask != mj:
            delta -= 1
        elif (old_mask & mj) == mj and old_mask != mj:
            delta -= 1
        # New contribution
        if (new_mask & mj) == new_mask and new_mask != mj:
            delta += 1
        elif (new_mask & mj) == mj and new_mask != mj:
            delta += 1
    return delta

def _greedy_init(seed):
    """Build 40 distinct subsets greedily to maximize comparable pairs."""
    rng = random.Random(seed)
    n = P['n']
    m = P['m']
    all_masks = list(range(1 << n))  # 4096 possible subsets
    rng.shuffle(all_masks)
    
    chosen = []
    used = set()
    
    for _ in range(m):
        best_mask = None
        best_score = -1
        # Try candidates, prefer those that form many comparable pairs with chosen
        for mask in all_masks:
            if mask in used:
                continue
            # Score: number of comparable pairs with already chosen sets
            s = 0
            for cm in chosen:
                if (mask & cm) == mask and mask != cm:
                    s += 1
                elif (mask & cm) == cm and mask != cm:
                    s += 1
            # Small random tiebreaker
            s += rng.random() * 0.001
            if s > best_score:
                best_score = s
                best_mask = mask
        if best_mask is not None:
            chosen.append(best_mask)
            used.add(best_mask)
        else:
            # Fallback: pick any unused
            for mask in all_masks:
                if mask not in used:
                    chosen.append(mask)
                    used.add(mask)
                    break
    
    return chosen

def _local_search(masks, seed, deadline):
    """Local search: try replacing each set to improve comparable pairs."""
    rng = random.Random(seed)
    n = P['n']
    m = len(masks)
    all_masks = list(range(1 << n))
    rng.shuffle(all_masks)
    
    current_score = _count_comparable_pairs(masks)
    best_score = current_score
    best_masks = list(masks)
    
    # Precompute for speed
    used_set = set(masks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order of indices
        indices = list(range(m))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() >= deadline:
                break
            old_mask = masks[idx]
            # Try a few random replacements
            for _ in range(20):
                new_mask = rng.choice(all_masks)
                if new_mask == old_mask or new_mask in used_set:
                    continue
                d = _delta_score(masks, idx, new_mask)
                if d > 0:
                    used_set.discard(old_mask)
                    masks[idx] = new_mask
                    used_set.add(new_mask)
                    current_score += d
                    if current_score > best_score:
                        best_score = current_score
                        best_masks = list(masks)
                    improved = True
                    break
                # Also try: remove a random element from old_mask, or add one
                # Try adding a random element
                if old_mask != (1 << n) - 1:
                    bits_not_in = [b for b in range(n) if not (old_mask >> b) & 1]
                    if bits_not_in:
                        b = rng.choice(bits_not_in)
                        new_mask2 = old_mask | (1 << b)
                        if new_mask2 not in used_set:
                            d2 = _delta_score(masks, idx, new_mask2)
                            if d2 > 0:
                                used_set.discard(old_mask)
                                masks[idx] = new_mask2
                                used_set.add(new_mask2)
                                current_score += d2
                                if current_score > best_score:
                                    best_score = current_score
                                    best_masks = list(masks)
                                improved = True
                                break
                # Try removing a random element
                if old_mask != 0:
                    bits_in = [b for b in range(n) if (old_mask >> b) & 1]
                    if bits_in:
                        b = rng.choice(bits_in)
                        new_mask3 = old_mask & ~(1 << b)
                        if new_mask3 not in used_set:
                            d3 = _delta_score(masks, idx, new_mask3)
                            if d3 > 0:
                                used_set.discard(old_mask)
                                masks[idx] = new_mask3
                                used_set.add(new_mask3)
                                current_score += d3
                                if current_score > best_score:
                                    best_score = current_score
                                    best_masks = list(masks)
                                improved = True
                                break
    
    return best_masks, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    best_overall = None
    best_score_overall = -1
    
    # Try multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        
        # Greedy init
        masks = _greedy_init(seed)
        score = _count_comparable_pairs(masks)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = list(masks)
        
        # Local search
        if time.time() < deadline - 5:
            ls_masks, ls_score = _local_search(masks, seed + 1000, deadline - 2)
            if ls_score > best_score_overall:
                best_score_overall = ls_score
                best_overall = ls_masks
    
    if best_overall is None:
        # Fallback
        best_overall = _greedy_init(0)
    
    # Convert to list of lists of ints
    result = []
    for mask in best_overall:
        s = [b for b in range(P['n']) if (mask >> b) & 1]
        result.append(s)
    
    return result
# EVOLVE-BLOCK-END
