"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _count_comparable(masks):
    """Count unordered pairs (i,j) where masks[i] is strict subset of masks[j] or vice versa."""
    n = len(masks)
    count = 0
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            if mi != mj and (mi & mj) == mi:
                count += 1
            elif mi != mj and (mi & mj) == mj:
                count += 1
    return count

def _greedy_init():
    """Build a good initial solution by greedily adding sets that maximize comparable pairs."""
    n = 12
    m = 40
    all_masks = list(range(1 << n))
    random.shuffle(all_masks)
    
    # Start with empty set and full set (they're comparable to everything)
    family = [0, (1 << n) - 1]
    used = set(family)
    
    while len(family) < m:
        best_mask = None
        best_gain = -1
        for mask in all_masks:
            if mask in used:
                continue
            gain = 0
            for fam in family:
                if mask != fam and ((mask & fam) == mask or (mask & fam) == fam):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        if best_mask is None:
            break
        family.append(best_mask)
        used.add(best_mask)
    
    # If we didn't fill up, add random unused masks
    if len(family) < m:
        for mask in all_masks:
            if len(family) >= m:
                break
            if mask not in used:
                family.append(mask)
                used.add(mask)
    
    return family[:m]

def _local_search(family, deadline):
    """Local search: try replacing one set with another to improve score."""
    n = 12
    m = 40
    all_masks = list(range(1 << n))
    current_score = _count_comparable(family)
    used = set(family)
    
    while time.time() < deadline:
        improved = False
        # Try removing each set and finding the best replacement
        for i in range(m):
            if time.time() >= deadline:
                break
            removed = family[i]
            # Try adding each unused mask
            for new_mask in all_masks:
                if new_mask in used:
                    continue
                if new_mask == removed:
                    continue
                # Quick check: will this improve?
                # Count gain from new_mask vs loss from removed
                gain = 0
                loss = 0
                for j in range(m):
                    if j == i:
                        continue
                    fam = family[j]
                    # Does new_mask comparable with fam?
                    if new_mask != fam and ((new_mask & fam) == new_mask or (new_mask & fam) == fam):
                        gain += 1
                    # Was removed comparable with fam?
                    if removed != fam and ((removed & fam) == removed or (removed & fam) == fam):
                        loss += 1
                delta = gain - loss
                if delta > 0:
                    family[i] = new_mask
                    used.discard(removed)
                    used.add(new_mask)
                    current_score += delta
                    improved = True
                    break  # Restart from beginning after improvement
        
        if not improved:
            # Try random perturbations to escape local optima
            for _ in range(10):
                if time.time() >= deadline:
                    break
                i = random.randint(0, m - 1)
                old = family[i]
                new_mask = random.choice(all_masks)
                if new_mask in used:
                    continue
                gain = 0
                loss = 0
                for j in range(m):
                    if j == i:
                        continue
                    fam = family[j]
                    if new_mask != fam and ((new_mask & fam) == new_mask or (new_mask & fam) == fam):
                        gain += 1
                    if old != fam and ((old & fam) == old or (old & fam) == fam):
                        loss += 1
                delta = gain - loss
                # Accept with some probability even if worse (simulated annealing lite)
                if delta >= 0 or random.random() < 0.1:
                    family[i] = new_mask
                    used.discard(old)
                    used.add(new_mask)
                    current_score += delta
    
    return family, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    # Multiple restarts with different seeds
    best_family = None
    best_score = -1
    
    for seed in range(50):
        if time.time() >= deadline:
            break
        random.seed(seed)
        family = _greedy_init()
        family, score = _local_search(family, deadline)
        if score > best_score:
            best_score = score
            best_family = family[:]
        
        # If we're doing well, do a focused refinement
        if time.time() >= deadline - 20:
            break
    
    # Final refinement on the best family found
    if best_family is None:
        random.seed(0)
        best_family = _greedy_init()
        best_family, best_score = _local_search(best_family, deadline)
    
    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        result.append([b for b in range(12) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
