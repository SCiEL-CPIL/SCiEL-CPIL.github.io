import time
import random
import math

def _count_pairs(family):
    """Count unordered pairs where one is a strict subset of the other."""
    count = 0
    n = len(family)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = family[i], family[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _incremental_score_add(family, mask):
    """Score of adding mask to family (comparability count with existing members)."""
    score = 0
    for s in family:
        if (mask & s) == mask or (mask & s) == s:
            score += 1
    return score

def _incremental_score_remove(family, idx):
    """Score lost by removing family[idx] from family."""
    mask = family[idx]
    score = 0
    for j in range(len(family)):
        if j == idx:
            continue
        s = family[j]
        if (mask & s) == mask or (mask & s) == s:
            score += 1
    return score

def _swap_score_change(family, idx, new_mask):
    """Score change when replacing family[idx] with new_mask."""
    old_mask = family[idx]
    # Remove old_mask: lose comparabilities with all others
    delta = 0
    for j in range(len(family)):
        if j == idx:
            continue
        s = family[j]
        old_comp = 1 if (old_mask & s) == old_mask or (old_mask & s) == s else 0
        new_comp = 1 if (new_mask & s) == new_mask or (new_mask & s) == s else 0
        delta += new_comp - old_comp
    return delta

def _greedy_construct(n, m):
    """Greedy construction: start with empty and full, add best candidate iteratively."""
    total_masks = 1 << n
    selected = [0, total_masks - 1]
    selected_set = {0, total_masks - 1}
    
    for _ in range(m - 2):
        best_score = -1
        best_mask = -1
        for mask in range(total_masks):
            if mask in selected_set:
                continue
            score = 0
            for s in selected:
                if (mask & s) == mask or (mask & s) == s:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask != -1:
            selected.append(best_mask)
            selected_set.add(best_mask)
        else:
            break
    return selected

def _local_search(family, n, max_iters=5000, deadline=None):
    """Local search with swap moves. Returns improved family."""
    total_masks = 1 << n
    m = len(family)
    current_score = _count_pairs(family)
    best_family = list(family)
    best_score = current_score
    family_set = set(family)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random swap: pick random index, random new mask
        idx = random.randint(0, m - 1)
        new_mask = random.randint(0, total_masks - 1)
        if new_mask in family_set:
            continue
        
        delta = _swap_score_change(family, idx, new_mask)
        if delta > 0:
            # Accept
            family_set.discard(family[idx])
            family[idx] = new_mask
            family_set.add(new_mask)
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
    
    return best_family, best_score

def _simulated_annealing(family, n, deadline=None):
    """Simulated annealing with swap moves."""
    total_masks = 1 << n
    m = len(family)
    current_score = _count_pairs(family)
    best_family = list(family)
    best_score = current_score
    family_set = set(family)
    
    T_start = 10.0
    T_end = 0.1
    # Estimate number of iterations we can do
    # Each iteration is ~O(m) for score change
    # Let's aim for ~20000 iterations
    n_iters = 20000
    cooling = (T_end / T_start) ** (1.0 / n_iters)
    T = T_start
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = random.randint(0, m - 1)
        new_mask = random.randint(0, total_masks - 1)
        if new_mask in family_set:
            T *= cooling
            continue
        
        delta = _swap_score_change(family, idx, new_mask)
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            family_set.discard(family[idx])
            family[idx] = new_mask
            family_set.add(new_mask)
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
        
        T *= cooling
    
    return best_family, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s (actually 90s) limit
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    best_family = None
    best_score = -1
    
    # Multiple restarts with different strategies
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Greedy initialization
        family = _greedy_construct(n, m)
        score = _count_pairs(family)
        
        if score > best_score:
            best_score = score
            best_family = list(family)
        
        # Local search
        if time.time() < deadline - 5:
            ls_family, ls_score = _local_search(family, n, max_iters=3000, deadline=deadline)
            if ls_score > best_score:
                best_score = ls_score
                best_family = ls_family
        
        # Simulated annealing on the best found so far
        if time.time() < deadline - 10:
            sa_family, sa_score = _simulated_annealing(best_family, n, deadline=deadline)
            if sa_score > best_score:
                best_score = sa_score
                best_family = sa_family
        
        # Another local search round
        if time.time() < deadline - 5:
            ls_family2, ls_score2 = _local_search(best_family, n, max_iters=3000, deadline=deadline)
            if ls_score2 > best_score:
                best_score = ls_score2
                best_family = ls_family2
    
    # Final verification
    if best_family is None:
        # Fallback: just use greedy
        best_family = _greedy_construct(n, m)
    
    # Convert to lists of integers
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
    
    # Ensure exactly 40 lists
    if len(result) < m:
        used_masks = set(best_family)
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)
        for mask in all_masks:
            if len(result) >= m:
                break
            if mask not in used_masks:
                used_masks.add(mask)
                subset = [i for i in range(n) if mask & (1 << i)]
                result.append(subset)
    
    # Ensure we have exactly 40
    if len(result) > m:
        result = result[:m]
    
    return result
