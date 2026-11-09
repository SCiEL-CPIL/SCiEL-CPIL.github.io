import time
import random
import math

N = 12
M = 40
FULL_MASK = (1 << N) - 1

def _count_pairs(masks):
    """Count comparable pairs in a list of masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _incremental_score(masks, idx, new_mask):
    """Compute score change when replacing masks[idx] with new_mask.
    Returns the delta in comparable pair count."""
    old_mask = masks[idx]
    delta = 0
    for j in range(M):
        if j == idx:
            continue
        c = masks[j]
        old_comp = 1 if ((old_mask & c) == old_mask or (old_mask & c) == c) else 0
        new_comp = 1 if ((new_mask & c) == new_mask or (new_mask & c) == c) else 0
        delta += new_comp - old_comp
    return delta

def _greedy_construct():
    """Greedy construction of 40 masks."""
    chosen = []
    all_masks = list(range(1 << N))
    
    for _ in range(M):
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in chosen:
                continue
            score = 0
            for c in chosen:
                if (mask & c) == mask or (mask & c) == c:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        chosen.append(best_mask)
        
    return chosen

def _local_search(masks, max_iter=50000, deadline=None):
    """Local search with incremental score updates to improve the set of masks."""
    current_score = _count_pairs(masks)
    all_masks = list(range(1 << N))
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        idx = random.randint(0, M - 1)
        old_mask = masks[idx]
        
        new_mask = random.choice(all_masks)
        if new_mask == old_mask:
            continue
        if new_mask in masks:
            continue
            
        delta = _incremental_score(masks, idx, new_mask)
        if delta > 0:
            masks[idx] = new_mask
            current_score += delta
        else:
            # Revert (no change needed since we didn't modify)
            pass
            
    return masks

def _simulated_annealing(masks, deadline=None):
    """Simulated annealing to escape local optima."""
    current_score = _count_pairs(masks)
    best_score = current_score
    best_masks = list(masks)
    all_masks = list(range(1 << N))
    
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = 1.0  # restart annealing
            cooling = 0.9995
        
        idx = random.randint(0, M - 1)
        old_mask = masks[idx]
        
        new_mask = random.choice(all_masks)
        if new_mask == old_mask:
            continue
        if new_mask in masks:
            continue
        
        delta = _incremental_score(masks, idx, new_mask)
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            masks[idx] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = list(masks)
        
        T *= cooling
        
        if random.random() < 0.001:
            T = max(T * 0.5, T_min)  # occasional aggressive cooling
    
    return best_masks

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    random.seed(42)
    
    # 1. Greedy initialization
    masks = _greedy_construct()
    
    # 2. Local search with incremental updates
    masks = _local_search(masks, max_iter=100000, deadline=deadline)
    
    # 3. Simulated annealing to escape local optima
    remaining_deadline = deadline - 5  # leave 5s buffer
    if time.time() < remaining_deadline:
        masks = _simulated_annealing(masks, deadline=remaining_deadline)
    
    # 4. Final local search polish
    if time.time() < deadline - 1:
        masks = _local_search(masks, max_iter=50000, deadline=deadline - 0.5)
    
    # Convert masks to lists of vertices
    result = []
    for mask in masks:
        result.append([b for b in range(N) if (mask >> b) & 1])
        
    return result
