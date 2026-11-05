import time
import random
import sys

N = 256

def _precompute():
    """Precompute all 4-APs and related structures for fast checking."""
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                all_4aps.append((a, b, c, e))
    
    # For each element x, precompute bitmask of all 4-APs containing x
    # 4ap_mask[x] is a list of (other1, other2, other3) tuples
    # But for speed, we precompute: for each x, a list of bitmasks representing
    # the other 3 elements in each 4-AP containing x
    # Then checking if adding x creates a 4-AP is: any((mask & bm) == bm for bm in x_4ap_bitmasks)
    
    x_4ap_bitmasks = [[] for _ in range(N + 1)]
    for (a, b, c, e) in all_4aps:
        x_4ap_bitmasks[a].append((1 << b) | (1 << c) | (1 << e))
        x_4ap_bitmasks[b].append((1 << a) | (1 << c) | (1 << e))
        x_4ap_bitmasks[c].append((1 << a) | (1 << b) | (1 << e))
        x_4ap_bitmasks[e].append((1 << a) | (1 << b) | (1 << c))
    
    return all_4aps, x_4ap_bitmasks

def _creates_4ap_fast(mask, x, x_4ap_bitmasks):
    """Check if adding x to mask creates a 4-AP using precomputed bitmasks."""
    for bm in x_4ap_bitmasks[x]:
        if (mask & bm) == bm:
            return True
    return False

def _count_3aps(mask):
    """Count 3-APs in set represented by bitmask."""
    count = 0
    for b in range(1, N + 1):
        if not (mask & (1 << b)):
            continue
        lower_mask = mask & ((1 << b) - 2)
        a = lower_mask
        while a:
            lb = a & (-a)
            a_idx = lb.bit_length() - 1
            c = 2 * b - a_idx
            if c > b and c <= N and (mask & (1 << c)):
                count += 1
            a ^= lb
    return count

def _delta_add(mask, x):
    """Count new 3-APs created by adding x to the set represented by mask."""
    count = 0
    # x as middle: (a, x, c) where a < x < c, c = 2x - a
    for a in range(1, x):
        if mask & (1 << a):
            c = 2 * x - a
            if c > x and c <= N and (mask & (1 << c)):
                count += 1
    # x as first: (x, b, c) where b = x+d, c = x+2d
    for b in range(x + 1, N + 1):
        if mask & (1 << b):
            c = 2 * b - x
            if c > b and c <= N and (mask & (1 << c)):
                count += 1
    # x as last: (a, b, x) where b = (a+x)/2
    for a in range(1, x):
        s = a + x
        if s % 2 == 0:
            b = s // 2
            if a < b < x and (mask & (1 << b)):
                count += 1
    return count

def _delta_remove(mask, x):
    """Count 3-APs destroyed by removing x from the set represented by mask."""
    return _delta_add(mask & ~(1 << x), x)

def _greedy_construct(x_4ap_bitmasks):
    """Greedy construction: add elements in order if no 4-AP is created."""
    mask = 0
    for x in range(1, N + 1):
        if not _creates_4ap_fast(mask, x, x_4ap_bitmasks):
            mask |= (1 << x)
    return mask

def _local_search(mask, score, x_4ap_bitmasks, deadline):
    """Hill climbing local search with swaps and additions."""
    best_mask = mask
    best_score = score
    
    while time.time() < deadline - 3.0:
        improved = False
        set_elements = [i for i in range(1, N + 1) if best_mask & (1 << i)]
        unset_elements = [i for i in range(1, N + 1) if not (best_mask & (1 << i))]
        
        # Try all additions
        for y in unset_elements[:]:
            if _creates_4ap_fast(best_mask, y, x_4ap_bitmasks):
                continue
            new_mask = best_mask | (1 << y)
            new_score = best_score + _delta_add(best_mask, y)
            if new_score > best_score:
                best_mask = new_mask
                best_score = new_score
                improved = True
                unset_elements.remove(y)
                set_elements.append(y)
        
        # Try all swaps
        for x in set_elements[:]:
            for y in unset_elements[:]:
                temp_mask = best_mask & ~(1 << x)
                if _creates_4ap_fast(temp_mask, y, x_4ap_bitmasks):
                    continue
                delta = _delta_add(temp_mask, y) - _delta_remove(best_mask, x)
                new_score = best_score + delta
                if new_score > best_score:
                    best_mask = temp_mask | (1 << y)
                    best_score = new_score
                    improved = True
                    set_elements.remove(x)
                    set_elements.append(y)
                    unset_elements.remove(y)
                    unset_elements.append(x)
                    break
        
        if not improved:
            break
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute
    all_4aps, x_4ap_bitmasks = _precompute()
    
    # Greedy construction
    mask = _greedy_construct(x_4ap_bitmasks)
    best_mask = mask
    best_score = _count_3aps(mask)
    
    # Simulated Annealing with multiple restarts
    random.seed(42)
    
    # Phase 1: SA from greedy solution
    current_mask = mask
    current_score = best_score
    
    T = 8.0
    T_min = 0.05
    cooling = 0.9997
    
    while time.time() < deadline - 20.0 and T > T_min:
        move_type = random.random()
        
        if move_type < 0.45:
            # Swap
            set_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
            unset_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
            
            if not set_elements or not unset_elements:
                T *= cooling
                continue
                
            x = random.choice(set_elements)
            y = random.choice(unset_elements)
            
            temp_mask = current_mask & ~(1 << x)
            if _creates_4ap_fast(temp_mask, y, x_4ap_bitmasks):
                T *= cooling
                continue
            
            new_mask = temp_mask | (1 << y)
            delta = _delta_add(temp_mask, y) - _delta_remove(current_mask, x)
            new_score = current_score + delta
            
        elif move_type < 0.75:
            # Add
            unset_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
            if not unset_elements:
                T *= cooling
                continue
            y = random.choice(unset_elements)
            
            if _creates_4ap_fast(current_mask, y, x_4ap_bitmasks):
                T *= cooling
                continue
            
            new_mask = current_mask | (1 << y)
            delta = _delta_add(current_mask, y)
            new_score = current_score + delta
            
        else:
            # Remove
            set_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
            if not set_elements:
                T *= cooling
                continue
            x = random.choice(set_elements)
            
            new_mask = current_mask & ~(1 << x)
            delta = -_delta_remove(current_mask, x)
            new_score = current_score + delta
        
        # Accept or reject
        if new_score >= current_score or random.random() < pow(2.718281828, delta / T):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling
    
    # Phase 2: Local search on best found
    best_mask, best_score = _local_search(best_mask, best_score, x_4ap_bitmasks, deadline)
    
    # Phase 3: Random restarts with local search
    while time.time() < deadline - 5.0:
        # Random restart: build a random valid set
        rand_mask = 0
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if not _creates_4ap_fast(rand_mask, x, x_4ap_bitmasks):
                rand_mask |= (1 << x)
        
        rand_score = _count_3aps(rand_mask)
        
        # Local search from random start
        ls_mask, ls_score = _local_search(rand_mask, rand_score, x_4ap_bitmasks, deadline)
        
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
    
    # Final local search polish
    best_mask, best_score = _local_search(best_mask, best_score, x_4ap_bitmasks, deadline)
    
    # Convert mask to sorted list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    
    return result
