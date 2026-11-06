"""Program for reciprocal weight of three-progression-free sets on [1,256]."""
import time
import random
import math

# EVOLVE-BLOCK-START
N = 256
RECIPS = [0.0] + [1.0 / i for i in range(1, N + 1)]

def _is_3ap_free_add(mask, x):
    """Check if adding x to set represented by mask keeps it 3-AP free.
    x is 1-indexed. mask is int where bit i-1 is set if i is in set.
    """
    if (mask >> (x - 1)) & 1:
        return True
    
    m = mask
    while m:
        lsb = m & -m
        a = lsb.bit_length()  # 1-indexed
        m ^= lsb
        
        # x is middle: c = 2x - a
        c = 2 * x - a
        if c > 0 and c <= N and c != a and c != x:
            if (mask >> (c - 1)) & 1:
                return False
        
        # x is end: b = (x + a) / 2
        if (x + a) % 2 == 0:
            b = (x + a) // 2
            if b > 0 and b <= N and b != a and b != x:
                if (mask >> (b - 1)) & 1:
                    return False
    return True

def _get_sum(mask):
    s = 0.0
    m = mask
    while m:
        lsb = m & -m
        i = lsb.bit_length()
        s += RECIPS[i]
        m ^= lsb
    return s

def _get_elements(mask):
    """Return list of elements in mask."""
    result = []
    m = mask
    while m:
        lsb = m & -m
        i = lsb.bit_length()
        result.append(i)
        m ^= lsb
    return result

def _greedy_ascending():
    mask = 0
    for x in range(1, N + 1):
        if _is_3ap_free_add(mask, x):
            mask |= (1 << (x - 1))
    return mask

def _polish(mask, deadline):
    """Final polish: try all single-element swaps to improve the sum."""
    best_mask = mask
    best_score = _get_sum(mask)
    elements = _get_elements(mask)
    element_set = set(elements)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in elements:
            if time.time() > deadline:
                break
            # Try removing i
            mask_without_i = best_mask ^ (1 << (i - 1))
            for j in range(1, N + 1):
                if j in element_set or j == i:
                    continue
                if not _is_3ap_free_add(mask_without_i, j):
                    continue
                new_mask = mask_without_i | (1 << (j - 1))
                new_score = best_score - RECIPS[i] + RECIPS[j]
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    elements = _get_elements(best_mask)
                    element_set = set(elements)
                    improved = True
                    break
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Start with greedy ascending
    best_mask = _greedy_ascending()
    best_score = _get_sum(best_mask)
    
    # Simulated Annealing with swap moves
    current_mask = best_mask
    current_score = best_score
    
    # Initial temperature
    T = 0.005
    cooling_rate = 0.995
    
    iteration = 0
    while time.time() < deadline:
        if T < 1e-6:
            break
        
        iteration += 1
        elements = _get_elements(current_mask)
        n_elements = len(elements)
        
        # Choose move type: 60% add, 20% remove, 20% swap
        move_type = random.random()
        
        if move_type < 0.6:
            # Add a random element
            x = random.randint(1, N)
            if (current_mask >> (x - 1)) & 1:
                T *= cooling_rate
                continue
            if _is_3ap_free_add(current_mask, x):
                new_mask = current_mask | (1 << (x - 1))
                new_score = current_score + RECIPS[x]
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            T *= cooling_rate
        
        elif move_type < 0.8:
            # Remove a random element
            if n_elements > 1:
                x = random.choice(elements)
                new_mask = current_mask ^ (1 << (x - 1))
                new_score = current_score - RECIPS[x]
                delta = new_score - current_score
                # Always valid to remove
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            T *= cooling_rate
        
        else:
            # Swap: remove one, add another
            if n_elements > 1:
                x = random.choice(elements)
                y = random.randint(1, N)
                if y == x or (current_mask >> (y - 1)) & 1:
                    T *= cooling_rate
                    continue
                mask_without_x = current_mask ^ (1 << (x - 1))
                if _is_3ap_free_add(mask_without_x, y):
                    new_mask = mask_without_x | (1 << (y - 1))
                    new_score = current_score - RECIPS[x] + RECIPS[y]
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            T *= cooling_rate
    
    # Final polish pass
    if time.time() < deadline - 2.0:
        best_mask, best_score = _polish(best_mask, deadline)
    
    # Convert best_mask to list
    result = _get_elements(best_mask)
    return sorted(result)

# EVOLVE-BLOCK-END
