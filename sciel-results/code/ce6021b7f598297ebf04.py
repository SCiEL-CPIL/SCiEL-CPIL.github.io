import random
import math
import time

def _score(mask):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set represented by mask.
    mask is an integer where bit i is 1 if i is in the set.
    """
    s = 0
    m = mask
    # d can be at most 255 because a+2d <= 512 implies 2d <= 511 (since a>=1)
    # Actually a+2d <= 512. If a=1, 1+2d <= 512 => 2d <= 511 => d <= 255.
    for d in range(1, 256):
        if (m & (m >> d) & (m >> (2 * d))) != 0:
            s += 1
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_mask = 0
    best_score = 0
    
    # Helper to create mask from set
    def make_mask(s):
        m = 0
        for x in s:
            m |= (1 << x)
        return m

    def get_elements(mask):
        elems = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elems.append(idx)
            m ^= lsb
        return elems

    # Greedy initialization with randomization
    def greedy_init():
        mask = 0
        elems = []
        current_score = 0
        for _ in range(32):
            best_elem = -1
            best_gain = -1
            best_new_score = -1
            
            # Collect candidates
            candidates = []
            for x in range(1, 513):
                if mask & (1 << x):
                    continue
                new_mask = mask | (1 << x)
                sc = _score(new_mask)
                candidates.append((sc, x))
            
            if not candidates:
                break
                
            # Sort by score descending
            candidates.sort(reverse=True, key=lambda t: t[0])
            
            # Pick from top 5 to add randomness
            top_k = min(5, len(candidates))
            sc, x = random.choice(candidates[:top_k])
            
            mask |= (1 << x)
            elems.append(x)
            current_score = sc
            
        return mask, elems

    # Run multiple restarts
    # 1. Greedy Start
    mask, elems = greedy_init()
    current_score = _score(mask)
    if current_score > best_score:
        best_score = current_score
        best_mask = mask

    # SA Polish on Greedy
    T = 1.0
    stagnation = 0
    max_stagnation = 10000
    
    while time.time() < deadline and stagnation < max_stagnation:
        # Pick i from current elements
        idx_i = random.randint(0, 31)
        i = elems[idx_i]
        
        # Pick j not in mask
        while True:
            j = random.randint(1, 512)
            if not (mask & (1 << j)):
                break
        
        new_mask = mask ^ (1 << i) ^ (1 << j)
        new_score = _score(new_mask)
        delta = new_score - current_score
        
        if delta > 0 or (T > 1e-6 and random.random() < math.exp(delta / T)):
            # Accept
            mask = new_mask
            current_score = new_score
            elems[idx_i] = j
            
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
            
            stagnation = 0
            if best_score == 255:
                break
        else:
            stagnation += 1
        
        T *= 0.999
        if T < 0.001:
            break
            
    # 2. Random Restarts
    while time.time() < deadline:
        # Random init
        current_elems = random.sample(range(1, 513), 32)
        mask = make_mask(current_elems)
        current_score = _score(mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = mask
            
        T = 1.0
        stagnation = 0
        max_stagnation = 10000
        
        while time.time() < deadline and stagnation < max_stagnation:
            idx_i = random.randint(0, 31)
            i = current_elems[idx_i]
            
            while True:
                j = random.randint(1, 512)
                if not (mask & (1 << j)):
                    break
            
            new_mask = mask ^ (1 << i) ^ (1 << j)
            new_score = _score(new_mask)
            delta = new_score - current_score
            
            if delta > 0 or (T > 1e-6 and random.random() < math.exp(delta / T)):
                mask = new_mask
                current_score = new_score
                current_elems[idx_i] = j
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                
                stagnation = 0
                if best_score == 255:
                    break
            else:
                stagnation += 1
                
            T *= 0.999
            if T < 0.001:
                break

    # Convert best_mask to list
    result = get_elements(best_mask)
    
    return result
