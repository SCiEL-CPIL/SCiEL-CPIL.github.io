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

def _greedy_init():
    """
    Construct an initial set using a greedy approach: add the element that maximizes the score increase.
    """
    mask = 0
    elements = []
    used = [False] * 513  # 1..512
    
    for _ in range(32):
        best_elem = -1
        best_score = -1
        # Try a subset of candidates to speed up? No, 512 is small.
        # But 32 * 512 * 255 ops is ~4M ops. This is fast enough for init.
        for x in range(1, 513):
            if not used[x]:
                new_mask = mask | (1 << x)
                sc = _score(new_mask)
                if sc > best_score:
                    best_score = sc
                    best_elem = x
        # Add best_elem
        mask |= (1 << best_elem)
        elements.append(best_elem)
        used[best_elem] = True
        
    return mask, elements

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_mask = 0
    best_score = 0
    
    # Helper to create mask from set
    def make_mask(s):
        m = 0
        for x in s:
            m |= (1 << x)
        return m

    # We will run multiple restarts
    # Each restart runs SA for a certain number of steps or time slice
    
    num_restarts = 0
    
    # Use greedy init for the first restart? Or random?
    # Let's try greedy init first, then random restarts.
    
    while time.time() < deadline:
        if num_restarts == 0:
            current_mask, current_list = _greedy_init()
        else:
            current_list = random.sample(range(1, 513), 32)
            current_mask = 0
            for x in current_list:
                current_mask |= (1 << x)
        
        current_set = set(current_list)
        current_score = _score(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        # SA parameters
        T = 1.0
        stagnation = 0
        max_stagnation = 10000
        
        # Run SA for this restart
        while time.time() < deadline and stagnation < max_stagnation:
            # Select i from current_list (O(1))
            idx = random.randint(0, 31)
            i = current_list[idx]
            
            # Select j not in current_set
            # Density is 1/16, so ~16 tries.
            while True:
                j = random.randint(1, 512)
                if j not in current_set:
                    break
            
            new_mask = current_mask ^ (1 << i) ^ (1 << j)
            new_score = _score(new_mask)
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta > 0 or (T > 1e-6 and random.random() < math.exp(delta / T)):
                # Accept
                current_set.remove(i)
                current_set.add(j)
                current_list[idx] = j # Update list in place
                current_mask = new_mask
                current_score = new_score
                stagnation = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
                if best_score == 255:
                    break
            else:
                stagnation += 1
            
            # Cooling
            T *= 0.999
            if T < 0.001:
                break
        
        num_restarts += 1
        # If we've run many restarts, maybe reduce time per restart?
        # The while loop handles this naturally.
        
    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx)
        m ^= lsb
        
    return result
