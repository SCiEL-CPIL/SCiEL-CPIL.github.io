import time
import random
import math

def _score_bitmask(mask):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set represented by mask.
    mask is an integer where bit i is 1 if (i+1) is in the set.
    """
    count = 0
    # Max possible d is such that a+2d <= 512.
    # If a=1, 1+2d <= 512 => 2d <= 511 => d <= 255.
    # But we only have 32 elements.
    # Max d is bounded by the max element in S.
    # Actually, we can just iterate d from 1 to 255.
    for d in range(1, 256):
        # Check if there exists a such that a, a+d, a+2d in S
        # Bit a-1 in mask, bit a+d-1 in mask, bit a+2d-1 in mask
        # Equivalent to: mask & (mask >> d) & (mask >> 2*d) != 0
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Random seed for reproducibility, but can be changed
    rng = random.Random(42)
    
    # Initialize with a structured set to give a good starting point
    # Try a few different initializations and pick the best
    candidates = []
    
    # 1. Arithmetic Progression 1..32
    s1 = set(range(1, 33))
    candidates.append(s1)
    
    # 2. Arithmetic Progression with step 2 (1, 3, ..., 63)
    s2 = set(range(1, 64, 2))
    candidates.append(s2)
    
    # 3. Arithmetic Progression with step 16 (1, 17, ..., 1+31*16)
    # 1 + 31*16 = 500. Fits in 512.
    s3 = set(1 + 16 * i for i in range(32))
    candidates.append(s3)
    
    # 4. Grid 4x8
    s4 = set()
    for i in range(4):
        for j in range(8):
            s4.add(1 + i * 8 + j) # 1..32? No, this is just 1..32.
            # Let's use a wider grid.
            # i in 0..3, j in 0..7. Value = 1 + i*16 + j?
            # Max: 1 + 3*16 + 7 = 56.
    s4 = set(1 + i * 16 + j for i in range(4) for j in range(8))
    candidates.append(s4)
    
    # 5. Random
    s5 = set(rng.sample(range(1, 513), 32))
    candidates.append(s5)

    # Convert to bitmask and score
    def set_to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m

    best_mask = 0
    best_score = -1
    for s in candidates:
        m = set_to_mask(s)
        sc = _score_bitmask(m)
        if sc > best_score:
            best_score = sc
            best_mask = m

    # Simulated Annealing
    # State: mask
    current_mask = best_mask
    current_score = best_score
    
    # Cooling schedule
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate neighbor: remove one element, add another
        # Find an element to remove
        # We can pick a random bit that is set
        bits_set = []
        temp_mask = current_mask
        idx = 0
        while temp_mask:
            if temp_mask & 1:
                bits_set.append(idx)
            temp_mask >>= 1
            idx += 1
        
        if not bits_set:
            break
            
        remove_idx = rng.choice(bits_set)
        
        # Find an element to add
        # Pick a random index not in bits_set
        # There are 512 - 32 = 480 available spots
        # We can pick a random number 1..512 and check if it's in the set
        # Or pick from available list.
        # To speed up, just pick random 1..512 and retry if collision (low prob)
        while True:
            add_idx = rng.randint(1, 512) - 1
            if not (current_mask & (1 << add_idx)):
                break
                
        new_mask = current_mask
        new_mask &= ~(1 << remove_idx)
        new_mask |= (1 << add_idx)
        
        new_score = _score_bitmask(new_mask)
        
        delta = new_score - current_score
        
        if delta > 0 or (rng.random() < math.exp(delta / T)):
            current_mask = new_mask
            current_score = new_score
        
        T *= alpha

    # Convert best_mask back to list
    result = []
    temp_mask = current_mask
    idx = 0
    while temp_mask:
        if temp_mask & 1:
            result.append(idx + 1)
        temp_mask >>= 1
        idx += 1
        
    return result
