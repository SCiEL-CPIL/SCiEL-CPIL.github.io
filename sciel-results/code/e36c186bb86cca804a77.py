import random
import time

def _score_from_list(lst):
    """
    Calculate score for a list of 32 distinct ints.
    """
    s_set = set(lst)
    valid_ds = set()
    # d in 1..255
    # Optimization: 
    # For a fixed d, we need a, a+d, a+2d in S.
    # This is equivalent to checking if (S & (S-d) & (S-2d)) is not empty?
    # Bitset approach might be faster if we use integers as bitsets, but 512 bits fits in Python int.
    
    # Let's try bitset approach for speed.
    # S is a set of ints in 1..512.
    # Create a bitmask B where bit i is 1 if i is in S.
    # For a given d, we want to know if there exists a such that bit a, a+d, a+2d are set.
    # This is equivalent to (B & (B >> d) & (B >> 2d)) != 0.
    
    # Let's implement this.
    mask = 0
    for x in lst:
        mask |= (1 << x)
        
    # Note: x is 1..512. Bit 0 is unused or used for 0.
    # Shifting right by d aligns bits.
    # If bit i is set in B, it means i is in S.
    # B >> d has bit i set if i+d is in S.
    # B >> 2d has bit i set if i+2d is in S.
    # Intersection non-zero means there is some i (which is 'a') such that a, a+d, a+2d are in S.
    
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
            valid_ds.add(d)
            
    return len(valid_ds)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0
    best_list = []
    
    def init_random():
        return random.sample(range(1, 513), 32)
        
    def init_spread():
        points = []
        # Try to cover the range
        for i in range(32):
            # Uniform distribution with jitter
            p = int((i / 32.0) * 512) + random.randint(0, 15)
            if p < 1: p = 1
            if p > 512: p = 512
            points.append(p)
        while len(set(points)) < 32:
            points[random.randint(0, 31)] = random.randint(1, 512)
        return points

    initial_sets = []
    for _ in range(10):
        initial_sets.append(init_random())
    for _ in range(10):
        initial_sets.append(init_spread())
    
    for init in initial_sets:
        if time.time() > deadline:
            break
            
        current_list = init.copy()
        current_score = _score_from_list(current_list)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
            
        no_improve_count = 0
        # Limit steps per init to ensure we try multiple inits
        max_steps = 5000
        
        while no_improve_count < 100 and max_steps > 0 and time.time() < deadline:
            max_steps -= 1
            
            idx = random.randint(0, 31)
            old_val = current_list[idx]
            
            new_val = random.randint(1, 512)
            if new_val == old_val:
                continue
            # Check if new_val is already in list (excluding idx)
            # Since we use a set for score, we can just check membership
            # But we need to ensure distinctness in the list.
            if new_val in current_list:
                # If new_val is at a different index, swapping is a different move, 
                # but here we are replacing. If new_val is in list, we can't just add it.
                # We could swap old_val with the element at new_val's index.
                # For simplicity, skip if duplicate.
                continue
                
            current_list[idx] = new_val
            new_score = _score_from_list(current_list)
            
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            else:
                # Revert
                current_list[idx] = old_val
                no_improve_count += 1
                
    if not best_list:
        return list(range(1, 33))
        
    return best_list
