import time
import random
import sys
import math

def construct():
    """
    Returns exactly 32 distinct integers in 1..512.
    Maximizes the number of distinct d > 0 such that a, a+d, a+2d are all in the set.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 32
    MAX_VAL = 512
    MAX_D = 255 # Max d such that 1+2d <= 512

    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        """
        valid_diffs = 0
        # Iterate d from 1 to 255
        # Optimization: We can break early if we've found all? No, max is 255.
        for d in range(1, MAX_D + 1):
            # Check if there is any a such that bits a, a+d, a+2d are set.
            # M has bit a set.
            # M >> d has bit a set if M has bit a+d set.
            # M >> 2d has bit a set if M has bit a+2d set.
            # Intersection non-zero means such 'a' exists.
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                valid_diffs += 1
        return valid_diffs

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def mask_to_list(mask):
        # Extract bits
        # This is only needed at the end
        res = []
        m = mask
        i = 1
        while m:
            if m & 1:
                res.append(i)
            m >>= 1
            i += 1
        return res

    # 1. Greedy Initial Construction
    # Start with empty set
    best_mask = 0
    best_score = 0
    current_set = set()
    
    # To speed up greedy, we might just pick random good ones, 
    # but full greedy is 32*512 = 16k evals, which is fast.
    # However, greedy gets stuck in local optima. 
    # Let's do a few random restarts of greedy? No, just one greedy then SA.
    
    # Actually, a better start is to mix greedy and random.
    # Let's generate 5 candidates:
    # 1. Greedy
    # 2. Random
    # 3. Arithmetic Progression (step 16)
    # 4. Dense (1..32)
    # 5. Spread (multiples of 16)
    
    candidates = []
    
    # Candidate 1: Greedy
    m = 0
    s = 0
    used = [False] * (MAX_VAL + 1)
    for _ in range(N):
        best_x = -1
        best_s = -1
        for x in range(1, MAX_VAL + 1):
            if used[x]:
                continue
            new_m = m | (1 << x)
            # We don't need full score every time? 
            # Full score is 255 ops. 512*255 = 130k ops per step. 32 steps = 4M ops.
            # This might be slow in Python (4M iterations of inner loop).
            # 4M simple bitwise ops is probably < 1s. Let's risk it.
            sc = score_mask(new_m)
            if sc > best_s:
                best_s = sc
                best_x = x
        m |= (1 << best_x)
        used[best_x] = True
    candidates.append(m)
    
    # Candidate 2: Random
    lst = random.sample(range(1, MAX_VAL + 1), N)
    candidates.append(list_to_mask(lst))
    
    # Candidate 3: AP
    lst = list(range(1, 1 + 31*16, 16))
    candidates.append(list_to_mask(lst))
    
    # Candidate 4: Dense
    lst = list(range(1, 33))
    candidates.append(list_to_mask(lst))
    
    # Candidate 5: Spread
    lst = [i*16 for i in range(1, 33)] # 16, 32, ..., 512
    # Ensure within range
    lst = [x for x in lst if x <= 512]
    if len(lst) < 32:
        # Fill remaining
        remaining = [x for x in range(1, 513) if x not in lst]
        lst += random.sample(remaining, 32 - len(lst))
    candidates.append(list_to_mask(lst))
    
    # Evaluate candidates
    best_mask = 0
    best_score = -1
    for m in candidates:
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_mask = m
            
    # 2. Simulated Annealing
    current_mask = best_mask
    current_score = best_score
    current_set = set(mask_to_list(current_mask)) # For O(1) lookup
    
    T = 5.0
    cooling_rate = 0.99999 # Very slow cooling
    min_T = 0.01
    
    # Reheat parameters
    no_improve_count = 0
    reheat_threshold = 5000 # If no improvement in 5000 steps, reheat
    reheat_T = 2.0
    
    while time.time() < deadline:
        # Generate move: Replace one element
        # Pick random index in the set? 
        # We need to pick an element to remove and an element to add.
        # Remove: Pick random element from current_set.
        # Add: Pick random element from 1..512 not in current_set.
        
        # To avoid creating duplicates, we pick remove_val from current_set, 
        # and add_val from range(1, 513) excluding current_set.
        
        # Efficient way:
        # 1. Pick remove_val
        remove_val = random.choice(list(current_set))
        
        # 2. Pick add_val
        # Randomly pick a number, if in set, pick another.
        # Or precompute available? 
        # Since 512 is small, rejection sampling is fine.
        while True:
            add_val = random.randint(1, MAX_VAL)
            if add_val not in current_set:
                break
        
        if add_val == remove_val:
            continue
            
        new_mask = current_mask ^ (1 << remove_val) ^ (1 << add_val)
        new_score = score_mask(new_mask)
        delta = new_score - current_score
        
        if delta > 0:
            accept = True
        else:
            if T < min_T:
                accept = False
            else:
                # Use math.exp for accuracy
                try:
                    accept = random.random() < math.exp(delta / T)
                except OverflowError:
                    accept = False
        
        if accept:
            current_mask = new_mask
            current_score = new_score
            current_set.remove(remove_val)
            current_set.add(add_val)
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            T *= cooling_rate
            if T < min_T:
                T = min_T
        else:
            no_improve_count += 1
            
        # Reheat logic
        if no_improve_count > reheat_threshold:
            T = reheat_T
            no_improve_count = 0
            reheat_threshold = int(reheat_threshold * 1.5) # Increase threshold for future reheats
            
    # 3. Final Polish: Hill Climbing
    # Try all single moves that improve score. Repeat until no improvement.
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Get list of elements
        curr_list = mask_to_list(current_mask)
        curr_set = set(curr_list)
        
        for remove_val in curr_list:
            # Try all possible add_vals
            for add_val in range(1, MAX_VAL + 1):
                if add_val in curr_set:
                    continue
                new_mask = current_mask ^ (1 << remove_val) ^ (1 << add_val)
                new_score = score_mask(new_mask)
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    curr_set.remove(remove_val)
                    curr_set.add(add_val)
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                    improved = True
                    break # Restart loop with new set
            if improved:
                break # Restart outer loop
                
    return mask_to_list(best_mask)

