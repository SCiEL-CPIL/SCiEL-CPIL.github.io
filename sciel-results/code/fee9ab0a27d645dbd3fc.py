import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    best_score = 0
    best_set = None
    all_nums = list(range(1, 513))
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Initialize: random set of 32 elements from 1..512
        S = set(random.sample(all_nums, 32))
        S_bits = 0
        for x in S:
            S_bits |= (1 << x)
        
        cur_score = _score_set_bits(S_bits)
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = set(S)
        
        # Local search with simulated annealing
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        no_improve = 0
        max_no_improve = 10000
        
        while time.time() < deadline - 5:
            if no_improve > max_no_improve:
                break
            
            # Pick random element to remove
            x = random.choice(list(S))
            
            # Pick random element to add
            while True:
                y = random.choice(all_nums)
                if y not in S:
                    break
            
            # Try swap
            # Update bits
            S_bits ^= (1 << x) # Remove x
            S_bits ^= (1 << y) # Add y
            
            new_score = _score_set_bits(S_bits)
            
            delta = new_score - cur_score
            if delta > 0 or random.random() < np.exp(delta / T):
                # Accept
                S.discard(x)
                S.add(y)
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
            else:
                # Reject, revert bits
                S_bits ^= (1 << x)
                S_bits ^= (1 << y)
                no_improve += 1
            
            T = max(T_min, T * cooling)
        
        # Reset temperature for next restart
        T = 5.0
    
    if best_set is None:
        return list(range(1, 33))
    
    return list(best_set)

def _score_set_bits(bits):
    count = 0
    # d ranges from 1 to 255
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2*d))) != 0:
            count += 1
    return count
