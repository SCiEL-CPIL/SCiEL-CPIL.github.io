import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    def score_bitset(b):
        count = 0
        # d can go up to 255 because 2d < 512
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))) != 0:
                count += 1
        return count

    def set_from_bitset(b):
        s = []
        i = 0
        while b:
            if b & 1:
                s.append(i + 1)
            b >>= 1
            i += 1
        return s

    # 1. Greedy Initialization
    current_mask = 0
    for _ in range(32):
        best_x = -1
        best_score = -1
        # Sample a subset of candidates if time is tight, otherwise all
        candidates = [x for x in range(1, 513) if not (current_mask & (1 << (x-1)))]
        # To speed up, we can just check all 512, it's fast enough for 32 steps
        for x in candidates:
            new_mask = current_mask | (1 << (x-1))
            s = score_bitset(new_mask)
            if s > best_score:
                best_score = s
                best_x = x
        if best_x != -1:
            current_mask |= (1 << (best_x - 1))
        else:
            break
            
    current_score = score_bitset(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    # 2. Local Search (Simulated Annealing)
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    while T > T_min and time.time() < deadline - 10.0:
        # Generate neighbor: swap one in, one out
        in_set = set_from_bitset(current_mask)
        not_in_set = [x for x in range(1, 513) if x not in set(in_set)]
        
        if not in_set or not not_in_set:
            break
            
        x_out = random.choice(in_set)
        x_in = random.choice(not_in_set)
        
        new_mask = (current_mask & ~(1 << (x_out - 1))) | (1 << (x_in - 1))
        new_score = score_bitset(new_mask)
        
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < pow(2.71828, delta / T):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                
        T *= cooling
        
    # 3. Final Polish with Hill Climbing
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            in_set = set_from_bitset(best_mask)
            not_in_set = [x for x in range(1, 513) if x not in set(in_set)]
            for x_out in in_set:
                if time.time() > deadline - 2.0: break
                for x_in in not_in_set:
                    if time.time() > deadline - 2.0: break
                    new_mask = (best_mask & ~(1 << (x_out - 1))) | (1 << (x_in - 1))
                    new_score = score_bitset(new_mask)
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = new_mask
                        improved = True
                        in_set = set_from_bitset(best_mask)
                        not_in_set = [x for x in range(1, 513) if x not in set(in_set)]
                        # Restart loops to ensure we check all pairs with new set? 
                        # Or just break inner loop? 
                        # Better to restart the whole hill climb if we find improvement to be safe.
                        break
                if improved:
                    break
                    
    return set_from_bitset(best_mask)
