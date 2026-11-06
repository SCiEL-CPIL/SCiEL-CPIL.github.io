import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    best_score = 0
    best_set = None
    all_nums = list(range(1, 513))
    
    # Precompute masks for faster checking? 
    # bits & (bits >> d) & (bits >> 2*d)
    # This is already very fast.
    
    # Initialization strategies
    inits = []
    
    # 1. Random
    inits.append(set(random.sample(all_nums, 32)))
    
    # 2. Dense block
    inits.append(set(range(1, 33)))
    
    # 3. Dense block with shift
    inits.append(set(range(10, 42)))
    
    # 4. Spread out
    inits.append(set(range(1, 513, 16))) # 1, 17, 33...
    # Ensure exactly 32
    if len(inits[-1]) < 32:
        # Fill with random
        remaining = [x for x in all_nums if x not in inits[-1]]
        inits[-1].update(random.sample(remaining, 32 - len(inits[-1])))
    elif len(inits[-1]) > 32:
        inits[-1] = set(list(inits[-1])[:32])

    # Add a few more random ones
    for _ in range(10):
        inits.append(set(random.sample(all_nums, 32)))

    for S_init in inits:
        if time.time() > deadline - 10:
            break
            
        S = set(S_init)
        S_list = list(S)
        S_bits = 0
        for x in S:
            S_bits |= (1 << x)
        
        cur_score = _score_set_bits(S_bits)
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = set(S)
        
        # Local Search: Best-First with Plateau Moves
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            best_swap_score = cur_score
            best_swap_x = -1
            best_swap_y = -1
            best_delta = 0
            
            # Try all single swaps
            for i in range(len(S_list)):
                x = S_list[i]
                # Optimization: Only check y that are not in S
                # We can iterate through all_nums and skip if in S
                # Or maintain a list of non-S elements?
                # For now, just loop.
                for y in all_nums:
                    if y in S:
                        continue
                    S_bits_new = S_bits ^ (1 << x) ^ (1 << y)
                    new_score = _score_set_bits(S_bits_new)
                    delta = new_score - cur_score
                    if delta > best_delta:
                        best_delta = delta
                        best_swap_score = new_score
                        best_swap_x = x
                        best_swap_y = y
            
            if best_swap_x != -1:
                # Apply best swap
                S.discard(best_swap_x)
                S.add(best_swap_y)
                S_list.remove(best_swap_x)
                S_list.append(best_swap_y)
                S_bits ^= (1 << best_swap_x) ^ (1 << best_swap_y)
                cur_score = best_swap_score
                no_improve = 0
                
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
            else:
                # No improvement found.
                # Try a random move to escape local minimum (Simulated Annealing style)
                # Pick a random x in S and random y not in S
                x = random.choice(S_list)
                while True:
                    y = random.choice(all_nums)
                    if y not in S:
                        break
                
                S.discard(x)
                S.add(y)
                S_list.remove(x)
                S_list.append(y)
                S_bits ^= (1 << x) ^ (1 << y)
                cur_score = _score_set_bits(S_bits)
                no_improve += 1
        
        # If we hit max_no_improve, we might be stuck. 
        # The random move above should have perturbed it.
        # If we still don't improve, the while loop will terminate.

    if best_set is None:
        return list(range(1, 33))
    
    return list(best_set)

def _score_set_bits(bits):
    count = 0
    # d can be at most (512-1)/2 = 255
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2*d))) != 0:
            count += 1
    return count
