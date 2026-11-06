import time
import random
import math

# Precompute masks for all possible differences d in 1..255
# S is represented as an integer where bit i is 1 if (i+1) is in the set.
# Range is 1..512, so we need bits 0..511.
# d ranges from 1 to 255 (max d such that a+2d <= 512 for a>=1 is 255).
# Actually, max d is 255 because 1, 1+d, 1+2d <= 512 => 2d <= 511 => d <= 255.

def _score_bitset(s_int):
    """
    Calculate the number of distinct d > 0 such that a 3-AP exists in the set.
    s_int: integer representing the set. Bit i corresponds to element i+1.
    """
    count = 0
    # We can iterate d from 1 to 255
    # Check if (s_int & (s_int >> d) & (s_int >> (2*d))) != 0
    for d in range(1, 256):
        if (s_int & (s_int >> d) & (s_int >> (2 * d))) != 0:
            count += 1
    return count

def _local_search(start_int, deadline):
    """
    Perform local search / simulated annealing starting from start_int.
    Returns the best integer found.
    """
    current_int = start_int
    current_score = _score_bitset(current_int)
    
    best_int = current_int
    best_score = current_score
    
    # Simulated Annealing parameters
    # Start temperature high enough to accept bad moves, cool down
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    # Precompute list of all bits 0..511
    all_bits = [1 << i for i in range(512)]
    
    # To speed up move generation, we can keep track of set bits and unset bits?
    # Or just pick random i and j.
    # Since 512 is small, picking random i, j is fine.
    # However, we need to ensure we swap an element IN the set with one OUT.
    
    # Let's maintain a list of indices in the set for faster removal?
    # Or just use bit operations.
    
    # Optimization: Precompute the bits that are currently set?
    # It's easier to just pick random i, j in 0..511.
    # If bit i is set and bit j is not set, swap them.
    
    while time.time() < deadline:
        if T < min_T:
            T = 5.0 # Restart with high temp? Or break?
            # Let's just break if we cool down too much and no improvement for a while
            # But for simplicity, let's just let it run until deadline.
            # Actually, restarting with high T is good for exploration.
            # Let's use a simple hill climbing with occasional random restarts of the temperature.
            pass
        
        # Pick random i, j
        i = random.randint(0, 511)
        j = random.randint(0, 511)
        
        if i == j:
            continue
            
        bit_i = 1 << i
        bit_j = 1 << j
        
        # Check if i is in set and j is not
        if (current_int & bit_i) and not (current_int & bit_j):
            new_int = (current_int ^ bit_i) | bit_j
            new_score = _score_bitset(new_int)
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_int = new_int
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_int = current_int
                
                T *= cooling
        
        # Occasionally, if stuck, do a larger perturbation?
        # Or just rely on SA.
        
    return best_int

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_global_int = 0
    best_global_score = -1
    
    # Initializations
    # 1. Random sets
    # 2. Structured sets: e.g., multiples of k
    
    inits = []
    
    # Random inits
    for _ in range(10):
        sample = random.sample(range(512), 32)
        s_int = 0
        for idx in sample:
            s_int |= (1 << idx)
        inits.append(s_int)
        
    # Structured: Consecutive
    s_int = 0
    for i in range(32):
        s_int |= (1 << i)
    inits.append(s_int)
    
    # Structured: Spread out
    s_int = 0
    for i in range(32):
        s_int |= (1 << (i * 16))
    inits.append(s_int)
    
    # Allocate time to each init
    # If we have 12 inits, and 140s total, ~11s each.
    # But we can just run one big SA loop and use the best so far as seed for next?
    # Or just run SA on each init sequentially until deadline.
    
    for init_int in inits:
        if time.time() > deadline:
            break
            
        # Give each init a sub-deadline
        # Or just run until global deadline, but that might bias towards first init.
        # Better: run a fixed number of iterations or time slice per init.
        # Let's do time slicing.
        time_per_init = 10.0
        init_deadline = time.time() + time_per_init
        
        res_int = _local_search(init_int, init_deadline)
        res_score = _score_bitset(res_int)
        
        if res_score > best_global_score:
            best_global_score = res_score
            best_global_int = res_int
            
        # If we found a very high score, maybe spend more time?
        # For now, keep it simple.
        
    # If we have time left, run more SA from the best found
    if time.time() < deadline:
        remaining_deadline = deadline
        res_int = _local_search(best_global_int, remaining_deadline)
        res_score = _score_bitset(res_int)
        if res_score > best_global_score:
            best_global_score = res_score
            best_global_int = res_int
            
    # Convert back to list of integers
    # Bit i corresponds to i+1
    result = []
    for i in range(512):
        if (best_global_int >> i) & 1:
            result.append(i + 1)
            
    # Ensure exactly 32 distinct integers
    # The logic above ensures this, but just in case
    if len(result) != 32:
        # Fallback to a valid set if something went wrong
        return list(range(1, 33))
        
    return result

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(f"Result: {s}")
    print(f"Score: {_score_bitset(0)}") # Just to trigger import
    # Actual score check
    s_int = 0
    for x in s:
        s_int |= (1 << (x-1))
    print(f"Actual Score: {_score_bitset(s_int)}")
