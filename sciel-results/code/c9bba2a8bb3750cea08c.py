"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import array

def construct():
    # Explicit deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s margin
    
    target = 123457
    max_n = target - 1  # 123456
    
    # Use array of unsigned ints for LPF storage (more memory efficient than list of Python ints)
    lpf = array.array('I', [0]) * (max_n + 1)
    if max_n >= 1:
        lpf[1] = 1
        
    # Sieve to find largest prime factor
    # Complexity O(N log log N)
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                # Update lpf[j] with the current prime i
                # Since we iterate i in increasing order, the last prime factor
                # assigned will be the largest.
                lpf[j] = i
                
    best_max = float('inf')
    best_pair = (1, target - 1)
    
    # Search for optimal a
    # Complexity O(N/2) due to symmetry
    half_target = target // 2
    for a in range(1, half_target + 1):
        # Check time occasionally
        if (a & 0xFF) == 0:
             if time.time() > deadline:
                 break
                 
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur_max = pa if pa > pb else pb
        
        if cur_max < best_max:
            best_max = cur_max
            best_pair = (a, b)
            # Early termination if we reach the theoretical minimum
            if best_max <= 3:
                break
                
    return [best_pair[0], best_pair[1]]
# EVOLVE-BLOCK-END
