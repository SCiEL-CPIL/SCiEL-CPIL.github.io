"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Start time to ensure we don't exceed budget, though this is O(1)
    start_time = time.time()
    
    # Set A: Upper half, contiguous, sum-free internally (min sum > max)
    # 65 + 66 = 131 > 128
    A = set(range(65, 129))
    
    # Set B: Lower part, weakly sum-free, max sum < 65
    # B = {17, ..., 32}
    # Min sum: 17+18=35. Max sum: 31+32=63.
    # All sums are < 65, so not in A.
    # Is B weakly sum-free?
    # For distinct x,y in B, is x+y in B?
    # Min sum 35. Max element 32.
    # 35 > 32, so x+y > max(B).
    # So x+y not in B.
    # Thus B is weakly sum-free.
    B = set(range(17, 33))
    
    S = A.union(B)
    
    # Convert to list of ints
    result = list(S)
    
    # Ensure types are Python int
    result = [int(x) for x in result]
    
    # Safety check for time (not really needed here)
    if time.time() - start_time > 50:
         return result
         
    return result
# EVOLVE-BLOCK-END
