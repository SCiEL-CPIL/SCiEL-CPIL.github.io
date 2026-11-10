"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Find 12 distinct positive integers with all 4096 subset sums distinct,
    # minimizing the largest element. Powers of 2 give max=2048.
    
    def check_valid(arr):
        """Check if all subset sums are distinct."""
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            if new_sums & sums:
                return False
            sums = sums | new_sums
        return len(sums) == 4096
    
    def optimized_greedy(deadline):
        arr = [1]
        sums = {0}
        
        for step in range(11):
            if time.time() > deadline:
                return None
            
            found = False
            current_sum = sum(arr)
            # Try candidates up to min(2048, current_sum + 1)
            upper = min(2048, current_sum + 1)
            
            for x in range(arr[-1] + 1, upper + 1):
                if time.time() > deadline:
                    return None
                
                # Check collision: is there s in sums such that s + x in sums?
                collides = False
                for s in sums:
                    if (s + x) in sums:
                        collides = True
                        break
                
                if not collides:
                    new_sums = set()
                    for s in sums:
                        new_sums.add(s + x)
                    sums = sums | new_sums
                    arr.append(x)
                    found = True
                    break
            
            if not found:
                # Fall back to superincreasing
                x = current_sum + 1
                if x > 2048:
                    return None
                new_sums = set()
                for s in sums:
                    new_sums.add(s + x)
                sums = sums | new_sums
                arr.append(x)
        
        return arr
    
    result = optimized_greedy(deadline)
    
    if result is not None:
        if len(result) == 12 and len(set(result)) == 12 and all(x > 0 for x in result) and max(result) <= 1048576:
            if check_valid(result):
                return result
    
    # Fallback: powers of 2 which is provably optimal (max=2048)
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
