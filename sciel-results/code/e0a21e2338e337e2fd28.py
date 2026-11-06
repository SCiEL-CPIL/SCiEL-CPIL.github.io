import time
from math import isqrt

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85 seconds to be safe
    
    n = 12
    max_val = 10000
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    def is_p_smooth(num, primes_set):
        for p in primes_set:
            while num % p == 0:
                num //= p
        return num == 1

    def find_clique(primes_set):
        def backtrack(current_set, start_idx):
            if len(current_set) == n:
                return current_set
            if time.time() > deadline:
                return None
                
            if max_val - start_idx + 1 < n - len(current_set):
                return None

            for x in range(start_idx, max_val + 1):
                compatible = True
                for y in current_set:
                    if not is_p_smooth(x + y, primes_set):
                        compatible = False
                        break
                
                if compatible:
                    res = backtrack(current_set + [x], x + 1)
                    if res is not None:
                        return res
            
            return None

        return backtrack([], 1)

    # Try prime sets in order of size
    # Size 1
    for p in small_primes:
        if time.time() > deadline: 
            break
        P = frozenset([p])
        clique = find_clique(P)
        if clique:
            return clique

    # Size 2
    for i in range(len(small_primes)):
        if time.time() > deadline: 
            break
        for j in range(i+1, len(small_primes)):
            P = frozenset([small_primes[i], small_primes[j]])
            clique = find_clique(P)
            if clique:
                return clique

    # Size 3
    for i in range(len(small_primes)):
        if time.time() > deadline: 
            break
        for j in range(i+1, len(small_primes)):
            if time.time() > deadline: 
                break
            for k in range(j+1, len(small_primes)):
                P = frozenset([small_primes[i], small_primes[j], small_primes[k]])
                clique = find_clique(P)
                if clique:
                    return clique

    # Fallback
    return list(range(1, 13))
# EVOLVE-BLOCK-END
