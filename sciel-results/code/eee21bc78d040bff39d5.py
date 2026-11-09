import random
import time

def construct():
    best_set = []
    start_time = time.time()
    limit = 512
    max_size = 48
    
    while time.time() - start_time < 55:
        candidates = list(range(1, limit + 1))
        random.shuffle(candidates)
        S = []
        bits_S = 1  # bitset: bit i set if sum i achievable by subset of S
        sum_S = 0
        n = 0
        
        for x in candidates:
            if n >= max_size:
                break
            
            valid = True
            
            # Case 1: x is mean of subset of S (size >= 2)
            if n >= 2:
                for k in range(2, n + 1):
                    target = k * x
                    if target > sum_S:
                        break
                    if (bits_S >> target) & 1:
                        valid = False
                        break
            
            if valid:
                # Case 2: some a in S is mean of subset of S U {x} containing x
                for a in S:
                    # sum(T) = k*a, x in T, a in T, T\{x,a} subset of S\{a,x}
                    # sum(T\{x,a}) = k*a - x - a = (k-1)*a - x
                    # Let k' = k-1 >= 1 (since k>=2). If k'=1, sum=0, need a=x, impossible.
                    # So k' >= 2, target = k'*a - x, must be subset sum of S\{a} with |U|=k'-1 elements
                    # Quick bound: target must be in [0, sum_S - a]
                    max_without_a = sum_S - a
                    found = False
                    for k_prime in range(2, n + 1):
                        target = k_prime * a - x
                        if target < 0:
                            continue
                        if target > max_without_a:
                            break  # k_prime increasing, target increasing
                        # Check if target is achievable by S\{a}
                        # Conservative: check if target is in subset sums of S
                        if not ((bits_S >> target) & 1):
                            continue
                        # target is achievable by S. Might use a.
                        # If target < a, it cannot use a (all positive), so valid for S\{a}
                        if target < a:
                            found = True
                            break
                        # If target >= a, it might use a. We need to verify.
                        # For speed, assume it uses a (conservative rejection)
                        # This may miss valid additions but keeps us safe
                        found = True
                        break
                    if found:
                        valid = False
                        break
            
            if valid:
                S.append(x)
                sum_S += x
                bits_S |= bits_S << x
                n += 1
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == max_size:
                break
    
    return best_set
