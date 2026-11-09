"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Helper to check if a set is valid
    def is_valid_set(S):
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j + 1, m):
                    c = S_list[k]
                    # Check a | b*c
                    if (b * c) % a == 0:
                        return False
        return True

    # Helper to check if x can be added to S
    def can_add(S, x):
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            b = S_list[i]
            for j in range(i + 1, m):
                c = S_list[j]
                # New triples involving x:
                # 1. x is the divisor: x | b*c
                if (b * c) % x == 0:
                    return False
                # 2. b is the divisor: b | x*c
                if (x * c) % b == 0:
                    return False
                # 3. c is the divisor: c | x*b
                if (x * b) % c == 0:
                    return False
        return True

    best_set = []
    
    # Strategy 1: Descending order (Large numbers are safer as divisors)
    order1 = sorted(candidates, reverse=True)
    S = []
    for x in order1:
        if time.time() > deadline - 5:
            break
        if can_add(S, x):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S.copy()

    # Strategy 2: Ascending order
    order2 = sorted(candidates)
    S = []
    for x in order2:
        if time.time() > deadline - 5:
            break
        if can_add(S, x):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S.copy()

    # Strategy 3: Randomized restarts
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        # Mix of descending and random
        order = list(candidates)
        if restart % 2 == 0:
            order.sort(key=lambda x: -x + random.random() * 50)
        else:
            random.shuffle(order)
        
        S = []
        for x in order:
            if time.time() > deadline - 10:
                break
            if can_add(S, x):
                S.append(x)
        if len(S) > len(best_set):
            best_set = S.copy()
            
    # Local Improvement: Try to add any element not in best_set
    # This is O(N * |S|^2) per check, which is fine.
    # We will try to improve the best_set found.
    current_set = set(best_set)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 2:
            break
        # Try to add any missing element
        for x in candidates:
            if x in current_set:
                continue
            if can_add(current_set, x):
                current_set.add(x)
                improved = True
                break # Restart loop to allow more additions
        
        # If no single addition works, try swaps?
        # Swaps don't increase size, so we skip unless we are stuck and want to explore different local minima.
        # But since we only care about size, and we already tried multiple greedy strategies, 
        # adding elements is the primary goal.
        
    best_set = list(current_set)
    
    # Final Verification
    if not is_valid_set(best_set):
        # Fallback to primes if something went wrong (should not happen)
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
        
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
