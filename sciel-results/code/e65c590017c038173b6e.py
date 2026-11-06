import time
from math import gcd, lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    def get_val(n, p):
        c = 0
        while n % p == 0:
            c += 1
            n //= p
        return c
    
    # Precompute valuations
    valuations = {p: {} for p in primes}
    for x in range(1, 129):
        for p in primes:
            valuations[p][x] = get_val(x, p)
            
    S = list(range(1, 129))
    
    # Greedy Pruning
    while time.time() < deadline:
        violated = False
        # Find a prime with |L_p| > 1
        target_p = None
        target_L = []
        
        # Sort primes by number of violations to handle most constrained first?
        # Or just iterate
        for p in primes:
            if not S:
                break
            # Find max valuation in S for p
            max_v = max(valuations[p][x] for x in S)
            H = [x for x in S if valuations[p][x] == max_v]
            L = [x for x in S if valuations[p][x] < max_v]
            
            if len(L) > 1:
                violated = True
                target_p = p
                target_L = L
                break
        
        if not violated:
            break
            
        # Choose an element to remove from target_L
        # Heuristic: remove the one that is "most low" for other primes
        # i.e., in L_q for many q
        best_x = None
        best_score = -1
        
        # Sample if list is large
        candidates = target_L
        if len(candidates) > 20:
            import random
            random.seed(42)
            candidates = random.sample(candidates, 20)
            
        for x in candidates:
            score = 0
            for q in primes:
                if q == target_p:
                    continue
                # Check if x is in L_q
                # Need max_v for q in current S
                # This is expensive. Approximate:
                # If valuations[q][x] is 0 and most others are >0, it's likely in L.
                # Let's just count how many primes q have valuations[q][x] < median?
                # Simpler: just remove the first one?
                # Let's try removing the one with smallest sum of valuations (most "deficient")
                pass
            
            # Simpler heuristic: Remove x with minimum valuations[target_p][x]
            # If tie, minimum sum of all valuations
            curr_score = valuations[target_p][x]
            if curr_score < best_score or best_x is None:
                best_score = curr_score
                best_x = x
                
        S.remove(best_x)
        
    return S
