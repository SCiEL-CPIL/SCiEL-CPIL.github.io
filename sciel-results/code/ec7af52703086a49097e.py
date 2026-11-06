import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    primes.sort()
    
    # Global best result
    best_L = 0
    best_classes = []
    
    def find_new_L(L, classes, new_p, new_r):
        k = L + 1
        while k <= 100000:
            if k % new_p != new_r:
                covered_by_old = False
                for p, r in classes:
                    if k % p == r:
                        covered_by_old = True
                        break
                if not covered_by_old:
                    return k - 1
            k += 1
        return 100000

    def search(used_primes, budget, L, classes):
        nonlocal best_L, best_classes
        
        if time.time() > deadline:
            return
            
        if L > best_L:
            best_L = L
            best_classes = classes.copy()
            
        if L >= 100000:
            return
            
        n = L + 1
        
        # Get unused primes, sorted
        unused = [p for p in primes if p not in used_primes]
        
        # Heuristic: Only consider the smallest 20 unused primes
        # This drastically reduces branching factor
        candidates = unused[:20]
        
        for p in candidates:
            cost = Fraction(1, p)
            if budget < cost:
                continue
                
            r = n % p
            
            new_L = find_new_L(L, classes, p, r)
            
            if new_L > L:
                new_used = used_primes | {p}
                new_budget = budget - cost
                new_classes = classes + [[p, r]]
                
                search(new_used, new_budget, new_L, new_classes)
                
                if time.time() > deadline:
                    return

    search(set(), Fraction(1), 0, [])
    
    return best_classes

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will wrap it in the required format.
# EVOLVE-BLOCK-END
