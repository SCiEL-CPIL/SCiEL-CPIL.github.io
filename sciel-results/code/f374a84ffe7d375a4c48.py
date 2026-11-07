"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

# Prime factorization for numbers 2..40
def factorize(n):
    factors = {}
    d = 2
    x = n
    while d * d <= x:
        while x % d == 0:
            factors[d] = factors.get(d, 0) + 1
            x //= d
        d += 1
    if x > 1:
        factors[x] = factors.get(x, 0) + 1
    return factors

# Precompute vectors for 2..40
V = {}
for i in range(2, 41):
    V[i] = factorize(i)

def check_distinct(S):
    """
    Check if all 2^k subset products of S are distinct.
    S is a list of integers.
    Returns True if distinct, False otherwise.
    """
    products = set()
    # Iterative subset product generation
    # Start with {1}
    prods = {1}
    for x in S:
        new_prods = set()
        for p in prods:
            new_prods.add(p * x)
        # If any new product already exists in the global set or within itself, we have a collision
        # But we can just add to a global set
        if new_prods & products:
            return False
        products.update(new_prods)
        prods = new_prods
    return True

def construct():
    start_time = time.time()
    deadline = 50.0  # 50s limit to be safe
    
    # Candidate pool: 2..40
    candidates = list(range(2, 41))
    
    # Heuristic: Primes are "more independent"
    primes = [i for i in candidates if len(V[i]) == 1]
    composites = [i for i in candidates if len(V[i]) > 1]
    
    # We want to maximize the size. Max possible is 16.
    # Let's try to find a set of size 16.
    
    # Strategy:
    # 1. Start with all primes.
    # 2. Try to add composites.
    # 3. If we get stuck, maybe remove a prime to add two composites?
    
    best_set = []
    
    # Initialize with primes
    current_set = list(primes)
    if check_distinct(current_set):
        best_set = list(current_set)
        
    # Try to add composites
    # Order composites by "independence" - maybe those with unique primes or higher powers?
    # Let's just try all composites
    for c in composites:
        if time.time() > deadline:
            break
        new_set = current_set + [c]
        if check_distinct(new_set):
            current_set = new_set
            if len(current_set) > len(best_set):
                best_set = list(current_set)
            if len(current_set) == 16:
                return best_set
                
    # If we didn't reach 16, try some random searches to see if we can swap
    # But 12 is the prime count. 16 is the theoretical max.
    # If greedy failed to reach 16, we might need to swap.
    
    # A known optimal set of 16 for 2..40?
    # Let's try a randomized search to improve best_set
    # We'll do a simple local search: pick a random element to remove, pick a random candidate to add.
    
    max_iterations = 500
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if len(best_set) == 16:
            break
            
        # Copy current best
        cur = list(best_set)
        
        # Try to add a random composite not in cur
        avail = [x for x in composites if x not in cur]
        if not avail:
            continue
            
        added = random.choice(avail)
        new_cur = cur + [added]
        
        if not check_distinct(new_cur):
            # Try to remove a random prime from cur to make space?
            # Or just skip.
            # If we remove a prime, we might be able to add 2 composites?
            # For now, just try direct addition.
            continue
        
        if len(new_cur) > len(best_set):
            best_set = new_cur
            
    return best_set

# EVOLVE-BLOCK-END
