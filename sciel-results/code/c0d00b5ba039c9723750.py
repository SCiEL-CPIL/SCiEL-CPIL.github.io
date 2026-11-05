# EVOLVE-BLOCK-START
import math
from fractions import Fraction
import random
import time

def _primes_up_to(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def construct():
    primes = _primes_up_to(1000)
    random.seed(42)
    
    # We will focus on small primes as they are more efficient for covering [1..L]
    # Sum of reciprocals of first k primes:
    # 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47 ~ 1.0
    # So we can use at most ~15 primes.
    
    # Candidate primes: small ones
    candidate_primes = [p for p in primes if p <= 50]
    
    # Find largest subset of candidate_primes with sum(1/p) <= 1
    # Greedy by smallest p (highest value 1/p) is not necessarily best for count, 
    # but for coverage density, small primes are key.
    # Let's just take the first N that fit.
    
    def get_max_subset(prime_list):
        # Sort by prime value (smallest first)
        # This maximizes the "power" of the set
        cost = Fraction(0)
        subset = []
        for p in prime_list:
            if cost + Fraction(1, p) <= 1:
                subset.append(p)
                cost += Fraction(1, p)
            else:
                # Try to see if a smaller prime later could fit? 
                # No, we sorted by value, so later primes are larger (smaller 1/p).
                # So if current doesn't fit, no later one will fit either 
                # IF we are filling greedily. 
                # But we might have skipped some? No, we took all small ones.
                # So this gives the "densest" set of small primes.
                pass
        return subset

    base_set = get_max_subset(candidate_primes)
    
    # It is possible that a slightly different set (e.g. swapping a large prime for a medium one) 
    # allows a better permutation. But let's start with this set.
    # Also, we might want to try different sets.
    
    # Simulation function: given a list of primes in a specific order, 
    # assign residues greedily to cover L+1, and return the final L.
    def simulate_order(order):
        L = 0
        # We need to track coverage to know if L+1 is covered.
        # For efficiency, we can just track the set of covered numbers up to some limit.
        # But L can grow. 
        # Let's use a boolean array up to a reasonable limit, e.g. 5000.
        # If L exceeds this, we cap it or extend.
        limit = 5000
        covered = [False] * (limit + 1)
        
        # Initialize L=0, so 1 is uncovered.
        # Wait, if L=0, we need to cover 1.
        
        for p in order:
            # Target number to cover is L + 1
            target = L + 1
            if target > limit:
                # If we've already covered up to limit, we might have exceeded it.
                # For now, assume we stop at limit.
                break
            
            # Residue that covers target
            r = target % p
            
            # Mark numbers covered by x = r mod p
            # Note: x=0 is not in 1..L usually, but r=0 means p, 2p, etc.
            # The class is {x | x % p == r}.
            # If r == 0, it covers p, 2p, ...
            # If r > 0, it covers r, r+p, ...
            
            start = r
            if start == 0:
                start = p # Smallest positive integer in class
            
            # Update coverage
            # We only care about extending L, so we check from L+1 upwards.
            # But the new class might cover numbers < L+1 too (already covered).
            # We just need to find the new contiguous L.
            
            # Optimization: Only update from L+1 to limit
            # But what if the new class covers L+1, but not L+2, and an OLD class covered L+2?
            # We need to check the full contiguous run from 1.
            # However, since 1..L were already covered, we only need to check L+1, L+2, ...
            
            # Let's mark the new class
            for x in range(start, limit + 1, p):
                covered[x] = True
            
            # Extend L
            while L < limit and covered[L+1]:
                L += 1
                
        return L

    # Simulated Annealing to find best permutation
    best_L = 0
    best_order = []
    
    # Initial solution: sorted by prime
    current_order = sorted(base_set)
    current_L = simulate_order(current_order)
    best_L = current_L
    best_order = current_order[:]
    
    # Annealing parameters
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    max_iters = 5000
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() - start_time > 100: # Keep some time for other tasks
            break
            
        # Pick two random indices to swap
        if len(current_order) < 2:
            break
        i1, i2 = random.sample(range(len(current_order)), 2)
        
        new_order = current_order[:]
        new_order[i1], new_order[i2] = new_order[i2], new_order[i1]
        
        new_L = simulate_order(new_order)
        delta = new_L - current_L
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_order = new_order
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_order = new_order[:]
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Convert best_order to classes
    # We need to re-simulate to get the correct residues for the best order
    classes = []
    L = 0
    limit = 5000
    covered = [False] * (limit + 1)
    
    for p in best_order:
        target = L + 1
        if target > limit:
            break
        r = target % p
        classes.append([p, r])
        
        start = r
        if start == 0:
            start = p
        for x in range(start, limit + 1, p):
            covered[x] = True
            
        while L < limit and covered[L+1]:
            L += 1
            
    return classes

# EVOLVE-BLOCK-END
