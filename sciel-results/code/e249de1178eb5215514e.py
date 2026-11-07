import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    
    best_set = []
    
    def is_valid_addition(a, current_set):
        # Check if adding 'a' to 'current_set' violates any condition
        # Conditions for distinct x, y, z in new_set: x does not divide y*z
        # New triples involving 'a':
        # 1. a, b, c where b, c in current_set
        # 2. b, a, c where b, c in current_set (same as 1 due to symmetry of product, but roles of a,b,c in divisibility matter)
        # Specifically:
        # For all b, c in current_set, b != c:
        #   - a does not divide b * c
        #   - b does not divide a * c
        #   - c does not divide a * b
        
        # Optimization: 
        # If current_set is empty or size 1, no triples exist yet.
        if len(current_set) < 2:
            return True
            
        # We can break early if a violation is found
        cs = current_set
        n = len(cs)
        for i in range(n):
            b = cs[i]
            for j in range(i + 1, n):
                c = cs[j]
                # Check a | b*c
                if (b * c) % a == 0:
                    return False
                # Check b | a*c
                if (a * c) % b == 0:
                    return False
                # Check c | a*b
                if (a * b) % c == 0:
                    return False
        return True

    def greedy_search(order):
        s = []
        for a in order:
            if is_valid_addition(a, s):
                s.append(a)
        return s

    # Define different sorting strategies
    strategies = []
    
    # 1. Descending value
    strategies.append(sorted(candidates, reverse=True))
    
    # 2. Ascending value
    strategies.append(sorted(candidates))
    
    # 3. Primes first (approximated by low number of divisors)
    # Precompute divisor counts
    div_counts = [0] * (N + 1)
    for i in range(2, N + 1):
        for j in range(i, N + 1, i):
            div_counts[j] += 1
    strategies.append(sorted(candidates, key=lambda x: div_counts[x]))
    
    # 4. Square-free first?
    def is_square_free(n):
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True
    sf_candidates = [x for x in candidates if is_square_free(x)]
    non_sf_candidates = [x for x in candidates if not is_square_free(x)]
    strategies.append(sorted(sf_candidates, reverse=True) + sorted(non_sf_candidates, reverse=True))
    
    # Run deterministic strategies
    for order in strategies:
        s = greedy_search(order)
        if len(s) > len(best_set):
            best_set = s
            
    # Randomized restarts if time remains
    # We can try shuffling the candidate list or shuffling the order of addition
    # A simple random order might find different local optima
    while time.time() < deadline:
        # Random shuffle
        order = candidates[:]
        random.shuffle(order)
        s = greedy_search(order)
        if len(s) > len(best_set):
            best_set = s
        # Optional: Local search? 
        # Given N=256, greedy is likely close to optimal for this specific constraint.
        # The prime set is 54. The upper half is ~128 but has conflicts.
        # A good greedy might find 100+.

    return best_set
