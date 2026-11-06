import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin for 150s limit
    
    # 1. Precomputation
    # Primes below 512
    primes = []
    is_prime = [True] * 512
    is_prime[0] = is_prime[1] = False
    for i in range(2, 512):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 512, i):
                if j < 512:
                    is_prime[j] = False
    
    # Weights
    weights = {a: 1.0 / (512 - a) for a in range(1, 512)}
    
    # Prime factors for each number 1..511
    prime_factors = [[] for _ in range(512)]
    for p in primes:
        for multiple in range(p, 512, p):
            prime_factors[multiple].append(p)
            
    # Initial Set: All primes
    S = set(primes)
    
    # Helper to check if a can replace its prime factors
    # Returns gain if valid and profitable, else -inf
    def try_swap_in(a):
        if a in S:
            return -float('inf')
        
        pfs = prime_factors[a]
        if not pfs:
            return -float('inf') # a=1? no, 1 has no factors but gcd(1,x)=1. 
            # Wait, 1 is coprime to everything. 
            # If we add 1, we don't remove anything. Gain = 1/511.
            # But 1 is not a prime. 
            # The problem says "integers a in 1..511".
            # Is 1 allowed? Yes.
            # But 1 doesn't conflict with anything.
            # So we should always include 1?
            # Yes, 1 is coprime to all. And 1/511 > 0.
            # So 1 should always be in the set.
            
        # Check if all prime factors are in S
        for p in pfs:
            if p not in S:
                return -float('inf')
                
        # Check coprimality with rest of S
        for x in S:
            if x in pfs:
                continue
            if gcd(a, x) != 1:
                return -float('inf')
                
        # Calculate gain
        gain = weights[a]
        for p in pfs:
            gain -= weights[p]
            
        return gain

    # Add 1 if not present (it is not a prime, so not in S initially)
    # 1 is coprime to all.
    S.add(1)
    
    # Iterative Absorption
    # We iterate until no improvement
    # To avoid infinite loops or slow convergence, we limit iterations
    max_iterations = 1000
    for _ in range(max_iterations):
        if time.time() > deadline - 10:
            break
            
        improved = False
        # Try to add composites in descending order of weight (ascending 512-a)
        # i.e., a from 511 down to 2
        for a in range(511, 1, -1):
            gain = try_swap_in(a)
            if gain > 1e-9:
                # Perform swap
                for p in prime_factors[a]:
                    S.remove(p)
                S.add(a)
                improved = True
                # After a swap, the set S has changed.
                # We could break and restart, or continue.
                # Continuing might pick up other opportunities, but the 'valid' check
                # for subsequent 'a's might be invalid if they depended on removed primes.
                # However, try_swap_in checks current S.
                # So it's safe to continue.
                
        if not improved:
            break
            
    # Local Search: Single swap
    # Try to replace x in S with y not in S
    if time.time() < deadline - 5:
        # Precompute coprime masks or just use gcd
        # Since 511 is small, O(N^2) is fine.
        S_list = list(S)
        for i, x in enumerate(S_list):
            if time.time() > deadline - 2:
                break
            for y in range(1, 512):
                if y in S:
                    continue
                # Check if y is coprime to S \ {x}
                valid = True
                for z in S_list:
                    if z == x:
                        continue
                    if gcd(y, z) != 1:
                        valid = False
                        break
                if not valid:
                    continue
                gain = weights[y] - weights[x]
                if gain > 1e-9:
                    S.remove(x)
                    S.add(y)
                    S_list[i] = y
                    # Restart local search? Or just continue.
                    # Restarting is safer.
                    break
            else:
                continue
            break # Restarted
            
    return list(S)
