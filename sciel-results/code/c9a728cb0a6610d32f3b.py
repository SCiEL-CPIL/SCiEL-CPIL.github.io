import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 256
    
    # 1. Generate primes up to N
    def _primes_up_to(n):
        if n < 2:
            return []
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:n + 1:i] = bytearray(len(sieve[i * i:n + 1:i]))
        return [i for i in range(2, n + 1) if sieve[i]]
        
    primes = _primes_up_to(N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Precompute prime factors for each number 1..N
    # factors[x] = list of prime indices dividing x
    factors = [[] for _ in range(N + 1)]
    for x in range(2, N + 1):
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors[x].append(prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors[x].append(prime_to_idx[temp])
            
    # 3. Attempt to solve using ILP (scipy.optimize.milp)
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Variables y_1 ... y_N (index 0 to N-1 corresponding to 1..N)
        # Note: y_0 corresponds to number 1. Number 1 has no prime factors.
        # It is always optimal to include 1.
        
        c = np.array([-float(i) for i in range(1, N + 1)]) # Minimize -sum
        
        # Constraints: For each prime, sum of y_x for x divisible by prime <= 1
        # A is (num_primes, N)
        # We can build it sparse or dense. N=256, P=54. Dense is fine.
        A = np.zeros((num_primes, N), dtype=float)
        for x in range(1, N + 1):
            idx = x - 1
            for p_idx in factors[x]:
                A[p_idx, idx] = 1.0
                
        b_lower = np.zeros(num_primes)
        b_upper = np.ones(num_primes)
        
        constraints = LinearConstraint(A, b_lower, b_upper)
        
        # Bounds: 0 <= y_i <= 1
        lb = np.zeros(N)
        ub = np.ones(N)
        bounds = Bounds(lb, ub)
        
        # Integrality: All variables are integers
        integrality = np.ones(N)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Success
            sol = res.x
            # Convert to list of ints
            result = []
            for i in range(N):
                if sol[i] > 0.5: # Threshold for binary
                    result.append(i + 1)
            
            # Verify
            if _verify(result, N):
                return result
    except Exception:
        pass
        
    # 4. Fallback: Greedy + Local Search
    # This is a robust heuristic if ILP fails.
    
    # Initialize with empty set
    selected = set()
    used_primes = set()
    
    # Greedy: Sort by value descending
    candidates = list(range(N, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        # Check if x conflicts
        conflict = False
        for p_idx in factors[x]:
            if p_idx in used_primes:
                conflict = True
                break
        if not conflict:
            selected.add(x)
            for p_idx in factors[x]:
                used_primes.add(p_idx)
                
    # Local Search: 1-swap
    # Try to swap one selected number with one unselected number
    # This is expensive if done naively. 
    # Better: Iterate through unselected numbers, see if we can replace a subset of selected numbers?
    # Simpler: Just try replacing one selected number 's' with one unselected 'u'
    # if u > s and (primes(u) - primes(s)) doesn't conflict with rest.
    
    # Actually, a better local search for this specific problem:
    # Try to add any unselected number by removing necessary conflicting numbers.
    # If gain > loss, do it.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_sum = sum(selected)
        
        # Try to improve by adding one number u and removing conflicting selected numbers
        for u in range(N, 0, -1):
            if u in selected:
                continue
            
            # Find which selected numbers conflict with u
            to_remove = []
            for s in selected:
                # Check if s and u share a prime
                common = set(factors[s]) & set(factors[u])
                if common:
                    to_remove.append(s)
            
            # Calculate change
            gain = u - sum(to_remove)
            if gain > 0:
                # Apply change
                for s in to_remove:
                    selected.remove(s)
                selected.add(u)
                improved = True
                break # Restart loop as set changed
                
    result = list(selected)
    
    if not result:
        result = [1]
        
    return [int(x) for x in result]

def _verify(result, N):
    if not result:
        return False
    for x in result:
        if not isinstance(x, int) or isinstance(x, bool) or x < 1 or x > N:
            return False
    if len(result) != len(set(result)):
        return False
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                return False
    return True
