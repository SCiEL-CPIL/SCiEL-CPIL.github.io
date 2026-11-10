# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # For each number 1..256, find its prime factors
    prime_factors = {}
    for n in range(1, MAX_N + 1):
        if n == 1:
            prime_factors[n] = []
        else:
            factors = []
            temp = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    factors.append(p)
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                factors.append(temp)
            prime_factors[n] = factors
    
    # Use ILP via scipy.optimize.milp
    # Variables: x_n for n in 1..256, binary
    # Objective: maximize sum(n * x_n)
    # Constraints: For each prime p, sum of x_n for n divisible by p <= 1
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Build constraint matrix: one row per prime, one column per number 1..256
        # A[p_idx, n-1] = 1 if p divides n, else 0
        n_primes = len(primes)
        n_vars = MAX_N
        
        # Build A matrix
        A = np.zeros((n_primes, n_vars), dtype=np.int8)
        for p_idx, p in enumerate(primes):
            for multiple in range(p, MAX_N + 1, p):
                A[p_idx, multiple - 1] = 1
        
        # Objective: maximize sum(n * x_n) => minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=np.float64)
        
        # Constraints: A @ x <= 1 for all primes
        constraint = LinearConstraint(A, lb=0, ub=1)
        
        # Bounds: 0 <= x_n <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: all variables are integer (binary)
        integrality = np.ones(n_vars)
        
        result = milp(c=c, constraints=constraint, bounds=bounds, integrality=integrality)
        
        if result.success:
            selected = [i + 1 for i in range(n_vars) if result.x[i] > 0.5]
            # Verify pairwise coprimality
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if gcd(selected[i], selected[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Ensure the result is a list of Python ints
                return [int(x) for x in sorted(selected)]
        else:
            # Fallback to greedy if ILP fails
            pass
    except Exception:
        pass
    
    # Fallback: Greedy approach
    S = [1]
    for n in range(256, 1, -1):
        if all(gcd(n, x) == 1 for x in S):
            S.append(n)
    
    # Ensure the result is a list of Python ints
    return [int(x) for x in sorted(S)]
# EVOLVE-BLOCK-END
