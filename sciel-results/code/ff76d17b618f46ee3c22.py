import time
import math

def construct():
    """
    Returns a list of distinct ints from 1..256, pairwise coprime, maximizing sum.
    Uses MILP to find the optimal set.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    N = 256
    
    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Fallback solution: Greedy largest coprime
    # This is a valid solution, used if MILP fails or times out
    def generate_fallback():
        sol = [1]
        for i in range(N, 1, -1):
            # Check if i is coprime to all in sol
            ok = True
            for s in sol:
                if math.gcd(i, s) != 1:
                    ok = False
                    break
            if ok:
                sol.append(i)
        return sol

    fallback = generate_fallback()
    
    # Try MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        n_vars = N
        n_constraints = len(primes)
        
        # Objective: maximize sum(i * x_i) -> minimize -sum(i * x_i)
        c = np.array([-i for i in range(1, N + 1)])
        
        # Constraints: For each prime p, sum of x_i for i divisible by p <= 1
        A = np.zeros((n_constraints, n_vars))
        for idx_p, p in enumerate(primes):
            for i in range(1, N + 1):
                if i % p == 0:
                    A[idx_p, i - 1] = 1
        
        b = np.ones(n_constraints)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        # Integrality: all variables are integer
        integrality = np.ones(n_vars)
        
        res = milp(c=c, constraints=LinearConstraint(A, ub=b), bounds=bounds, integrality=integrality)
        
        if res.status == 0:
            x = res.x
            result = [int(i + 1) for i in range(N) if x[i] > 0.5]
            # Ensure 1 is in the result? 
            # 1 has no prime factors, so it doesn't appear in any constraint.
            # Its coefficient is -1 (minimization), so it will be set to 1 to minimize (maximize sum).
            # So 1 should be included.
            return result
        else:
            # If MILP fails, return fallback
            return fallback
            
    except Exception:
        # If scipy is not available or error occurs, return fallback
        return fallback
