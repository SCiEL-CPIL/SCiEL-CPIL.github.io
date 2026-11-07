import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time
from math import gcd

def construct():
    n = 256
    start_time = time.time()
    deadline = 85  # 85 seconds to leave margin
    
    # Generate primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # Objective: Maximize sum of selected integers
    # Variables x_0 ... x_255 correspond to integers 1 ... 256
    c = np.arange(1, n + 1, dtype=float)
    
    # Constraints: For each prime p, sum of x_i for i divisible by p <= 1
    # A_ub has shape (num_primes, n)
    A_ub = np.zeros((num_primes, n), dtype=float)
    b_ub = np.ones(num_primes, dtype=float)
    
    for j, p in enumerate(primes):
        for i in range(n):
            if (i + 1) % p == 0:
                A_ub[j, i] = 1.0
                
    # Bounds: x_i in [0, 1]
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    
    # Integrality: all variables are integers
    integrality = np.ones(n)
    
    # Solve
    res = milp(c=-c, # Minimize -sum to maximize sum
               constraints=LinearConstraint(A_ub, ub=b_ub),
               integrality=integrality,
               bounds=bounds)
    
    if not res.success:
        # Fallback: Largest power of prime strategy
        # This is a valid solution, just maybe not optimal
        result = [1]
        for p in primes:
            val = p
            while val * p <= n:
                val *= p
            result.append(val)
        return result
        
    x = res.x
    selected = []
    for i in range(n):
        if x[i] > 0.5:
            selected.append(int(i + 1))
    
    # Verify and try to improve with local search
    if time.time() - start_time > deadline:
        return selected
    
    # Local search: try to add numbers or swap
    selected_set = set(selected)
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try adding each number not in selected
        for num in range(1, n + 1):
            if num in selected_set:
                continue
            # Check if num is coprime with all in selected
            if all(gcd(num, s) == 1 for s in selected_set):
                selected_set.add(num)
                selected = list(selected_set)
                improved = True
                break
        # Try swapping: remove one, add two
        if not improved:
            for i, s1 in enumerate(list(selected_set)):
                if (time.time() - start_time) > deadline:
                    break
                for num in range(1, n + 1):
                    if num in selected_set:
                        continue
                    # Check if num is coprime with all in selected except s1
                    if all(gcd(num, s) == 1 for s in selected_set if s != s1):
                        # Check if removing s1 and adding num improves sum
                        if num > s1:
                            selected_set.remove(s1)
                            selected_set.add(num)
                            selected = list(selected_set)
                            improved = True
                            break
                if improved:
                    break
    
    return list(selected_set)
