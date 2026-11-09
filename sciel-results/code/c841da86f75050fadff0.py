"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We aim to finish well before.
    deadline = start_time + 50
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # 1. Precompute prime factors for each number 1..N
    # prime_factors[i] is a set of prime factors of i
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            # i is prime
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
                
    # Get all primes <= N
    primes = [i for i in range(2, N + 1) if prime_factors[i] == {i}]
    prime_index = {p: idx for idx, p in enumerate(primes)}
    num_primes = len(primes)
    num_vars = N
    
    # 2. Construct MILP
    # Objective: Maximize sum(i * x_i) -> Minimize sum(-i * x_i)
    c = np.array(numbers, dtype=float)
    
    # Constraints: For each prime p, sum of x_i for i divisible by p <= 1
    # A_ub is (num_primes, num_vars)
    # Using sparse matrix for efficiency, though dense is likely fine for 55x256
    rows = []
    cols = []
    vals = []
    
    for i, num in enumerate(numbers):
        for p in prime_factors[num]:
            if p in prime_index:
                rows.append(prime_index[p])
                cols.append(i)
                vals.append(1.0)
                
    A_ub_sparse = csr_matrix((vals, (rows, cols)), shape=(num_primes, num_vars))
    b_ub = np.ones(num_primes)
    
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(num_vars)
    
    # 3. Solve MILP
    best_result = None
    
    try:
        res = milp(
            c=-c,
            constraints=LinearConstraint(A_ub_sparse, ub=b_ub),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            x = res.x
            # Extract selected numbers
            selected = [int(num) for num, xi in zip(numbers, x) if xi > 0.5]
            
            # Verify solution validity
            if len(selected) > 0:
                # Check distinctness
                if len(set(selected)) == len(selected):
                    # Check range
                    if all(1 <= num <= N for num in selected):
                        # Check pairwise coprime
                        is_valid = True
                        for i in range(len(selected)):
                            for j in range(i+1, len(selected)):
                                if math.gcd(selected[i], selected[j]) != 1:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if is_valid:
                            best_result = selected
                            
    except Exception as e:
        # If MILP fails, fallback to heuristic
        pass

    # 4. Fallback Heuristic (from inspiration)
    if best_result is None or time.time() > deadline - 5:
        # Simple valid solution: 1 + largest prime power for each prime
        # This is always valid and pairwise coprime
        fallback = [1]
        for p in primes:
            power = p
            while power * p <= N:
                power *= p
            fallback.append(power)
        
        # Ensure unique and sorted? Not required, but good for consistency.
        # The task requires distinct integers.
        # prime powers for distinct primes are distinct.
        # 1 is distinct from any prime power >= 2.
        best_result = fallback
        
    # Final check: Ensure we return a list of Python ints
    if best_result is None:
        # Absolute worst case
        best_result = [1]
        
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
