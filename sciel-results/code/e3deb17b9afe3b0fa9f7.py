import time
import math
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

def construct():
    start_time = time.time()
    deadline = 80.0 # 10s margin
    
    N = 256
    # 1 is always included
    result = [1]
    
    # Find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # Map prime to index
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Candidate numbers 2..N
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Objective: maximize sum of n * x_n
    # milp minimizes c @ x. So c = -n
    c = [-n for n in candidates]
    
    # Constraints: For each prime p, sum(x_n for n where p|n) <= 1
    # A matrix shape (num_primes, num_candidates)
    A_rows = []
    A_cols = []
    A_data = []
    
    for j, n in enumerate(candidates):
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                A_rows.append(prime_index[p])
                A_cols.append(j)
                A_data.append(1)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is a prime factor
            if temp in prime_index:
                A_rows.append(prime_index[temp])
                A_cols.append(j)
                A_data.append(1)
            
    A = csc_matrix((A_data, (A_rows, A_cols)), shape=(num_primes, num_candidates))
    
    # Constraints: A x <= 1
    # Lower bound: 0, Upper bound: 1 for each prime constraint
    cons = LinearConstraint(A, lb=0, ub=1)
    
    # Bounds for x: 0 <= x <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer
    integrality = 1
    
    res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
    
    if time.time() > deadline:
        # Fallback to seed strategy if MILP fails or takes too long (unlikely)
        return [1] + [a for a in range(2, N+1) if all(a%d for d in range(2, math.isqrt(a)+1))]

    if not res.success:
        # Fallback
        return [1] + [a for a in range(2, N+1) if all(a%d for d in range(2, math.isqrt(a)+1))]
        
    x = res.x
    selected = []
    for j, n in enumerate(candidates):
        if x[j] > 0.5: # Binary check
            selected.append(int(n))
            
    result.extend(selected)
    return result
