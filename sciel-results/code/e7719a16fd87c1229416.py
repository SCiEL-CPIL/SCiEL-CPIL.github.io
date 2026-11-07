import math
import time
import sys

def construct():
    start = time.time()
    # Safety margin: Evaluator kills at 90s. We aim for 85s hard stop.
    deadline = start + 85.0
    N = 256

    # 1. Sieve of Eratosthenes for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # 2. Precompute prime factors for each number 1..N
    # Using a list of sets for fast intersection/union
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                prime_factors[i].add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            prime_factors[i].add(x)
    prime_factors[1] = set()
    
    best_set = None
    
    # 3. Attempt Exact Solution via MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Constraint Matrix A: num_primes x (N+1)
        # A[p_idx, n] = 1 if prime p_idx divides n
        A = np.zeros((num_primes, N + 1), dtype=np.int8)
        for idx, p in enumerate(primes):
            for multiple in range(p, N + 1, p):
                A[idx, multiple] = 1
        
        # Objective: Minimize -sum(n * x_n)
        c = np.zeros(N + 1, dtype=np.float64)
        for n in range(1, N + 1):
            c[n] = -float(n)
            
        # Constraints: 0 <= sum(x_n for n multiple of p) <= 1
        cons = LinearConstraint(A, lb=0, ub=1)
        
        # Bounds: 0 <= x_n <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: All variables are integer
        integrality = np.ones(N + 1, dtype=np.int8)
        
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.success:
            selected = []
            for n in range(1, N + 1):
                if res.x[n] > 0.5:
                    selected.append(int(n))
            if selected:
                # Verify the solution
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if math.gcd(selected[i], selected[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_set = set(selected)
    except Exception:
        best_set = None

    # 4. Fallback: Enhanced Local Search
    if best_set is None:
        # Initialization: Greedy approach
        # Sort numbers by value descending
        candidates = list(range(N, 0, -1))
        best_set = set()
        used_primes = set()
        
        for n in candidates:
            if n == 1:
                best_set.add(1)
                continue
            pf = prime_factors[n]
            if len(pf & used_primes) == 0:
                best_set.add(n)
                used_primes |= pf
        
        best_sum = sum(best_set)
        
        # Local Search Loop
        # We try to improve the sum by swapping elements
        # Moves:
        # 1. Add a number (if possible)
        # 2. Swap 1 out for 1 in
        # 3. Swap 1 out for 2 in
        # 4. Swap 2 out for 1 in
        # 5. Swap 2 out for 2 in
        
        unchanged = 0
        while unchanged < 100 and time.time() < deadline - 2.0:
            improved = False
            
            # Move 1: Add
            # Try to add the largest possible number
            for n in range(N, 0, -1):
                if n not in best_set and len(prime_factors[n] & used_primes) == 0:
                    best_set.add(n)
                    used_primes |= prime_factors[n]
                    best_sum += n
                    improved = True
                    break
            
            # Move 2: 1-for-1 Swap
            if not improved:
                for n_out in list(best_set):
                    if n_out == 1:
                        continue
                    freed = prime_factors[n_out]
                    remaining = used_primes - freed
                    for n_in in range(N, 0, -1):
                        if n_in not in best_set and len(prime_factors[n_in] & remaining) == 0:
                            if n_in > n_out:
                                best_set.remove(n_out)
                                best_set.add(n_in)
                                used_primes = remaining | prime_factors[n_in]
                                best_sum += n_in - n_out
                                improved = True
                                break
                    if improved:
                        break
            
            # Move 3: 1-for-2 Swap (Remove 1, Add 2)
            if not improved:
                for n_out in list(best_set):
                    if n_out == 1:
                        continue
                    freed = prime_factors[n_out]
                    remaining = used_primes - freed
                    # Try to add two numbers n_in1, n_in2
                    # n_in1 > n_out is required for improvement potentially, or sum(n_in1, n_in2) > n_out
                    for n_in1 in range(N, 0, -1):
                        if n_in1 not in best_set and len(prime_factors[n_in1] & remaining) == 0:
                            rem2 = remaining | prime_factors[n_in1]
                            for n_in2 in range(n_in1 - 1, 0, -1):
                                if n_in2 not in best_set and n_in2 != n_in1 and len(prime_factors[n_in2] & rem2) == 0:
                                    if n_in1 + n_in2 > n_out:
                                        best_set.remove(n_out)
                                        best_set.add(n_in1)
                                        best_set.add(n_in2)
                                        used_primes = rem2 | prime_factors[n_in2]
                                        best_sum += n_in1 + n_in2 - n_out
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break

            # Move 4: 2-for-1 Swap (Remove 2, Add 1)
            if not improved:
                lst = [x for x in best_set if x != 1]
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        a, b = lst[i], lst[j]
                        freed = prime_factors[a] | prime_factors[b]
                        remaining = used_primes - freed
                        for n_in in range(N, 0, -1):
                            if n_in not in best_set and len(prime_factors[n_in] & remaining) == 0:
                                if n_in > a + b:
                                    best_set.remove(a)
                                    best_set.remove(b)
                                    best_set.add(n_in)
                                    used_primes = remaining | prime_factors[n_in]
                                    best_sum += n_in - a - b
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

            # Move 5: 2-for-2 Swap
            if not improved:
                lst = [x for x in best_set if x != 1]
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        a, b = lst[i], lst[j]
                        freed = prime_factors[a] | prime_factors[b]
                        remaining = used_primes - freed
                        for n_in1 in range(N, 0, -1):
                            if n_in1 not in best_set and len(prime_factors[n_in1] & remaining) == 0:
                                rem2 = remaining | prime_factors[n_in1]
                                for n_in2 in range(n_in1 - 1, 0, -1):
                                    if n_in2 not in best_set and n_in2 != n_in1 and len(prime_factors[n_in2] & rem2) == 0:
                                        if n_in1 + n_in2 > a + b:
                                            best_set.remove(a)
                                            best_set.remove(b)
                                            best_set.add(n_in1)
                                            best_set.add(n_in2)
                                            used_primes = rem2 | prime_factors[n_in2]
                                            best_sum += n_in1 + n_in2 - a - b
                                            improved = True
                                            break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break

            if not improved:
                unchanged += 1
            else:
                unchanged = 0

    # Final Validation and Conversion
    result = [int(x) for x in best_set]
    
    # Ensure distinct
    result = list(set(result))
    
    # Final Check
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert math.gcd(result[i], result[j]) == 1, f"Non-coprime: {result[i]}, {result[j]}"
            assert 1 <= result[i] <= N
            assert isinstance(result[i], int)
            
    return result
