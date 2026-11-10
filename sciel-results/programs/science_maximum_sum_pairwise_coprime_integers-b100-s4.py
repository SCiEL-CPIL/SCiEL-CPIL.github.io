import time
import numpy as np
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin

    MAX_N = 256

    # 1. Try ILP approach
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Generate primes
        def get_primes(n):
            sieve = [True] * (n + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, isqrt(n) + 1):
                if sieve[i]:
                    for j in range(i * i, n + 1, i):
                        sieve[j] = False
            return [i for i, is_prime in enumerate(sieve) if is_prime]

        primes = get_primes(MAX_N)
        num_primes = len(primes)
        prime_to_idx = {p: i for i, p in enumerate(primes)}

        # Build constraint matrix
        A = np.zeros((num_primes, MAX_N), dtype=int)
        for n in range(1, MAX_N + 1):
            for p in primes:
                if n % p == 0:
                    A[prime_to_idx[p], n - 1] = 1

        # Objective: Minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=float)

        # Constraints: A @ x <= 1
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))

        # Integrality: all integer
        integrality = np.ones(MAX_N)

        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.status == 0:
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]
            
            # Verify
            if len(selected) > 0:
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if gcd(selected[i], selected[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return [int(n) for n in selected]
        # If ILP fails or returns invalid, fall through
    except Exception:
        pass

    # 2. Fallback: Greedy + Local Search
    S = [1]
    for n in range(256, 1, -1):
        if all(gcd(n, x) == 1 for x in S):
            S.append(n)

    best_set = set(S)
    best_sum = sum(S)

    # Local Search Optimization
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(best_set)
        for a in S_list:
            if a == 1:
                continue
            # Try to replace a with b
            for b in range(a + 1, 257):
                if b in best_set:
                    continue
                new_set = (best_set - {a}) | {b}
                # Check if new_set is valid
                ok = True
                # Optimization: only check gcd with elements that might conflict
                # But set is small, full check is fine
                for x in new_set:
                    for y in new_set:
                        if x < y and gcd(x, y) != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    new_sum = sum(new_set)
                    if new_sum > best_sum:
                        best_set = new_set
                        best_sum = new_sum
                        improved = True
                        break
        # Try adding a new number?
        # The greedy started with largest to smallest, so it's maximal.
        # But swaps can open up space for larger numbers?
        # If we swap a small a with large b, we might free up factors that allow adding c?
        # The current loop only does 1-for-1 swaps.
        # We can add a step to try adding a number if it fits.
        if time.time() < deadline:
            for b in range(256, 1, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set):
                    best_set.add(b)
                    best_sum += b
                    improved = True
                    break

    return [int(x) for x in sorted(best_set)]
