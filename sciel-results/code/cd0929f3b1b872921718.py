import time
import numpy as np
from math import gcd, isqrt
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin before 90s kill

    MAX_N = 256

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

    # Pre-compute coprime masks using bitsets for speed
    # coprime_mask[n] is a set of numbers m in 1..256 such that gcd(n, m) == 1
    coprime_with = [set() for _ in range(MAX_N + 1)]
    for n in range(1, MAX_N + 1):
        for m in range(1, MAX_N + 1):
            if gcd(n, m) == 1:
                coprime_with[n].add(m)

    # 1. Try ILP approach with time limit
    best_result = None
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Build constraint matrix: for each prime p, at most one selected number can be divisible by p
        A = np.zeros((num_primes, MAX_N), dtype=float)
        for n in range(1, MAX_N + 1):
            for p in primes:
                if n % p == 0:
                    A[prime_to_idx[p], n - 1] = 1.0

        # Objective: Minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=float)

        # Constraints: A @ x <= 1
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))

        # Integrality: all integer
        integrality = np.ones(MAX_N)

        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": 30})

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
                    best_result = [int(n) for n in selected]
    except Exception:
        pass

    # 2. If ILP gave a result, use it as starting point; otherwise use greedy
    if best_result is not None:
        best_set = set(best_result)
    else:
        # Greedy: pick largest coprime numbers
        S = []
        for n in range(256, 0, -1):
            if all(gcd(n, x) == 1 for x in S):
                S.append(n)
        best_set = set(S)

    best_sum = sum(best_set)

    # Local Search Optimization with time budget
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(best_set)

        # 1-for-1 swaps: replace a with b where b > a and b is coprime with all others
        for a in S_list:
            if a == 1:
                continue
            for b in range(256, a, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set if x != a):
                    new_sum = best_sum - a + b
                    if new_sum > best_sum:
                        best_set = (best_set - {a}) | {b}
                        best_sum = new_sum
                        improved = True
                        break
            if improved:
                break

        # Add numbers if they fit
        if time.time() < deadline:
            for b in range(256, 1, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set):
                    best_set.add(b)
                    best_sum += b
                    improved = True
                    break

        # 2-for-1 swaps: remove two, add one bigger one
        if time.time() < deadline:
            found = False
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    a1 = S_list[i]
                    a2 = S_list[j]
                    if a1 == 1 or a2 == 1:
                        continue
                    for b in range(256, a1 + a2, -1):
                        if b in best_set or b == a1 or b == a2:
                            continue
                        if all(gcd(b, x) == 1 for x in best_set if x not in (a1, a2)):
                            new_sum = best_sum - a1 - a2 + b
                            if new_sum > best_sum:
                                best_set = (best_set - {a1, a2}) | {b}
                                best_sum = new_sum
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if found:
                    break

        # 1-for-2 swaps: remove one, add two smaller ones (rarely helps but try)
        if time.time() < deadline:
            found = False
            for a in S_list:
                if a == 1:
                    continue
                for b1 in range(256, 1, -1):
                    if b1 in best_set or b1 == a:
                        continue
                    if not all(gcd(b1, x) == 1 for x in best_set if x != a):
                        continue
                    for b2 in range(b1 - 1, 1, -1):
                        if b2 in best_set or b2 == a or b2 == b1:
                            continue
                        if not all(gcd(b2, x) == 1 for x in best_set if x != a):
                            continue
                        if gcd(b1, b2) != 1:
                            continue
                        new_sum = best_sum - a + b1 + b2
                        if new_sum > best_sum:
                            best_set = (best_set - {a}) | {b1, b2}
                            best_sum = new_sum
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    return [int(x) for x in sorted(best_set)]
