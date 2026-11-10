import time
import numpy as np
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s margin before 90s kill

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
    num_primes = len(primes)

    # Precompute conflict matrix: conflict[i][j] = True if gcd(i+1, j+1) > 1
    # Using 0-indexed: conflict[i][j] means numbers (i+1) and (j+1) conflict
    conflict = np.zeros((MAX_N, MAX_N), dtype=np.bool_)
    for i in range(MAX_N):
        for j in range(i + 1, MAX_N):
            if gcd(i + 1, j + 1) > 1:
                conflict[i][j] = True
                conflict[j][i] = True

    # 1. Try ILP approach with time limit
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Build constraint matrix: A[p_idx, n-1] = 1 if p divides n
        A = np.zeros((num_primes, MAX_N), dtype=np.int8)
        for p_idx, p in enumerate(primes):
            for multiple in range(p, MAX_N + 1, p):
                A[p_idx, multiple - 1] = 1

        # Objective: Minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=np.float64)

        # Constraints: A @ x <= 1
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))

        # Integrality: all integer
        integrality = np.ones(MAX_N)

        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        # Time limit for ILP: use remaining time minus buffer
        time_left = deadline - time.time() - 5
        if time_left > 5:
            options = {'mip_rel_gap': 0, 'mip_abs_gap': 0, 'time_limit': max(time_left, 10)}
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)

            if res.status == 0 and res.x is not None:
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

    # 2. Fallback: Multi-start Greedy + Local Search
    best_set = set()
    best_sum = 0

    def run_greedy(order):
        S = [1]
        for n in order:
            if all(gcd(n, x) == 1 for x in S):
                S.append(n)
        return S

    # Try descending order
    S1 = run_greedy(range(256, 1, -1))
    if sum(S1) > best_sum:
        best_set = set(S1)
        best_sum = sum(S1)

    # Try a few random permutations if time allows
    import random
    random.seed(42)
    num_restarts = 20
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
        perm = list(range(2, 257))
        random.shuffle(perm)
        S_rand = run_greedy(perm)
        if sum(S_rand) > best_sum:
            best_set = set(S_rand)
            best_sum = sum(S_rand)

    # Local Search Optimization with conflict matrix
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        S_list = sorted(best_set)
        
        # Try single swaps: remove a, add b
        for a in S_list:
            if a == 1:
                continue
            a_idx = a - 1
            for b in range(256, a, -1):
                if b in best_set:
                    continue
                b_idx = b - 1
                # Check if b conflicts with any element in best_set except a
                ok = True
                for x in S_list:
                    if x == a or x == b:
                        continue
                    x_idx = x - 1
                    if conflict[b_idx][x_idx]:
                        ok = False
                        break
                if ok:
                    new_sum = best_sum - a + b
                    if new_sum > best_sum:
                        best_set = (best_set - {a}) | {b}
                        best_sum = new_sum
                        S_list = sorted(best_set)
                        improved = True
                        break
        
        # Try single additions
        if time.time() < deadline - 2:
            for b in range(256, 1, -1):
                if b in best_set:
                    continue
                b_idx = b - 1
                ok = True
                for x in best_set:
                    if x == b:
                        continue
                    x_idx = x - 1
                    if conflict[b_idx][x_idx]:
                        ok = False
                        break
                if ok:
                    best_set.add(b)
                    best_sum += b
                    improved = True
                    break
        
        # Try 2-swaps: remove a1, a2 and add b1, b2
        if time.time() < deadline - 5:
            S_list = sorted(best_set)
            found_two_swap = False
            for i, a1 in enumerate(S_list):
                if a1 == 1:
                    continue
                for a2 in S_list[i+1:]:
                    if a2 == 1:
                        continue
                    a1_idx, a2_idx = a1 - 1, a2 - 1
                    for b1 in range(256, max(a1, a2), -1):
                        if b1 in best_set or b1 == a1 or b1 == a2:
                            continue
                        b1_idx = b1 - 1
                        # Check b1 against rest
                        ok_b1 = True
                        for x in S_list:
                            if x == a1 or x == a2 or x == b1:
                                continue
                            if conflict[b1_idx][x - 1]:
                                ok_b1 = False
                                break
                        if not ok_b1:
                            continue
                        for b2 in range(256, b1, -1):
                            if b2 in best_set or b2 == a1 or b2 == a2 or b2 == b1:
                                continue
                            b2_idx = b2 - 1
                            # Check b2 against rest and b1
                            if conflict[b1_idx][b2_idx]:
                                continue
                            ok_b2 = True
                            for x in S_list:
                                if x == a1 or x == a2 or x == b1 or x == b2:
                                    continue
                                if conflict[b2_idx][x - 1]:
                                    ok_b2 = False
                                    break
                            if ok_b2:
                                new_sum = best_sum - a1 - a2 + b1 + b2
                                if new_sum > best_sum:
                                    best_set = (best_set - {a1, a2}) | {b1, b2}
                                    best_sum = new_sum
                                    improved = True
                                    found_two_swap = True
                                    break
                        if found_two_swap:
                            break
                    if found_two_swap:
                        break
                if found_two_swap:
                    break

    return [int(x) for x in sorted(best_set)]
