import time
import numpy as np
from math import gcd, isqrt
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill

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

    # Precompute prime factors for each number
    prime_factors = {}
    for n in range(1, MAX_N + 1):
        factors = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        prime_factors[n] = factors

    # Precompute conflict bitmasks: conflict_bits[n] has bit m-1 set if gcd(n,m) != 1
    conflict_bits = [0] * (MAX_N + 1)
    for n in range(1, MAX_N + 1):
        bits = 0
        for m in range(1, MAX_N + 1):
            if n != m and gcd(n, m) != 1:
                bits |= (1 << (m - 1))
        conflict_bits[n] = bits

    def is_compatible_with_all(n, S):
        if not S:
            return True
        s_mask = 0
        for x in S:
            s_mask |= (1 << (x - 1))
        return (conflict_bits[n] & s_mask) == 0

    # Greedy strategies for initial solutions
    def greedy_descending():
        S = {1}
        for n in range(MAX_N, 1, -1):
            if is_compatible_with_all(n, S):
                S.add(n)
        return S

    def prime_power_greedy():
        S = {1}
        for p in primes:
            best = p
            power = p
            while power * p <= MAX_N:
                power *= p
                best = power
            if is_compatible_with_all(best, S):
                S.add(best)
        return S

    def distinct_prime_greedy():
        S = {1}
        used_primes = set()
        for n in range(MAX_N, 1, -1):
            factors = prime_factors[n]
            if factors.isdisjoint(used_primes):
                if is_compatible_with_all(n, S):
                    S.add(n)
                    used_primes.update(factors)
        return S

    def randomized_greedy(seed_val):
        rng = random.Random(seed_val)
        S = {1}
        candidates = list(range(MAX_N, 1, -1))
        rng.shuffle(candidates)
        for n in candidates:
            if is_compatible_with_all(n, S):
                S.add(n)
        return S

    # Local search with bitmask-based compatibility
    def local_search(S, S_sum, time_limit):
        improved = True
        no_improve_count = 0
        min_improve_count = 8

        while improved and time.time() < time_limit:
            improved = False
            S_list = sorted(S, reverse=True)

            # Move type 1: Try adding a number if it fits
            for b in range(MAX_N, 1, -1):
                if b in S:
                    continue
                if is_compatible_with_all(b, S):
                    S.add(b)
                    S_sum += b
                    improved = True
                    no_improve_count = 0
                    break

            # Move type 2: Try 1-for-1 swaps
            for a in list(S_list):
                if a == 1:
                    continue
                for b in range(MAX_N, a, -1):
                    if b in S:
                        continue
                    if is_compatible_with_all(b, S - {a}):
                        new_sum = S_sum - a + b
                        if new_sum > S_sum:
                            S.discard(a)
                            S.add(b)
                            S_sum = new_sum
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break

            # Move type 3: Try 2-for-1 swaps (remove two small, add one big)
            if len(S) >= 2:
                S_sorted = sorted(S)
                for i in range(len(S_sorted)):
                    a = S_sorted[i]
                    if a == 1:
                        continue
                    for j in range(i + 1, len(S_sorted)):
                        b = S_sorted[j]
                        if b == 1:
                            continue
                        if a + b >= MAX_N:
                            continue
                        for c in range(MAX_N, max(a, b), -1):
                            if c in S:
                                continue
                            if is_compatible_with_all(c, S - {a, b}):
                                new_sum = S_sum - a - b + c
                                if new_sum > S_sum:
                                    S.discard(a)
                                    S.discard(b)
                                    S.add(c)
                                    S_sum = new_sum
                                    improved = True
                                    no_improve_count = 0
                                    break
                        if improved:
                            break
                    if improved:
                        break

            # Move type 4: Try 1-for-2 swaps (remove one, add two)
            for a in list(S):
                if a == 1:
                    continue
                found = False
                for b in range(MAX_N, 1, -1):
                    if b in S or b == a:
                        continue
                    if not is_compatible_with_all(b, S - {a}):
                        continue
                    for c in range(MAX_N, b, -1):
                        if c in S or c == a or c == b:
                            continue
                        if b + c <= a:
                            continue
                        if not is_compatible_with_all(c, S - {a}) or gcd(b, c) != 1:
                            continue
                        new_sum = S_sum - a + b + c
                        if new_sum > S_sum:
                            S.discard(a)
                            S.add(b)
                            S.add(c)
                            S_sum = new_sum
                            improved = True
                            found = True
                            no_improve_count = 0
                            break
                    if found:
                        break
                if improved:
                    break

            if not improved:
                no_improve_count += 1
                if no_improve_count >= min_improve_count:
                    break

        return S, S_sum

    # Try ILP with scipy
    best_set = None
    best_sum = 0

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix

        # Build constraint matrix: for each prime, sum of selected numbers divisible by it <= 1
        # A has shape (num_primes, MAX_N)
        rows = []
        cols = []
        vals = []
        for n in range(1, MAX_N + 1):
            for p in prime_factors[n]:
                idx = prime_to_idx[p]
                rows.append(idx)
                cols.append(n - 1)
                vals.append(1.0)

        A = csr_matrix((vals, (rows, cols)), shape=(num_primes, MAX_N))

        c = -np.arange(1, MAX_N + 1, dtype=float)
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))
        integrality = np.ones(MAX_N)
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        ilp_time = min(40, max(5, deadline - time.time() - 15))
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={
                'time_limit': ilp_time,
                'mip_rel_gap': 0.0,
                'mip_abs_gap': 0.0,
                'presolve': 'on',
                'parallel': 'on',
            }
        )

        if res.status in (0, 2, 3):  # optimal, time limit, or gap
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]

            # Verify validity
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
                    best_set = set(selected)
                    best_sum = sum(selected)
    except Exception:
        pass

    # Generate multiple candidate initializations
    candidates = [
        greedy_descending(),
        prime_power_greedy(),
        distinct_prime_greedy(),
        randomized_greedy(42),
        randomized_greedy(123),
        randomized_greedy(777),
        randomized_greedy(1234),
        randomized_greedy(5678),
    ]

    # Add ILP result if valid
    if best_set is not None:
        candidates.append(best_set)

    # Find best candidate
    for cand in candidates:
        cand_sum = sum(cand)
        if best_set is None or cand_sum > best_sum:
            best_set = cand
            best_sum = cand_sum

    # Run local search on best candidate
    if best_set is not None and time.time() < deadline - 5:
        best_set, best_sum = local_search(best_set, best_sum, deadline - 3)

    # Try local search on second-best candidate if time permits
    sorted_candidates = sorted(candidates, key=lambda s: sum(s), reverse=True)
    if len(sorted_candidates) >= 2 and time.time() < deadline - 8:
        second_best = sorted_candidates[1]
        if second_best is not best_set:
            sb_set, sb_sum = local_search(set(second_best), sum(second_best), deadline - 3)
            if sb_sum > best_sum:
                best_set = sb_set
                best_sum = sb_sum

    # Final polish: try to add any remaining compatible numbers
    if time.time() < deadline - 1:
        for b in range(MAX_N, 1, -1):
            if b in best_set:
                continue
            if is_compatible_with_all(b, best_set):
                best_set.add(b)
                best_sum += b

        # One more round of 1-for-1 swaps
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for a in list(best_set):
                if a == 1:
                    continue
                for b in range(MAX_N, a, -1):
                    if b in best_set:
                        continue
                    if is_compatible_with_all(b, best_set - {a}):
                        new_sum = best_sum - a + b
                        if new_sum > best_sum:
                            best_set.discard(a)
                            best_set.add(b)
                            best_sum = new_sum
                            improved = True
                            break
                if improved:
                    break

    result = sorted(best_set)
    if not result:
        result = [1]

    return [int(x) for x in result]
