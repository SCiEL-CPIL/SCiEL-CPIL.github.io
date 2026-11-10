import time
import numpy as np
from math import gcd, isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

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

    # Precompute compatibility matrix
    compat = [[False] * (MAX_N + 1) for _ in range(MAX_N + 1)]
    for a in range(1, MAX_N + 1):
        for b in range(1, MAX_N + 1):
            if a == b or gcd(a, b) == 1:
                compat[a][b] = True

    # Precompute bitmask for each number: which other numbers it's compatible with
    compat_mask = [0] * (MAX_N + 1)
    for a in range(1, MAX_N + 1):
        mask = 0
        for b in range(1, MAX_N + 1):
            if a == b or gcd(a, b) == 1:
                mask |= (1 << (b - 1))
        compat_mask[a] = mask

    def is_compatible_with_all(n, S):
        if not S:
            return True
        s_mask = 0
        for x in S:
            s_mask |= (1 << (x - 1))
        return (compat_mask[n] & s_mask) == s_mask

    def is_valid_set(S):
        lst = list(S)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if not compat[lst[i]][lst[j]]:
                    return False
        return True

    # Strategy 1: Prime-power greedy
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

    # Strategy 2: Descending greedy
    def greedy_descending():
        S = {1}
        for n in range(MAX_N, 1, -1):
            if is_compatible_with_all(n, S):
                S.add(n)
        return S

    # Strategy 3: Prime upgrade greedy
    def prime_upgrade_greedy():
        S = {1}
        for p in primes:
            if p <= MAX_N:
                S.add(p)
        for p in list(S):
            if p == 1:
                continue
            for b in range(MAX_N, p, -1):
                if b in S:
                    continue
                ok = all(compat[b][x] for x in S if x != p)
                if ok:
                    S.discard(p)
                    S.add(b)
                    break
        return S

    # Strategy 4: Bundle greedy
    def bundle_greedy():
        S = {1}
        for p in sorted(primes, reverse=True):
            best = None
            for n in range(MAX_N, 1, -1):
                if n in S:
                    continue
                factors = prime_factors[n]
                if p in factors and all(q <= p for q in factors):
                    if is_compatible_with_all(n, S):
                        best = n
                        break
            if best is not None:
                S.add(best)
        return S

    # Strategy 5: Distinct prime greedy
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

    # Strategy 6: Randomized greedy
    import random
    def randomized_greedy(seed_val):
        rng = random.Random(seed_val)
        S = {1}
        candidates = list(range(MAX_N, 1, -1))
        rng.shuffle(candidates)
        for n in candidates:
            if is_compatible_with_all(n, S):
                S.add(n)
        return S

    # Enhanced local search with simulated annealing
    def local_search(S, S_sum, time_limit, use_sa=True):
        improved = True
        no_improve_count = 0
        current_temp = 100.0
        min_temp = 0.1
        
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
                    ok = all(compat[b][x] for x in S if x != a)
                    if ok:
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
            
            # Move type 3: Try 2-for-1 swaps
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
                            ok = all(compat[c][x] for x in S if x != a and x != b)
                            if ok:
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
            
            # Move type 4: Try 1-for-2 swaps
            for a in list(S):
                if a == 1:
                    continue
                found = False
                for b in range(MAX_N, 1, -1):
                    if b in S or b == a:
                        continue
                    ok_b = all(compat[b][x] for x in S if x != a)
                    if not ok_b:
                        continue
                    for c in range(MAX_N, b, -1):
                        if c in S or c == a or c == b:
                            continue
                        if b + c <= a:
                            continue
                        ok_c = all(compat[c][x] for x in S if x != a)
                        if not ok_c or not compat[b][c]:
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

            # Simulated annealing: accept worse moves with probability exp(-delta/temp)
            if use_sa and not improved and current_temp > min_temp:
                import math
                # Try a random 1-for-1 swap that might be worse
                a = random.choice([x for x in S if x != 1]) if len(S) > 1 else None
                if a:
                    for _ in range(50):
                        b = random.randint(2, MAX_N)
                        if b in S:
                            continue
                        ok = all(compat[b][x] for x in S if x != a)
                        if not ok:
                            continue
                        delta = (a - b)  # positive means we're losing
                        if delta < 0 or random.random() < math.exp(-delta / current_temp):
                            S.discard(a)
                            S.add(b)
                            S_sum = S_sum - a + b
                            improved = True
                            no_improve_count = 0
                            break
                
                current_temp *= 0.95

            if not improved:
                no_improve_count += 1
                if no_improve_count >= 5:
                    break

        return S, S_sum

    # Try ILP
    best_set = None
    best_sum = 0

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        A = np.zeros((num_primes, MAX_N), dtype=np.int8)
        for n in range(1, MAX_N + 1):
            for p in prime_factors[n]:
                A[prime_to_idx[p], n - 1] = 1

        c = -np.arange(1, MAX_N + 1, dtype=float)
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))
        integrality = np.ones(MAX_N)
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.status == 0:
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]
            if len(selected) > 0:
                if is_valid_set(set(selected)):
                    best_set = set(selected)
                    best_sum = sum(selected)
    except Exception:
        pass

    # Generate multiple candidate initializations
    candidates = [
        prime_power_greedy(),
        greedy_descending(),
        prime_upgrade_greedy(),
        bundle_greedy(),
        distinct_prime_greedy(),
        randomized_greedy(42),
        randomized_greedy(123),
        randomized_greedy(777),
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
    if best_set is not None and time.time() < deadline - 10:
        best_set, best_sum = local_search(best_set, best_sum, deadline - 5, use_sa=True)

    # Try local search on second-best candidate if time permits
    sorted_candidates = sorted(candidates, key=lambda s: sum(s), reverse=True)
    if len(sorted_candidates) >= 2 and time.time() < deadline - 15:
        second_best = sorted_candidates[1]
        if second_best is not best_set:
            sb_set, sb_sum = local_search(set(second_best), sum(second_best), deadline - 5, use_sa=False)
            if sb_sum > best_sum:
                best_set = sb_set
                best_sum = sb_sum

    # Final polish
    if time.time() < deadline - 2:
        for b in range(MAX_N, 1, -1):
            if b in best_set:
                continue
            if is_compatible_with_all(b, best_set):
                best_set.add(b)
                best_sum += b

        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for a in list(best_set):
                if a == 1:
                    continue
                for b in range(MAX_N, a, -1):
                    if b in best_set:
                        continue
                    ok = all(compat[b][x] for x in best_set if x != a)
                    if ok:
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
