import time
import random
from math import isqrt

_START = time.time()
_HARD_DEADLINE = _START + 85.0

def construct():
    deadline = time.time() + 82.0
    N = 12
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL

    # Precompute smallest prime factor
    spf = list(range(MAX_SUM + 1))
    for i in range(2, isqrt(MAX_SUM) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_prime_factors(n):
        if n <= 1:
            return frozenset()
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return frozenset(factors)

    # Precompute prime factors for all sums
    pf_cache = [frozenset()] * (MAX_SUM + 1)
    for s in range(2, MAX_SUM + 1):
        pf_cache[s] = get_prime_factors(s)

    def evaluate(a):
        """Return the set of all prime factors of pairwise sums."""
        primes = set()
        for i in range(N):
            for j in range(i + 1, N):
                primes.update(pf_cache[a[i] + a[j]])
        return primes

    best_primes = None
    best_set = None

    def update_best(a, primes):
        nonlocal best_primes, best_set
        if best_primes is None or len(primes) < len(best_primes):
            best_primes = primes
            best_set = a[:]

    # --- Strategy 1: Base multiplier with local search ---
    bases = [30, 60, 120, 210, 360, 840, 2, 4, 6, 8, 10, 12, 15, 18, 20, 24, 25]

    for base in bases:
        if time.time() > deadline - 5.0:
            break

        base_factors = pf_cache[base]
        max_k = MAX_VAL // base

        if max_k < N:
            continue

        search_limit = min(max_k, 150)

        cur_k = list(range(1, N + 1))

        def get_score_k(k_list):
            union = set(base_factors)
            for i in range(N - 1):
                for j in range(i + 1, N):
                    union.update(pf_cache[k_list[i] + k_list[j]])
            return union

        cur_primes = get_score_k(cur_k)

        improved = True
        iterations = 0
        while improved and iterations < 300 and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            indices = list(range(N))
            random.shuffle(indices)

            for i in indices:
                if time.time() > deadline - 5.0:
                    break

                old_k = cur_k[i]
                for new_k in range(1, search_limit + 1):
                    if new_k == old_k or new_k in cur_k:
                        continue

                    cur_k[i] = new_k
                    new_primes = get_score_k(cur_k)

                    if len(new_primes) < len(cur_primes):
                        cur_primes = new_primes
                        improved = True
                        break
                    cur_k[i] = old_k

                if improved:
                    x_vals = [base * k for k in cur_k]
                    if all(x <= MAX_VAL for x in x_vals):
                        update_best(x_vals, cur_primes)
                    break

        # Random restarts for this base
        for restart in range(30):
            if time.time() > deadline - 5.0:
                break
            cur_k = random.sample(range(1, search_limit + 1), N)
            cur_primes = get_score_k(cur_k)

            improved = True
            iterations = 0
            while improved and iterations < 100 and time.time() < deadline - 5.0:
                improved = False
                iterations += 1
                indices = list(range(N))
                random.shuffle(indices)

                for i in indices:
                    if time.time() > deadline - 5.0:
                        break
                    old_k = cur_k[i]
                    for new_k in range(1, search_limit + 1):
                        if new_k == old_k or new_k in cur_k:
                            continue
                        cur_k[i] = new_k
                        new_primes = get_score_k(cur_k)
                        if len(new_primes) < len(cur_primes):
                            cur_primes = new_primes
                            improved = True
                            break
                        cur_k[i] = old_k
                    if improved:
                        x_vals = [base * k for k in cur_k]
                        if all(x <= MAX_VAL for x in x_vals):
                            update_best(x_vals, cur_primes)
                        break

        if best_primes is not None and len(best_primes) <= 4:
            break

    # --- Strategy 2: Direct search with smooth numbers ---
    if best_primes is None or len(best_primes) > 5:
        pool = set()
        for a in range(0, 15):
            for b in range(0, 15):
                for c in range(0, 15):
                    val = (2 ** a) * (3 ** b) * (5 ** c)
                    if 1 <= val <= MAX_VAL:
                        pool.add(val)
        pool = sorted(pool)

        if len(pool) >= N:
            for restart in range(80):
                if time.time() > deadline - 5.0:
                    break
                cur_x = random.sample(pool[:200], N)
                cur_primes = evaluate(cur_x)

                improved = True
                iterations = 0
                while improved and iterations < 200 and time.time() < deadline - 5.0:
                    improved = False
                    iterations += 1
                    indices = list(range(N))
                    random.shuffle(indices)

                    for i in indices:
                        if time.time() > deadline - 5.0:
                            break
                        old_x = cur_x[i]
                        idx = pool.index(old_x)
                        for next_idx in range(max(0, idx - 10), min(len(pool), idx + 50)):
                            new_x = pool[next_idx]
                            if new_x == old_x or new_x in cur_x:
                                continue
                            cur_x[i] = new_x
                            new_primes = evaluate(cur_x)
                            if len(new_primes) < len(cur_primes):
                                cur_primes = new_primes
                                improved = True
                                break
                            cur_x[i] = old_x
                        if improved:
                            update_best(cur_x, cur_primes)
                            break

    # --- Strategy 3: General random restarts with local search ---
    for restart in range(300):
        if time.time() > deadline - 2.0:
            break

        a = random.sample(range(1, MAX_VAL + 1), N)
        current_primes = evaluate(a)
        update_best(a, current_primes)

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            indices = list(range(N))
            random.shuffle(indices)

            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                old_val = a[i]
                # Try replacing with a random value
                for _ in range(20):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val == old_val or new_val in a:
                        continue
                    a[i] = new_val
                    new_primes = evaluate(a)
                    if len(new_primes) < len(current_primes):
                        current_primes = new_primes
                        improved = True
                        update_best(a, current_primes)
                        break
                    a[i] = old_val
                if improved:
                    break

    if best_set is None:
        best_set = [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30]

    return best_set
