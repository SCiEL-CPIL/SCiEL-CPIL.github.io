import time
import random
import math
from math import isqrt

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    n = 12
    max_val = 10000
    limit = 2 * max_val  # max sum is 20000

    # Precompute smallest prime factor (SPF) sieve up to 2*max_val
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute prime factor frozensets for all numbers up to limit
    prime_factors_cache = [None] * (limit + 1)
    prime_factors_cache[0] = frozenset()
    prime_factors_cache[1] = frozenset()
    for x in range(2, limit + 1):
        factors = set()
        tmp = x
        while tmp > 1:
            p = spf[tmp]
            factors.add(p)
            while tmp % p == 0:
                tmp //= p
        prime_factors_cache[x] = frozenset(factors)

    def initial_score(nums):
        """Compute initial prime counts and distinct count."""
        prime_count = {}
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                for p in prime_factors_cache[s]:
                    prime_count[p] = prime_count.get(p, 0) + 1
        distinct = sum(1 for c in prime_count.values() if c > 0)
        return prime_count, distinct

    def remove_pairs_for_value(nums, idx, val, prime_count):
        """Remove all pairs involving idx where nums[idx] would be val."""
        for j in range(n):
            if j == idx:
                continue
            s = val + nums[j]
            for p in prime_factors_cache[s]:
                prime_count[p] -= 1

    def add_pairs_for_value(nums, idx, val, prime_count):
        """Add all pairs involving idx where nums[idx] would be val."""
        for j in range(n):
            if j == idx:
                continue
            s = val + nums[j]
            for p in prime_factors_cache[s]:
                prime_count[p] = prime_count.get(p, 0) + 1

    def distinct_count(prime_count):
        """Count primes with nonzero count."""
        return sum(1 for c in prime_count.values() if c > 0)

    # Initial best: multiples of 6
    best = [6 * (i + 1) for i in range(n)]  # 6, 12, 18, ..., 72
    best_prime_count, best_score = initial_score(best)

    # Try various structured initial solutions
    structured_starts = []
    # Multiples of 6 with various base multipliers
    for base_mult in range(1, 20):
        nums = [6 * base_mult * (i + 1) for i in range(n)]
        if all(1 <= x <= max_val for x in nums) and len(set(nums)) == n:
            structured_starts.append(nums)
    # Multiples of 12
    for base_mult in range(1, 12):
        nums = [12 * base_mult * (i + 1) for i in range(n)]
        if all(1 <= x <= max_val for x in nums) and len(set(nums)) == n:
            structured_starts.append(nums)
    # Multiples of 30
    for base_mult in range(1, 6):
        nums = [30 * base_mult * (i + 1) for i in range(n)]
        if all(1 <= x <= max_val for x in nums) and len(set(nums)) == n:
            structured_starts.append(nums)
    # Consecutive multiples of 6
    for offset in range(0, 5):
        nums = [6 * (offset + i) for i in range(n)]
        if all(1 <= x <= max_val for x in nums) and len(set(nums)) == n:
            structured_starts.append(nums)
    # Consecutive multiples of 12
    for offset in range(0, 4):
        nums = [12 * (offset + i) for i in range(n)]
        if all(1 <= x <= max_val for x in nums) and len(set(nums)) == n:
            structured_starts.append(nums)
    # Consecutive multiples of 30
    for offset in range(0, 3):
        nums = [30 * (offset + i) for i in range(n)]
        if all(1 <= x <= max_val for x in nums) and len(set(nums)) == n:
            structured_starts.append(nums)

    for nums in structured_starts:
        pc, sc = initial_score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]
            best_prime_count = pc

    # Simulated annealing / local search with fast incremental scoring
    if time.time() < deadline - 3:
        random.seed(42)
        current = best[:]
        current_pc, current_score = initial_score(current)
        best_found = best_score
        best_found_nums = best[:]

        # Temperature schedule
        T = 2.0
        cooling = 0.99995
        min_T = 0.01

        no_improve = 0
        max_no_improve = 5000

        # Precompute candidate value pools
        pool_6 = list(range(6, max_val + 1, 6))
        pool_12 = list(range(12, max_val + 1, 12))
        pool_30 = list(range(30, max_val + 1, 30))
        pool_3 = list(range(3, max_val + 1, 3))
        pool_2 = list(range(2, max_val + 1, 2))

        step = 0
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            step += 1

            i = random.randint(0, n - 1)
            old_val = current[i]

            # Choose a strategy for new value
            r = random.random()
            if r < 0.30:
                # Small perturbation: nearby multiple of 6
                base = (old_val // 6) * 6
                delta = random.choice([-6, -12, -18, -24, 6, 12, 18, 24, 30, 36])
                new_val = base + delta
            elif r < 0.50:
                # Random from pool_6
                new_val = random.choice(pool_6)
            elif r < 0.65:
                # Random from pool_12
                new_val = random.choice(pool_12)
            elif r < 0.75:
                # Random from pool_30
                new_val = random.choice(pool_30)
            elif r < 0.85:
                # Random from pool_3
                new_val = random.choice(pool_3)
            else:
                # Random from pool_2
                new_val = random.choice(pool_2)

            if new_val <= 0 or new_val > max_val:
                no_improve += 1
                continue
            if new_val in current:
                no_improve += 1
                continue

            # Try the move with incremental scoring
            # Remove old pairs, add new pairs
            remove_pairs_for_value(current, i, old_val, current_pc)
            current[i] = new_val
            add_pairs_for_value(current, i, new_val, current_pc)
            new_score = distinct_count(current_pc)

            # Accept or reject
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if new_score < best_found:
                    best_found = new_score
                    best_found_nums = current[:]
            else:
                # Metropolis accept with probability exp(-delta/T)
                delta = new_score - current_score
                if delta > 0 and T > min_T:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        no_improve = 0
                    else:
                        # Reject: restore old value
                        remove_pairs_for_value(current, i, new_val, current_pc)
                        current[i] = old_val
                        add_pairs_for_value(current, i, old_val, current_pc)
                        no_improve += 1
                else:
                    # Reject: restore old value
                    remove_pairs_for_value(current, i, new_val, current_pc)
                    current[i] = old_val
                    add_pairs_for_value(current, i, old_val, current_pc)
                    no_improve += 1

            T = max(min_T, T * cooling)

            # If we stuck, try a restart with a different seed
            if no_improve > 2000 and time.time() < deadline - 5:
                # Restart from best found
                current = best_found_nums[:]
                current_pc, current_score = initial_score(current)
                T = 1.5
                no_improve = 0

    if best_found < best_score:
        best = best_found_nums
        best_score = best_found

    return best
