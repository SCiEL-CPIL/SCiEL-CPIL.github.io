"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    n = 12
    max_val = 10000

    # Precompute smallest prime factor (SPF) sieve up to 2*max_val
    limit = 2 * max_val
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors_set(x):
        """Return set of prime factors of x using SPF."""
        factors = set()
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        return factors

    def score(nums):
        """Count distinct prime factors across all pairwise sums."""
        primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                primes |= prime_factors_set(s)
        return len(primes)

    # Initial best: small numbers
    best = list(range(1, 13))
    best_score = score(best)

    # Strategy 1: Arithmetic progressions
    for a in range(1, 200):
        for d in range(1, 200):
            if time.time() > deadline:
                break
            nums = [a + i * d for i in range(n)]
            if any(x > max_val or x <= 0 for x in nums):
                continue
            sc = score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]
        if time.time() > deadline:
            break

    # Strategy 2: Multiples of a base
    for base in range(2, 500):
        if time.time() > deadline:
            break
        # Consecutive multiples
        for sm in range(1, 10):
            nums = [base * (sm + i) for i in range(n)]
            if any(x > max_val for x in nums):
                break
            sc = score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]

    # Strategy 3: Smooth numbers (2^a * 3^b * 5^c * 7^d)
    smooth = set()
    for a in range(0, 14):
        v1 = 2 ** a
        if v1 > max_val:
            break
        for b in range(0, 10):
            v2 = v1 * (3 ** b)
            if v2 > max_val:
                break
            for c in range(0, 7):
                v3 = v2 * (5 ** c)
                if v3 > max_val:
                    break
                for d in range(0, 5):
                    v4 = v3 * (7 ** d)
                    if v4 > max_val:
                        break
                    smooth.add(v4)
    smooth = sorted(smooth)
    if len(smooth) >= n:
        # Try first 12
        nums = smooth[:n]
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]
        # Try last 12
        nums = smooth[-n:]
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]
        # Try some middle selections
        for start_idx in range(0, max(1, len(smooth) - n), 50):
            if time.time() > deadline:
                break
            nums = smooth[start_idx:start_idx + n]
            sc = score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]

    # Strategy 4: All even numbers (consecutive even)
    for start_even in range(2, 100, 2):
        if time.time() > deadline:
            break
        nums = [start_even + 2 * i for i in range(n)]
        if any(x > max_val for x in nums):
            continue
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]

    # Strategy 5: All multiples of 4
    for start_m4 in range(4, 200, 4):
        if time.time() > deadline:
            break
        nums = [start_m4 + 4 * i for i in range(n)]
        if any(x > max_val for x in nums):
            continue
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]

    # Strategy 6: All multiples of 6
    for start_m6 in range(6, 200, 6):
        if time.time() > deadline:
            break
        nums = [start_m6 + 6 * i for i in range(n)]
        if any(x > max_val for x in nums):
            continue
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]

    # Strategy 7: Local search with targeted replacements
    if time.time() < deadline - 5:
        random.seed(42)
        current = best[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 300

        while time.time() < deadline - 3 and no_improve < max_no_improve:
            i = random.randint(0, n - 1)
            old_val = current[i]

            # Generate candidate replacements biased toward numbers with few prime factors
            strategy = random.randint(0, 4)
            if strategy == 0:
                # Random even
                new_val = random.randint(1, max_val // 2) * 2
            elif strategy == 1:
                # Multiple of 3
                new_val = random.randint(1, max_val // 3) * 3
            elif strategy == 2:
                # Multiple of 4
                new_val = random.randint(1, max_val // 4) * 4
            elif strategy == 3:
                # Multiple of 6
                new_val = random.randint(1, max_val // 6) * 6
            else:
                new_val = random.randint(1, max_val)

            if new_val > max_val or new_val <= 0:
                no_improve += 1
                continue
            if new_val in current:
                no_improve += 1
                continue

            current[i] = new_val
            sc = score(current)
            if sc < current_score:
                current_score = sc
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best = current[:]
            else:
                current[i] = old_val
                no_improve += 1

    return best
# EVOLVE-BLOCK-END
