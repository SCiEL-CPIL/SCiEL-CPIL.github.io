import math
import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    def is_b_smooth(n, primes):
        """Check if n is B-smooth (all prime factors in primes)."""
        if n <= 1:
            return True
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    def generate_b_smooth_numbers(limit, primes):
        """Generate all B-smooth numbers <= limit."""
        smooths = {1}
        # Use DFS to generate all combinations
        def dfs(idx, current):
            if current > limit:
                return
            if idx == len(primes):
                smooths.add(current)
                return
            p = primes[idx]
            val = current
            while val <= limit:
                smooths.add(val)
                dfs(idx + 1, val)
                val *= p
        dfs(0, 1)
        return smooths

    # Try increasing B values
    # B=3: primes = [2, 3]
    # B=5: primes = [2, 3, 5]
    # etc.

    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    best_max_p = float('inf')
    best_a = 1
    best_b = target - 1

    for B in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397]:
        if time.time() > deadline:
            break

        # Get all primes <= B
        primes_up_to_b = [p for p in primes_list if p <= B]

        # Generate all B-smooth numbers <= target
        smooths = generate_b_smooth_numbers(target, primes_up_to_b)

        # For each B-smooth number a, check if b = target - a is also B-smooth
        found = False
        for a in sorted(smooths):
            if a < 1 or a > target - 1:
                continue
            b = target - a
            if b < 1:
                continue
            if b in smooths:
                # Both are B-smooth, so max(P(a), P(b)) <= B
                if B < best_max_p:
                    best_max_p = B
                    best_a = a
                    best_b = b
                    found = True
                    # Since we're checking in order of a, and we want any valid pair,
                    # we can break after finding the first for this B
                    # But let's continue to see if there's a better pair (smaller max)
                    # Actually, for this B, all valid pairs have max <= B
                    # We want to find the minimum B, so once we find one for B, 
                    # we should try smaller B first. But we're iterating B in increasing order,
                    # so the first B where we find a valid pair is optimal.
                    break

        if found:
            # We found the optimal B, no need to check larger B
            break

    # Fallback: if no B-smooth pair found (shouldn't happen for small B),
    # do a full LPF-based search
    if best_max_p == float('inf'):
        # Precompute largest prime factor
        lpf = [0] * (target + 1)
        lpf[1] = 1
        for i in range(2, target + 1):
            if lpf[i] == 0:
                for j in range(i, target + 1, i):
                    lpf[j] = i

        half = target // 2
        for a in range(1, half + 1):
            b = target - a
            pa = lpf[a]
            pb = lpf[b]
            cur_max = pa if pa > pb else pb
            if cur_max < best_max_p:
                best_max_p = cur_max
                best_a = a
                best_b = b
                if best_max_p == 3:
                    break
            if time.time() > deadline:
                break

    return [best_a, best_b]
