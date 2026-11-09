import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    N = 123457

    # Precomputed primes up to ~300 (enough for K-smooth search)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]

    def generate_smooth_bfs(primes_list, limit):
        """Generate all primes_list-smooth numbers <= limit via BFS-like expansion."""
        smooths = {1}
        for p in primes_list:
            # For each existing smooth number, multiply by powers of p
            new_smooths = set()
            for s in smooths:
                v = s * p
                while v <= limit:
                    new_smooths.add(v)
                    v *= p
            smooths = smooths | new_smooths
        return smooths

    # Fallback: exhaustive LPF scan (guaranteed correct)
    def exhaustive_fallback(deadline):
        # Sieve for largest prime factor using array for fast lookup
        lpf = [1] * (N + 1)
        for i in range(2, N + 1):
            if lpf[i] == 1:  # i is prime
                for j in range(i, N + 1, i):
                    lpf[j] = i

        best_max_pf = float('inf')
        best_a = 1
        best_b = N - 1

        # Scan all possible decompositions using array lookup (faster than set hashing)
        for a in range(1, N // 2 + 1):
            if time.time() > deadline:
                break
            b = N - a
            m = lpf[a] if lpf[a] > lpf[b] else lpf[b]
            if m < best_max_pf:
                best_max_pf = m
                best_a = a
                best_b = b
                if m <= 3:
                    break
        return [best_a, best_b]

    # Primary strategy: try increasing K (prime bound)
    # Since we increase K, the first K that yields a valid pair gives the minimum max(P(a),P(b))
    for i in range(len(primes)):
        if time.time() - start_time > DEADLINE - 15:
            # Not enough time left for smooth approach; fall back to exhaustive
            return exhaustive_fallback(start_time + DEADLINE)

        current_primes = primes[:i + 1]
        max_p = current_primes[-1]

        smooths = generate_smooth_bfs(current_primes, N)

        # Check if any a in smooths has b = N - a also in smooths
        # Using set lookup for O(1) membership test
        found = False
        best_pair = None
        for a in sorted(smooths):
            b = N - a
            if 1 <= b <= N and b in smooths:
                found = True
                best_pair = (a, b)
                break

        if found:
            a, b = best_pair
            return [a, b]

    # If we exhaust all primes without finding a pair, fall back to exhaustive scan
    if time.time() - start_time < DEADLINE - 5:
        return exhaustive_fallback(start_time + DEADLINE)

    # Last resort: return a valid (if not optimal) decomposition
    return [1, N - 1]
