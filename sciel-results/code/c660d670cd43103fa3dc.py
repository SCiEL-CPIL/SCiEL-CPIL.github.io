import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin from 150s limit

    # Strategy: Smooth number backward search
    # Generate all k <= 10^7 composed solely of primes < 50 (5-smooth to 47-smooth)
    # These are the only candidates capable of producing high-density collisions
    # for M <= 10^18 due to the rapid growth of sigma(k) for larger prime factors.
    # Then group these sparse candidates by k*sigma(k) to identify M with max preimages.

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    max_k = 10_000_000
    max_M = 10**18

    # Generate all smooth numbers up to max_k using recursive generation
    def generate_smooth_numbers(idx, current_val, factors):
        """Recursively generate all numbers composed of primes[idx:] up to max_k"""
        if current_val > max_k:
            return
        if idx == len(primes):
            if current_val > 1:
                yield current_val, factors
            return
        p = primes[idx]
        # Try all powers of p
        p_pow = 1
        while p_pow * current_val <= max_k:
            yield from generate_smooth_numbers(idx + 1, current_val * p_pow, factors + [(p, 0)] if p_pow == 1 else factors[:-1] + [(p, 0)])
            # Actually, let's track exponents properly
            p_pow *= p

    # Better approach: generate with proper exponent tracking
    def generate_smooth_v2(idx, current_val, exps):
        """Generate smooth numbers, tracking exponents for each prime"""
        if current_val > max_k:
            return
        if idx == len(primes):
            if current_val > 1:
                yield current_val, exps[:]
            return
        p = primes[idx]
        p_pow = 1
        for a in range(0, 50):  # max exponent safety
            new_val = current_val * p_pow
            if new_val > max_k:
                break
            new_exps = exps[:]
            new_exps[idx] = a
            yield from generate_smooth_v2(idx + 1, new_val, new_exps)
            p_pow *= p

    # Compute sigma from exponents
    def sigma_from_exps(exps):
        s = 1
        for i, a in enumerate(exps):
            if a > 0:
                p = primes[i]
                s *= (p**(a+1) - 1) // (p - 1)
        return s

    # Group by M = k * sigma(k)
    m_to_k_list = defaultdict(list)

    # Use iterative approach to avoid deep recursion
    # Start with all combinations of exponents for the first few primes
    # This is more efficient than recursive generation for 15 primes

    # Actually, let's use a priority queue / heap approach to generate smooth numbers in order
    # Or just use the recursive approach but with memoization

    # Let's try a different approach: generate all smooth numbers by iterating
    # over exponent combinations for the first N primes where the product is bounded

    # For 15 primes, the number of combinations is large but we can prune early
    # Let's use recursion with pruning

    def gen_smooth(idx, current_val, exps, results):
        if time.time() > deadline - 5:
            return
        if current_val > max_k:
            return
        if idx == len(primes):
            if current_val > 1:
                results.append((current_val, exps[:]))
            return
        p = primes[idx]
        p_pow = 1
        for a in range(0, 50):
            new_val = current_val * p_pow
            if new_val > max_k:
                break
            exps[idx] = a
            gen_smooth(idx + 1, new_val, exps, results)
            p_pow *= p
        exps[idx] = 0

    results = []
    exps = [0] * len(primes)
    gen_smooth(0, 1, exps, results)

    # Group by M
    for k, k_exps in results:
        sigma_k = sigma_from_exps(k_exps)
        M = k * sigma_k
        if M <= max_M:
            m_to_k_list[M].append(k)

    # Find best M
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3

    for M, k_list in m_to_k_list.items():
        if len(k_list) > best_count:
            best_count = len(k_list)
            best_M = M
            best_k_list = k_list

    # If we have time, we could also try a hybrid approach
    # But for now, let's return the best found

    # Ensure types
    return {'M': int(best_M), 'k': [int(k) for k in best_k_list]}

if __name__ == "__main__":
    construct()
