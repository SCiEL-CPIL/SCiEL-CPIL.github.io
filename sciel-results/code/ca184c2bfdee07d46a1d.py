import time
import math

def construct():
    deadline = time.time() + 80
    target = 123457

    # Precompute largest prime factor (LPF) for all numbers up to target-1
    # This is the most reliable way to find the optimal solution.
    # We use a modified Sieve of Eratosthenes to find the largest prime factor.
    
    max_n = target - 1
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    
    # Standard sieve for largest prime factor
    # For each prime p, update lpf for all multiples of p
    # Since we iterate p in increasing order, the last prime that divides n is the largest.
    # Wait, that's not quite right. If we iterate p from 2 to max_n, and for each p,
    # we update lpf[j] = p for all j that are multiples of p, then lpf[j] will end up being
    # the largest prime factor of j.
    
    # Optimization: Only iterate p up to sqrt(max_n) for the sieve part?
    # No, because a number can be prime itself, in which case its largest prime factor is itself.
    # So we need to handle primes > sqrt(max_n).
    
    # Efficient approach:
    # 1. Initialize lpf with 0.
    # 2. For i from 2 to max_n:
    #    If lpf[i] == 0, then i is prime.
    #    For j in range(i, max_n+1, i):
    #        lpf[j] = i
    # This works because if i is prime, it updates all its multiples.
    # Since i increases, the last update to lpf[j] comes from the largest prime factor of j.
    
    # This is O(N log log N) roughly, which is very fast for N=123456.
    
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
                # Actually, we want the LARGEST prime factor.
                # So we should always update lpf[j] = i if i is a prime factor.
                # Since i is increasing, the last i that divides j will be the largest.
                # So we can just do:
                lpf[j] = i

    # Now lpf[n] contains the largest prime factor of n.
    # We need to find a, b such that a + b = target and max(lpf[a], lpf[b]) is minimized.
    
    best_max = float('inf')
    best_a = 1
    
    # We can iterate a from 1 to target//2.
    # b = target - a.
    # Check max(lpf[a], lpf[b]).
    
    # Optimization: If we find a solution where best_max is small, we can stop early?
    # The theoretical minimum is 2 (if both are powers of 2, but their sum is odd, so one must be odd, so min is 3? No, 1 is allowed, P(1)=1. But 1+123456. P(123456)=643. So max is 643.
    # If we find a pair with max=2, that's impossible since sum is odd.
    # If we find max=3, that means both are 3-smooth.
    # We can stop if best_max == 2 (impossible) or maybe 3?
    # Let's just run the full loop. It's only 60k iterations.
    
    for a in range(1, target // 2 + 1):
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
            # If we find a very small max, we can break?
            # The smallest possible max is 2 (impossible) or 3.
            # If best_max == 3, it's likely optimal. But let's be safe and run a bit more or just finish.
            # Given the small N, finishing the loop is cheap.
            if best_max <= 3:
                break

    return [best_a, target - best_a]
