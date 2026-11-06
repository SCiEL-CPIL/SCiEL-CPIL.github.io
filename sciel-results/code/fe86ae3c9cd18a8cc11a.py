import time
import math

def construct():
    deadline = time.time() + 50.0
    
    def generate_smooth(primes, limit):
        """Generate all numbers <= limit composed of primes in 'primes'."""
        smooth_numbers = set()
        
        def dfs(idx, current):
            if current > limit:
                return
            if idx == len(primes):
                smooth_numbers.add(current)
                return
            # Option 1: do not use primes[idx]
            dfs(idx + 1, current)
            # Option 2: use primes[idx]
            p = primes[idx]
            while current * p <= limit:
                current *= p
                dfs(idx + 1, current)
                if time.time() > deadline:
                    return

        dfs(0, 1)
        return sorted(smooth_numbers)

    def find_pair(primes, low, high):
        # We need n and n+1 to be smooth.
        # So we generate smooth numbers up to high+1.
        # Then check for consecutive integers in the range [low, high].
        nums = generate_smooth(primes, high + 1)
        # Filter to range [low, high+1]
        # Actually, we just check if n is in nums and n+1 is in nums.
        # Using a set for O(1) lookup is better.
        num_set = set(nums)
        
        for n in range(low, high + 1):
            if n in num_set and (n + 1) in num_set:
                return n
        return None

    N_LOW = 10000
    N_HIGH = 1000000

    # Try 11-smooth (primes 2,3,5,7,11)
    # If a pair exists, max LPF is <= 11.
    result = find_pair([2, 3, 5, 7, 11], N_LOW, N_HIGH)
    if result is not None:
        return result

    # Try 13-smooth (primes 2,3,5,7,11,13)
    # If a pair exists, max LPF is <= 13.
    result = find_pair([2, 3, 5, 7, 11, 13], N_LOW, N_HIGH)
    if result is not None:
        return result

    # Fallback: General sieve method if no smooth pair found (should not happen)
    # This ensures we return a valid answer even if logic fails.
    M = N_HIGH + 1
    spf = list(range(M + 1))
    spf[0] = 0
    spf[1] = 1
    limit = int(math.isqrt(M))
    for i in range(2, limit + 1):
        if spf[i] == i:
            for j in range(i * i, M + 1, i):
                if spf[j] == j:
                    spf[j] = i

    lpf = [1] * (M + 1)
    for k in range(2, M + 1):
        p = spf[k]
        rest = k // p
        prev = lpf[rest]
        lpf[k] = p if p >= prev else prev

    best_n = N_LOW
    best_L = max(lpf[N_LOW], lpf[N_LOW + 1])
    for n in range(N_LOW + 1, N_HIGH + 1):
        L = max(lpf[n], lpf[n + 1])
        if L < best_L:
            best_L = L
            best_n = n
        if time.time() > deadline:
            break
    return best_n
