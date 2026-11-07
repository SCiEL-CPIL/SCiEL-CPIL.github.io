import time
import sys

def construct():
    start_time = time.time()
    N = 10_000_000
    
    # Step 1: Sieve for Smallest Prime Factor (SPF)
    # This allows fast computation of sigma(n)
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Helper to compute sigma(n) using SPF
    def get_sigma(n):
        if n <= 1:
            return n
        res = 1
        while n > 1:
            p = spf[n]
            power = 1
            while n % p == 0:
                n //= p
                power *= p
            res *= (power * (p + 1) // (p - 1))
        return res

    # Step 2: Generate offsets
    # We want to check offsets (da, db) such that |2*da - 3*db| is small.
    # We'll generate a list of tuples (score, da, db) where score = |2*da - 3*db|
    # We limit the range of da, db to [-100, 100] initially.
    # If no solution is found quickly, we might need to expand, but likely not.
    offsets = []
    max_delta = 100
    for da in range(-max_delta, max_delta + 1):
        for db in range(-max_delta, max_delta + 1):
            val = abs(2 * da - 3 * db)
            if val == 0:
                continue # This corresponds to a/b = 1.5 exactly, which is invalid
            offsets.append((val, abs(da) + abs(db), da, db))
    
    # Sort by val, then by magnitude
    offsets.sort()
    
    # Step 3: Search k from high to low
    # k_max is approx N/3
    k_max = N // 3
    # We also need to ensure b = 2k + db <= N and a = 3k + da <= N
    # So k <= (N - max_da) / 3 and k <= (N - max_db) / 2
    # Let's just check bounds inside the loop.
    
    # Time limit check
    # If we don't find anything in 80 seconds, return a fallback.
    # But this search should be very fast if a solution exists.
    
    for k in range(k_max, 0, -1):
        # Check time
        if time.time() - start_time > 80:
            # Fallback to a known valid pair or seed
            # Seed was [11, 6]. sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = |1.833 - 1.5| = 0.333
            return [11, 6]
            
        for val, mag, da, db in offsets:
            a = 3 * k + da
            b = 2 * k + db
            
            # Bounds check
            if a < 1 or a > N or b < 1 or b > N:
                continue
                
            # Check if ratio is exactly 3/2
            # 2a == 3b
            if 2 * a == 3 * b:
                continue
                
            # Compute sigmas
            sa = get_sigma(a)
            sb = get_sigma(b)
            
            if sa == sb:
                return [a, b]
                
    # If no solution found (should not happen), return fallback
    return [11, 6]
