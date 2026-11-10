import time

def construct():
    N = 123457
    deadline = time.time() + 5.0  # Very fast, no need for 85s

    # Primes to consider for smoothness
    # We want to minimize max(P(a), P(b)).
    # Try increasing bounds on the largest prime factor.
    
    # 1. Check for 7-smooth numbers (primes 2, 3, 5, 7)
    # If a pair exists, max(P) <= 7.
    # 2. Check for 11-smooth numbers (primes 2, 3, 5, 7, 11)
    # If a pair exists, max(P) <= 11.
    
    def get_smooth_numbers(primes, limit):
        smooths = {1}
        current = [1]
        for p in primes:
            new_vals = []
            for x in current:
                val = x
                while val * p <= limit:
                    val *= p
                    new_vals.append(val)
                    smooths.add(val)
            current.extend(new_vals)
        return smooths

    # Try B = 7
    primes_7 = [2, 3, 5, 7]
    smooths_7 = get_smooth_numbers(primes_7, N)
    for a in smooths_7:
        b = N - a
        if a >= 1 and b >= 1 and b in smooths_7:
            return [a, b]

    # Try B = 11
    primes_11 = [2, 3, 5, 7, 11]
    smooths_11 = get_smooth_numbers(primes_11, N)
    for a in smooths_11:
        b = N - a
        if a >= 1 and b >= 1 and b in smooths_11:
            return [a, b]

    # Fallback: Sieve and find min (should not be reached if 11-smooth pair exists)
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_max = N + 1
    best_a = 1
    best_b = N - 1
    for a in range(1, N):
        b = N - a
        m = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if m < best_max:
            best_max = m
            best_a = a
            best_b = b
    
    return [best_a, best_b]
