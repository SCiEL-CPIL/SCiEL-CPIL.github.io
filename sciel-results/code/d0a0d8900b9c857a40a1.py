import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # --- Phase 1: Smooth-number set expansion (fast, targeted search) ---
    # Numbers with large prime factors are unlikely to be optimal for minimizing
    # max(P(a), P(b)), so we search within the set of smooth numbers first.
    primes_to_test = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                       53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
                       109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                       173, 179, 181, 191, 193, 197, 199]

    smooth_set = {1}

    def find_pair_in_set(smooth_set, N):
        """Find a,b in smooth_set with a+b=N, minimizing max of largest prime factors."""
        best_val = float('inf')
        best_a = None
        for a in smooth_set:
            b = N - a
            if b in smooth_set and b > 0:
                # For smooth numbers, the largest prime factor is at most the
                # largest prime used in the set construction.
                # We'll compute exact P later; for now track candidates.
                if a > 0:
                    # Heuristic: prefer pairs where both are "more smooth"
                    # We'll refine with exact P computation
                    if best_a is None:
                        best_a = a
                        best_val = b  # placeholder
        return best_a

    # Build smooth set incrementally and check for pairs
    best_a_smooth = None
    best_max_pf = float('inf')

    for p in primes_to_test:
        if time.time() > deadline:
            break

        # Expand smooth_set with prime p
        current_size = len(smooth_set)
        new_numbers = set()
        for s in list(smooth_set):
            val = s
            while val <= N:
                if val != s:  # skip the original (already in set)
                    new_numbers.add(val)
                val *= p
                if val > N:
                    break
        smooth_set.update(new_numbers)

        # Check for pairs where both a and N-a are in smooth_set
        # and compute their actual largest prime factors
        found = False
        for a in smooth_set:
            if time.time() > deadline:
                break
            b = N - a
            if b in smooth_set and b > 0 and a > 0:
                # Compute P(a) and P(b) - they are smooth so factorization is fast
                pa = _largest_prime_factor(a)
                pb = _largest_prime_factor(b)
                m = max(pa, pb)
                if m < best_max_pf:
                    best_max_pf = m
                    best_a_smooth = a
                    found = True

    if best_a_smooth is not None:
        return [best_a_smooth, N - best_a_smooth]

    # --- Phase 2: Full LPF sieve fallback (exact global search) ---
    if time.time() > deadline:
        return [1, N - 1]

    # Build smallest prime factor sieve
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Compute largest prime factor for each number
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            if p > max_pf:
                max_pf = p
            while n % p == 0:
                n //= p
        lpf[i] = max_pf

    # Find optimal a
    best_a = 1
    best_val = max(lpf[1], lpf[N - 1])
    for a in range(2, N // 2 + 1):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a

    return [best_a, N - best_a]


def _largest_prime_factor(n):
    """Compute largest prime factor of n. Fast for smooth numbers."""
    if n <= 1:
        return 1
    max_pf = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            max_pf = max(max_pf, d)
            n //= d
        d += 1
    if n > 1:
        max_pf = max(max_pf, n)
    return max_pf
