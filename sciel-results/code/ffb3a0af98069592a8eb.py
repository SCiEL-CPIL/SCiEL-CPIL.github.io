import time

def construct():
    deadline = time.time() + 50.0
    N_MAX = 1000000
    N_MIN = 10000
    
    # Helper to generate P-smooth numbers up to limit
    def generate_smooth(primes, limit):
        # Using a set to store smooth numbers
        smooths = set([1])
        # We can use a priority queue or just iterate
        # Given the small limit, a simple recursive generation or iterative multiplication works
        # But we need all numbers up to 10^6.
        # Number of 11-smooth numbers is small.
        
        # Better: Sieve-like approach for specific primes
        # Or just generate all combinations
        import itertools
        
        # Max power for each prime
        max_powers = []
        for p in primes:
            power = 1
            count = 0
            while power * p <= limit:
                power *= p
                count += 1
            max_powers.append(count)
            
        # Generate all products
        # This might be slow if we do it naively for many primes, but primes list is small (2,3,5,7,11)
        # Max powers:
        # 2: ~19
        # 3: ~12
        # 5: ~8
        # 7: ~7
        # 11: ~5
        # Total combinations: 20 * 13 * 9 * 8 * 6 = 124,800. Very manageable.
        
        results = []
        
        def gen(idx, current_val):
            if idx == len(primes):
                if current_val <= limit:
                    results.append(current_val)
                return
            
            p = primes[idx]
            val = current_val
            for _ in range(max_powers[idx] + 1):
                if val > limit:
                    break
                gen(idx + 1, val)
                val *= p
        
        gen(0, 1)
        return sorted(results)

    # Check for consecutive pairs in a list of smooth numbers
    def find_best_consecutive(smooths, n_min, n_max):
        best_n = None
        best_score = float('inf')
        # Convert to set for O(1) lookup
        smooth_set = set(smooths)
        
        for n in smooths:
            if n < n_min or n > n_max:
                continue
            if (n + 1) in smooth_set:
                # Both n and n+1 are smooth
                # Calculate max LPF
                # Since they are P-smooth, max LPF is at most P
                # We want to minimize max LPF.
                # For 7-smooth, max LPF is 7.
                # For 11-smooth, max LPF is 11.
                # We should check smaller P first.
                score = max(7, 11) # Placeholder, determined by caller
                # Actually, we just need to return n if found.
                # The caller will know the prime bound.
                return n
        return None

    # Strategy:
    # 1. Check 7-smooth (primes 2,3,5,7)
    # 2. Check 11-smooth (primes 2,3,5,7,11)
    # 3. Fallback to full sieve for 13-smooth or lower
    
    # 1. Check 7-smooth
    smooth_7 = generate_smooth([2, 3, 5, 7], N_MAX + 1)
    n_7 = find_best_consecutive(smooth_7, N_MIN, N_MAX)
    if n_7 is not None:
        return n_7

    # 2. Check 11-smooth
    smooth_11 = generate_smooth([2, 3, 5, 7, 11], N_MAX + 1)
    n_11 = find_best_consecutive(smooth_11, N_MIN, N_MAX)
    if n_11 is not None:
        return n_11

    # 3. Fallback: Full Sieve
    # This will find the true minimum, which is expected to be 13.
    limit = N_MAX + 1
    lpf = [0] * (limit + 1)
    
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            for j in range(i, limit + 1, i):
                lpf[j] = i
        if i % 10000 == 0 and time.time() > deadline:
            break

    best_n = N_MIN
    best_score = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN + 1, N_MAX + 1):
        current_score = max(lpf[n], lpf[n + 1])
        if current_score < best_score:
            best_score = current_score
            best_n = n
        if (n - N_MIN) % 10000 == 0 and time.time() > deadline:
            break

    return best_n
