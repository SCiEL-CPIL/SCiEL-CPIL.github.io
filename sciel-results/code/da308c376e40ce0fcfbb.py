def construct():
    from math import gcd
    
    N = 512
    
    # Precompute divisors for all numbers up to N
    # divs[i] is a list of divisors of i
    divs = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            divs[j].append(i)
    
    # Find M with maximum number of divisors
    best_M = 1
    max_divs = 0
    for M in range(1, N + 1):
        if len(divs[M]) > max_divs:
            max_divs = len(divs[M])
            best_M = M
            
    # The set of divisors of best_M is a valid candidate
    candidate_set = set(divs[best_M])
    
    # Try to see if we can add more elements to this set
    # We check all x not in candidate_set
    # If x is compatible with all in candidate_set, add it
    # Note: We proved earlier that if best_M is "large" (like 480), no x can be added.
    # But for smaller M, maybe?
    # Actually, if we pick M with max divs, it's likely 480 or 504.
    # For 480, no addition possible.
    
    # However, what if a union of two divisor sets is larger?
    # We already analyzed this. Single set is better.
    
    # Just to be safe, run a greedy expansion on the best_M divisors
    # And also try a few other high-divisor M's
    
    best_size = len(candidate_set)
    best_result = list(candidate_set)
    
    # Try all M with high divisor count
    # Sort M by divisor count descending
    M_candidates = sorted(range(1, N + 1), key=lambda x: len(divs[x]), reverse=True)
    
    for M in M_candidates[:20]: # Top 20
        current = set(divs[M])
        # Greedy add
        # To speed up, only try x that are compatible with M
        # lcm(x, M) <= N
        for x in range(1, N + 1):
            if x in current:
                continue
            if M * x // gcd(M, x) > N:
                continue
            # Check against all in current
            valid = True
            for y in current:
                if x * y // gcd(x, y) > N:
                    valid = False
                    break
            if valid:
                current.add(x)
                
        if len(current) > best_size:
            best_size = len(current)
            best_result = list(current)
            
    return best_result
