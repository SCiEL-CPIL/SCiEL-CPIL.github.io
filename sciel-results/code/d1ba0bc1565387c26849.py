def construct():
    N = 123457
    
    # Helper to generate boolean array of smooth numbers
    def get_smooth_bool(primes, limit):
        is_smooth = bytearray(limit + 1)
        is_smooth[1] = 1
        
        # Generate all smooth numbers using a list
        smooths = [1]
        for p in primes:
            new_smooths = []
            for s in smooths:
                val = s
                while val * p <= limit:
                    val *= p
                    new_smooths.append(val)
            smooths.extend(new_smooths)
        
        # Mark in boolean array
        for s in smooths:
            if s <= limit:
                is_smooth[s] = 1
                
        return is_smooth

    # Check primes in increasing order of max prime factor
    # We check 5-smooth, then 7-smooth, then 11-smooth, etc.
    # This follows the directive to use boolean arrays and linear scan.
    
    prime_limits = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for p_max in prime_limits:
        primes = [p for p in all_primes if p <= p_max]
        is_smooth = get_smooth_bool(primes, N)
        
        # Linear scan for the first valid pair
        for a in range(1, N // 2 + 1):
            if is_smooth[a] and is_smooth[N - a]:
                return [a, N - a]
                
    # Fallback if no small prime factor solution found
    return [1, N - 1]
