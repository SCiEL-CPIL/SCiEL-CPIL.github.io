def construct():
    import math
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Primes to check in increasing order
    # We can just generate the first few primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def get_smooth_numbers(prime_list, limit):
        smooth = {1}
        for p in prime_list:
            new_smooth = set()
            for n in smooth:
                val = n
                while val <= limit:
                    new_smooth.add(val)
                    val *= p
            smooth.update(new_smooth)
        return smooth

    # Check for consecutive smooth numbers
    # We start with the smallest set of primes
    # If we use primes [2], we check 2-smooth numbers.
    # If we use primes [2,3], we check 3-smooth numbers.
    # Note: A number is 3-smooth if its prime factors are subset of {2,3}.
    # So checking [2,3] covers [2] as well.
    # We want the *minimum* largest prime factor.
    # So we should check P=2 first. Then P=3. Then P=5.
    
    for i in range(1, len(primes) + 1):
        current_primes = primes[:i]
        P = current_primes[-1]
        
        smooth_nums = get_smooth_numbers(current_primes, MAX_N + 1)
        
        # Check for n in [MIN_N, MAX_N] such that n and n+1 are in smooth_nums
        found = None
        for n in range(MIN_N, MAX_N + 1):
            if n in smooth_nums and (n + 1) in smooth_nums:
                found = n
                break
        
        if found is not None:
            return found
            
    # Fallback if no small prime factor pair found (unlikely)
    # Just return 10000 or find the best one via trial division
    return 10000
