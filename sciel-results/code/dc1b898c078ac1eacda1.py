import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Primes in increasing order
    # We want to minimize the largest prime factor, so we try smaller primes first
    prime_lists = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 5, 7, 11, 13, 17, 19, 23],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
    ]
    
    for primes in prime_lists:
        if time.time() > deadline:
            break
            
        # Generate k-smooth numbers up to N
        smooth = {1}
        for p in primes:
            new_smooth = set()
            for s in list(smooth):
                val = s
                while val * p <= N:
                    val *= p
                    new_smooth.add(val)
            smooth.update(new_smooth)
        
        # Check for pair a + b = N
        # To avoid duplicates and speed up, we can just check all a
        # If a is in smooth, and N-a is in smooth, we have a solution
        # We want to return [a, b]
        for a in sorted(smooth):
            if a > N:
                break
            b = N - a
            if b in smooth:
                # Ensure positive integers
                if a > 0 and b > 0:
                    return [a, b]
                    
    # Fallback
    return [1, 123456]
