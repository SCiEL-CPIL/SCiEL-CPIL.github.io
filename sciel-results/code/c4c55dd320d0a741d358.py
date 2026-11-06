import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    max_val = max_n + 24  # we need up to n+24
    
    # Compute 23-smooth part for all numbers from 1 to max_val
    smooth = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        temp = x
        s = 1
        for p in primes:
            while temp % p == 0:
                s *= p
                temp //= p
        smooth[x] = s
    
    # Now slide a window of size 24 over n from 1000 to max_n
    # For each n, look at smooth[n+1] ... smooth[n+24]
    # Count distinct values
    
    best_count = 24
    best_n = 1000
    
    # Use a sliding window approach with a count dict
    from collections import defaultdict
    
    # Initialize window for n=1000: indices 1001 to 1024
    counts = defaultdict(int)
    for i in range(1001, 1025):
        counts[smooth[i]] += 1
    
    distinct = len(counts)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, max_n + 1):
        # Remove smooth[n] (the element leaving the window)
        leaving = smooth[n]
        counts[leaving] -= 1
        if counts[leaving] == 0:
            del counts[leaving]
        
        # Add smooth[n+24] (the element entering the window)
        entering = smooth[n + 24]
        counts[entering] += 1
        
        distinct = len(counts)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count <= 2:  # Can't do much better than 2
                break
        
        # Time check
        if time.time() > deadline:
            break
    
    return best_n
