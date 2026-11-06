from collections import Counter

def construct():
    MAX_N = 1000000
    MAX_M = MAX_N + 24
    primes_below_24 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth parts
    smooth = [1] * (MAX_M + 1)
    for p in primes_below_24:
        # Optimization: Use local variables for speed
        sm = smooth
        for multiple in range(p, MAX_M + 1, p):
            count = 0
            temp = multiple
            while temp % p == 0:
                count += 1
                temp //= p
            if count > 0:
                sm[multiple] *= p ** count
    
    best_n = 1000
    best_count = 25
    
    # Sliding window
    window_freq = Counter()
    # Initial window for n=1000: indices 1001 to 1024
    for i in range(1, 25):
        window_freq[smooth[1000 + i]] += 1
    best_count = len(window_freq)
    best_n = 1000
    
    # Slide window
    for n in range(1001, MAX_N + 1):
        val_remove = smooth[n]
        val_add = smooth[n + 24]
        
        # Update counts
        window_freq[val_remove] -= 1
        if window_freq[val_remove] == 0:
            del window_freq[val_remove]
            
        window_freq[val_add] += 1
        
        count = len(window_freq)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
