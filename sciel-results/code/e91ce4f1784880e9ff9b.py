import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need smooth parts for numbers up to 1000000 + 24 = 1000024
    max_val = 1000024
    sp = [1] * (max_val + 1)
    
    # Sieve: for each small prime p, multiply sp[m] by p^exp for each multiple m of p
    # This is O(N log log N) roughly
    for p in small_primes:
        for m in range(p, max_val + 1, p):
            temp = m
            exp = 0
            while temp % p == 0:
                exp += 1
                temp //= p
            if exp > 0:
                sp[m] *= (p ** exp)
    
    best_n = 1000
    best_count = float('inf')
    
    # Sliding window over sp[n+1 : n+25] for n in [1000, 1000000]
    # Window for n: indices n+1 to n+24 (inclusive)
    
    # Initialize counter for n=1000: window [1001, 1024]
    freq = {}
    distinct_count = 0
    
    for i in range(1001, 1025):
        val = sp[i]
        if val not in freq:
            freq[val] = 1
            distinct_count += 1
        else:
            freq[val] += 1
            
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = 1000
    
    if best_count == 1:
        return best_n
    
    # Slide window: for new_n from 1001 to 1000000
    # Old window (for new_n-1): [new_n, new_n+23]
    # New window (for new_n): [new_n+1, new_n+24]
    # Remove sp[new_n], add sp[new_n+24]
    for new_n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Remove leftmost
        old_val = sp[new_n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct_count -= 1
            # Optional: del freq[old_val] to save memory, but keeping it allows faster lookup
            # Since we only check existence for new values, and decrement for old, 
            # keeping zero entries is fine as long as we don't count them in distinct_count.
            # However, if a value re-enters, we need to know if it was 0.
            # So we must keep the key or check carefully.
            # If we delete, we need to check `if val not in freq` on add.
            # If we don't delete, we check `if freq[val] == 0` on add.
            # Not deleting is slightly faster for lookup? Maybe. Let's keep it simple.
            
        # Add new rightmost
        new_val = sp[new_n + 24]
        if new_val not in freq:
            freq[new_val] = 1
            distinct_count += 1
        else:
            # If it was 0, we already handled the distinct_count increment above?
            # No, if it was 0, it's in freq with value 0.
            # So we just increment. But we need to increment distinct_count if it was 0.
            if freq[new_val] == 0:
                distinct_count += 1
            freq[new_val] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = new_n
            if best_count == 1:
                break
    
    return best_n
