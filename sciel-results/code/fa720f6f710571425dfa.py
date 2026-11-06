import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_val = 1000024
    small_part = [1] * (max_val + 1)
    
    # Sieve-like approach: for each prime p, multiply small_part[x] by p^k
    # where p^k is the exact power of p dividing x
    for p in primes:
        for multiple in range(p, max_val + 1, p):
            val = multiple
            factor_part = 1
            while val % p == 0:
                factor_part *= p
                val //= p
            small_part[multiple] *= factor_part
    
    best_n = 1000
    min_distinct = 25  # max possible is 24, use 25 as initial
    
    # Sliding window with frequency count for O(1) per step
    # Window for n is [n+1, n+24]
    
    # Initialize frequency map for first window: n=1000, indices 1001 to 1024
    freq = {}
    for i in range(1001, 1025):
        v = small_part[i]
        freq[v] = freq.get(v, 0) + 1
    cur_count = len(freq)
    
    if cur_count < min_distinct:
        min_distinct = cur_count
        best_n = 1000
    
    if min_distinct <= 2:
        return best_n
    
    # Sliding window: move from n=1000 to n=999999
    # When going from n to n+1:
    #   Remove small_part[n+1], add small_part[n+25]
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Remove element at position n (which was n+1-1... wait)
        # Window for n-1 was [(n-1)+1, (n-1)+24] = [n, n+23]
        # Window for n is [n+1, n+24]
        # So remove small_part[n], add small_part[n+24]
        
        out_val = small_part[n]
        freq[out_val] -= 1
        if freq[out_val] == 0:
            del freq[out_val]
            cur_count -= 1
        
        in_val = small_part[n + 24]
        if in_val in freq:
            freq[in_val] += 1
        else:
            freq[in_val] = 1
            cur_count += 1
        
        if cur_count < min_distinct:
            min_distinct = cur_count
            best_n = n
            if min_distinct <= 2:
                break
    
    return best_n
