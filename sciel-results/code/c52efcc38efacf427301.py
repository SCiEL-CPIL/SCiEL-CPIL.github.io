import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    MAX_N = 1000000
    WINDOW = 24
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth part for all numbers up to MAX_N + WINDOW
    limit = MAX_N + WINDOW + 1
    smooth = list(range(limit + 1))  # smooth[m] will be the smooth part of m
    
    # For each small prime, divide it out completely from each multiple
    for p in small_primes:
        for m in range(p, limit + 1, p):
            # Remove all factors of p from smooth[m]
            # Actually, we want to KEEP the p-power, so we need a different approach
            pass
    
    # Better approach: for each number, compute its smooth part directly
    # smooth_part[m] = product of p^e for p in small_primes where p^e || m
    
    # Use a sieve-like method:
    # Initialize smooth_part[m] = 1 for all m
    # For each small prime p, for each multiple m of p, multiply smooth_part[m] by p for each factor of p in m
    
    smooth_part = [1] * (limit + 1)
    for p in small_primes:
        # For each power p^k, mark multiples
        power = p
        while power <= limit:
            for m in range(power, limit + 1, power):
                smooth_part[m] *= p
            power *= p
    
    # Now smooth_part[m] is the complete factor composed of primes strictly below 24
    # Verify: for m = 12 = 2^2 * 3, smooth_part should be 12
    # p=2: power=2, multiples: 2,4,6,8,10,12,... smooth_part[12]*=2 -> 2
    #       power=4, multiples: 4,8,12,16,... smooth_part[12]*=4 -> 8
    #       power=8, multiples: 8,16,... not 12
    # p=3: power=3, multiples: 3,6,9,12,... smooth_part[12]*=3 -> 24
    #       That's wrong! 12 = 2^2*3, smooth part should be 12, not 24.
    
    # The issue: when I multiply by p for each power, I'm overcounting.
    # For m=12, p=2: 12 is multiple of 2 (x1), multiple of 4 (x1), so I multiply by 2*4=8. But 2^2=4, so I should multiply by 4, not 8.
    
    # Correct approach: for each m, factor out small primes.
    # Or: for each p, for each m, count how many times p divides m, then multiply by p^count.
    
    # Let me redo this properly:
    smooth_part = [1] * (limit + 1)
    for p in small_primes:
        for m in range(p, limit + 1):
            if m % p == 0:
                count = 0
                temp = m
                while temp % p == 0:
                    count += 1
                    temp //= p
                smooth_part[m] *= (p ** count)
    
    # This is O(limit * sum(1/p)) which should be fine for limit ~ 1e6
    
    # Now find n in [1000, MAX_N] that minimizes distinct smooth parts in window [n+1, n+24]
    best_n = 1000
    best_count = WINDOW + 1
    
    # Slide window
    # For efficiency, use a frequency counter
    from collections import Counter
    
    # Initial window: n=1000, numbers 1001..1024
    freq = Counter()
    for i in range(1, WINDOW + 1):
        freq[smooth_part[1000 + i]] += 1
    
    best_count = len(freq)
    best_n = 1000
    
    for n in range(1001, MAX_N + 1):
        # Remove n (which was n-1+1 = old leftmost)
        old_val = smooth_part[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        # Add n+24
        new_val = smooth_part[n + WINDOW]
        freq[new_val] += 1
        
        cur_count = len(freq)
        if cur_count < best_count:
            best_count = cur_count
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
