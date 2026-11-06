import time

def construct():
    start_time = time.time()
    DEADLINE = 80  # seconds, safety margin before 90s kill
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 24  # we need up to n+24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Compute 23-smooth part for each number up to MAX_VAL
    smooth = list(range(MAX_VAL + 1))
    for p in PRIMES:
        for i in range(p, MAX_VAL + 1, p):
            while smooth[i] % p == 0:
                smooth[i] //= p
    
    # Now smooth[i] is the part of i with all primes < 24 removed
    # The 23-smooth part of i is i // smooth[i]
    # Actually, smooth[i] after division is the "rough" part
    # The smooth part = original_i // smooth[i]
    
    # Let's recompute: for each i, smooth_part[i] = largest divisor of i using only primes in PRIMES
    smooth_part = [1] * (MAX_VAL + 1)
    for i in range(2, MAX_VAL + 1):
        x = i
        sp = 1
        for p in PRIMES:
            while x % p == 0:
                sp *= p
                x //= p
        smooth_part[i] = sp
    
    # This is O(MAX_VAL * sum of exponents) which might be slow
    # Better: use the sieve approach
    smooth_part = [1] * (MAX_VAL + 1)
    for p in PRIMES:
        for i in range(p, MAX_VAL + 1, p):
            smooth_part[i] *= p  # This is wrong, need to handle powers
    
    # Correct sieve approach:
    smooth_part = [1] * (MAX_VAL + 1)
    for p in PRIMES:
        power = p
        while power <= MAX_VAL:
            for i in range(power, MAX_VAL + 1, power):
                smooth_part[i] *= p
            power *= p
    
    # Hmm, this overcounts. Let me use a different approach.
    # For each number, divide out all factors of each prime.
    
    # Most efficient: initialize smooth_part[i] = i, then for each prime p, divide out all p factors
    smooth_part = list(range(MAX_VAL + 1))
    for p in PRIMES:
        for i in range(p, MAX_VAL + 1, p):
            while smooth_part[i] % p == 0:
                smooth_part[i] //= p
    # Now smooth_part[i] is the rough part (coprime to product of primes < 24)
    # The 23-smooth part of i is i // smooth_part[i]
    
    # Actually, let me just store the smooth part directly
    # smooth_part[i] after the above = i with all small prime factors removed = rough part
    # smooth value = i // rough_part
    
    # For counting distinct smooth parts in window [n+1, n+24]:
    # smooth_value[i] = i // smooth_part[i]
    
    # Let me precompute smooth_value
    smooth_value = [0] * (MAX_VAL + 1)
    for i in range(1, MAX_VAL + 1):
        smooth_value[i] = i // smooth_part[i]
    
    # Now search for n in [1000, 1000000] minimizing len(set(smooth_value[n+1:n+25]))
    best_n = 1000
    best_count = 100
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_n
    
    # Search
    for n in range(1000, MAX_N + 1):
        if time.time() - start_time > DEADLINE:
            break
        window = smooth_value[n+1:n+25]
        cnt = len(set(window))
        if cnt < best_count:
            best_count = cnt
            best_n = n
    
    return best_n
