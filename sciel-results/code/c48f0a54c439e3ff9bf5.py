<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Phase 0: Try p=2 first
    k = 0
    while k < 200 and all_factors_at_most_p(4 + k + 1, 2):
        k += 1
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Generate primes up to 100000 using sieve
    primes_100k = sieve_primes(100000)
    
    # Phase 1: Check all primes up to 100000
    for p in primes_100k:
        if time.time() > deadline:
            break
        if p == 2:
            continue  # Already checked
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() > deadline:
                break
            if not all_factors_at_most_p(p_sq + k + 1, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Check primes from 100000 to 1000000 using a larger sieve
    if time.time() < deadline - 5:
        primes_1m = sieve_primes(1000000)
        # Only check primes > 100000
        for p in primes_1m:
            if time.time() > deadline:
                break
            if p <= 100000:
                continue
            
            k = 0
            p_sq = p * p
            # For larger p, limit the check to save time
            max_check = min(200, 100)
            while k < max_check:
                if time.time() > deadline:
                    break
                if not all_factors_at_most_p(p_sq + k + 1, p):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # Phase 3: If we still have time, try some larger primes
    # Use a segmented approach or just try specific ranges
    if time.time() < deadline - 10:
        # Try primes in ranges around powers of 10
        ranges_to_try = [(10**6, 2*10**6), (2*10**6, 5*10**6), (5*10**6, 10**7)]
        
        for low, high in ranges_to_try:
            if time.time() > deadline - 5:
                break
            
            # Generate primes in this range using a simple trial division approach
            # This is slower but we only do it if we have time
            for p in range(low, high, 2):
                if time.time() > deadline:
                    break
                if not is_prime(p):
                    continue
                
                k = 0
                p_sq = p * p
                max_check = min(200, 50)
                while k < max_check:
                    if time.time() > deadline:
                        break
                    if not all_factors_at_most_p(p_sq + k + 1, p):
                        break
                    k += 1
                
                if k > best_k:
                    best_k = k
                    best_p = p
                
                # Limit the number of primes we check per range to avoid timeout
                if (p - low) > 50000:
                    break
    
    return {'p': best_p, 'k': best_k}
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Phase 0: Try p=2 first
    k = check_run_length(2, 200, deadline)
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Generate primes up to 200000 using sieve
    primes_200k = sieve_primes(200000)
    
    # Phase 1: Check all primes up to 200000
    for p in primes_200k:
        if time.time() > deadline:
            break
        if p == 2:
            continue  # Already checked
        
        k = check_run_length(p, 200, deadline)
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Check primes from 200000 to 2000000 using a larger sieve
    if time.time() < deadline - 5:
        primes_2m = sieve_primes(2000000)
        # Only check primes > 200000
        for p in primes_2m:
            if time.time() > deadline:
                break
            if p <= 200000:
                continue
            
            k = check_run_length(p, 200, deadline)
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # Phase 3: If we still have time, use segmented sieve for larger primes
    if time.time() < deadline - 10:
        # Use segmented sieve to generate primes in larger ranges
        def segmented_sieve(low, high):
            """Generate primes in [low, high] using segmented sieve."""
            if low < 2:
                low = 2
            if high < low:
                return []
            
            # Get primes up to sqrt(high)
            limit = int(math.isqrt(high)) + 1
            base_primes = sieve_primes(limit)
            
            # Segmented sieve
            size = high - low + 1
            is_prime_seg = [True] * size
            
            for p in base_primes:
                # Find the first multiple of p in [low, high]
                start = ((low + p - 1) // p) * p
                if start == p:
                    start += p
                for j in range(start - low, size, p):
                    is_prime_seg[j] = False
            
            return [low + i for i in range(size) if is_prime_seg[i]]
        
        # Try primes in ranges, prioritizing denser ranges first
        ranges_to_try = [(2*10**6, 10*10**6), (10*10**6, 50*10**6), (50*10**6, 200*10**6)]
        
        for low, high in ranges_to_try:
            if time.time() > deadline - 5:
                break
            
            # Generate primes in this range using segmented sieve
            try:
                primes_in_range = segmented_sieve(low, high)
            except MemoryError:
                # Fall back to smaller segments
                seg_size = 100000
                for seg_low in range(low, high, seg_size):
                    if time.time() > deadline - 2:
                        break
                    seg_high = min(seg_low + seg_size - 1, high)
                    try:
                        primes_in_seg = segmented_sieve(seg_low, seg_high)
                    except MemoryError:
                        continue
                    for p in primes_in_seg:
                        if time.time() > deadline:
                            break
                        k = check_run_length(p, 200, deadline)
                        if k > best_k:
                            best_k = k
                            best_p = p
                continue
            
            for p in primes_in_range:
                if time.time() > deadline:
                    break
                k = check_run_length(p, 200, deadline)
                if k > best_k:
                    best_k = k
                    best_p = p
    
    return {'p': best_p, 'k': best_k}
>>>>>>> REPLACE
