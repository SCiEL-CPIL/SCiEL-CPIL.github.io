def construct():
    import time
    start_time = time.time()
    DEADLINE = 85
    
    N = 123457
    
    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # Iterate K from 1 to N
    # But we can stop early.
    # We'll iterate K up to a limit, say 100, or until time is up.
    # If not found, fallback to full scan.
    
    for K in range(1, 100):
        if time.time() - start_time > DEADLINE:
            break
            
        # Get primes <= K
        # Since primes are sorted, we can find the index
        # Or just filter
        p_list = [p for p in primes if p <= K]
        
        if not p_list:
            # Only 1 is K-smooth
            if N == 2: return [1, 1] # N is 123457, so no
            continue
            
        # Generate all K-smooth numbers <= N
        smooth_nums = {1}
        # We can do this by BFS or simple iteration
        # Start with 1
        # For each prime, multiply existing elements
        # This is not strictly correct for "all combinations", it's better to use recursion or a heap.
        # Simpler:
        # Initialize list with 1
        # For each prime p in p_list:
        #   new_nums = []
        #   for x in current_list:
        #     while x * p <= N:
        #       x *= p
        #       new_nums.append(x)
        #   current_list += new_nums
        # This generates all products of powers of the first i primes.
        
        current_list = [1]
        for p in p_list:
            new_nums = []
            for x in current_list:
                val = x
                while val * p <= N:
                    val *= p
                    new_nums.append(val)
            current_list.extend(new_nums)
        
        # Convert to set for O(1) lookup
        smooth_set = set(current_list)
        
        # Check for a, b in smooth_set such that a+b=N
        # a must be in smooth_set, b = N-a must be in smooth_set
        # Also a, b >= 1
        # Iterate over smooth_set
        # To avoid duplicate checks (a,b) and (b,a), we can just check all.
        found = False
        for a in smooth_set:
            if a > N: continue
            b = N - a
            if b >= 1 and b in smooth_set:
                # Found
                return [a, b]
                
    # Fallback: Full scan with LPF sieve
    # This ensures we always return a valid answer
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
        if best_max_pf == 1:
            break
            
    return [best_a, best_b]
