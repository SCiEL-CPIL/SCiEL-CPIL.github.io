import math

def construct():
    max_n = 1000000
    min_n = 10000
    limit = max_n + 1
    
    # Generate powerful numbers up to limit
    # A number x is powerful if for every prime p dividing x, p^2 divides x.
    # This is equivalent to: x = a * b^2 where a is squarefree? No.
    # x is powerful iff x = y^2 * z where z is a product of distinct primes squared?
    # Simpler: Check each number.
    # Or generate them.
    
    # Let's generate them by iterating possible square factors.
    # x = s^2 * k.
    # Actually, just filter 1..limit.
    # To check if x is powerful:
    # Factorize x. For each p^e, if e>=2, ok. If e==1, not powerful.
    
    # We can use a sieve to find smallest prime factor (SPF) to speed up factorization.
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    powerfuls = []
    for x in range(1, limit + 1):
        is_pow = True
        temp = x
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count == 1:
                is_pow = False
                break
        if is_pow:
            powerfuls.append(x)
            
    # Sort descending to find large products first
    powerfuls.sort(reverse=True)
    
    best_val = 0
    best_n = -1
    
    # Iterate over pairs
    # We want to maximize A * B
    # n = A * S1, n+1 = B * S2
    # n is a multiple of A. n+1 is a multiple of B.
    # n = k*A. k*A + 1 = m*B => k*A - m*B = -1.
    # This Diophantine equation has solutions iff gcd(A,B)=1.
    # We need n in [10000, 1000000].
    
    for i, A in enumerate(powerfuls):
        # If A * max_powerful < best_val, we can break outer loop?
        # max B is powerfuls[0].
        if A * powerfuls[0] <= best_val:
            break
            
        for j, B in enumerate(powerfuls):
            prod = A * B
            if prod <= best_val:
                break
                
            if math.gcd(A, B) != 1:
                continue
                
            # Solve n = A * k, n + 1 = B * m
            # A * k = B * m - 1
            # A * k = -1 (mod B)
            # k = -A^{-1} (mod B)
            
            # Find inverse of A mod B
            try:
                inv_A = pow(A, -1, B)
            except ValueError:
                # gcd != 1, already checked, but just in case
                continue
                
            k = (-inv_A) % B
            # k is in [0, B-1]
            # n = A * k
            # But n must be positive. If k==0, n=0, not in range.
            # The general solution is n = A * (k + t*B) for integer t.
            # n = A*k + t*A*B.
            # Let base_n = A * k.
            # If base_n == 0, the smallest positive solution is A*B.
            # Actually, n = A * k. If k=0, n=0. Next is n=AB.
            # The solutions are n = A * k + t * (A * B).
            
            base_n = A * k
            modulus = A * B
            
            # We need n in [min_n, max_n]
            # n = base_n + t * modulus
            
            if base_n == 0:
                # Smallest positive is modulus
                first_n = modulus
            else:
                first_n = base_n
                
            # Find smallest t >= 0 such that first_n + t * modulus >= min_n
            # Actually, we can just find the range of t.
            # n_min_t = ceil((min_n - base_n) / modulus)
            # n_max_t = floor((max_n - base_n) / modulus)
            
            if base_n > max_n:
                continue
                
            # Calculate t_min
            if base_n >= min_n:
                t_min = 0
            else:
                t_min = math.ceil((min_n - base_n) / modulus)
                
            # Calculate t_max
            t_max = math.floor((max_n - base_n) / modulus)
            
            if t_min <= t_max:
                # Valid n exists
                # Pick the smallest valid n? Or any?
                # The value is A*B, which is constant for this pair.
                # We just need to record the best A*B and a corresponding n.
                if prod > best_val:
                    best_val = prod
                    # Choose a valid n
                    n_candidate = base_n + t_min * modulus
                    if n_candidate < min_n:
                         n_candidate = base_n + (t_min+1)*modulus # Should not happen if ceil is correct
                    # Ensure n_candidate is in range
                    if min_n <= n_candidate <= max_n:
                        best_n = n_candidate
                    else:
                        # Try to find a valid one
                        # t_min might have been 0 but base_n < min_n?
                        # If base_n < min_n, t_min is at least 1.
                        # So n_candidate should be >= min_n.
                        pass

    if best_n == -1:
        # Fallback: should not happen as 1 is powerful, A=1, B=1 -> n=0, n+1=1.
        # But n must be >= 10000.
        # If no pair found, return a default?
        # There is always a solution. e.g., n=10000.
        # Powerful part of 10000 = 10000 (100^2).
        # Powerful part of 10001 = 101*99 = 101*9*11. 9 is 3^2. So powerful part is 9.
        # Product = 90000.
        # Our loop should find this.
        # A=10000, B=9. gcd(10000,9)=1.
        # n is multiple of 10000. n=10000.
        # n+1=10001. 10001 % 9 == 1? 10001 = 9*1111 + 2. No.
        # So n=10000 is not a solution for A=10000, B=9.
        # But the algorithm finds the best A*B.
        
        # If best_n is still -1, something went wrong.
        # Return 10000 as a safe bet?
        return 10000

    return best_n
