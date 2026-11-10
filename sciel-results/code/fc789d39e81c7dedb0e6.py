import math

def construct():
    max_n = 1000000
    limit = max_n + 1
    
    # 1. Generate all powerful numbers up to limit
    # Powerful numbers are of the form a^2 * b^3
    powerful_set = set()
    # a^2
    for a in range(1, int(math.isqrt(limit)) + 1):
        powerful_set.add(a * a)
    # b^3
    for b in range(1, int(round(limit ** (1/3))) + 2):
        powerful_set.add(b * b * b)
    # a^2 * b^3
    # b^3 <= limit => b <= 100
    # a^2 <= limit / b^3 => a <= sqrt(limit / b^3)
    for b in range(1, int(round(limit ** (1/3))) + 2):
        b3 = b * b * b
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            powerful_set.add(a * a * b3)
            
    powerful_list = list(powerful_set)
    
    # 2. Compute prime support bitmask for each powerful number
    # Primes up to 1000 are sufficient for factors of numbers <= 10^6
    # 1000 is 2^10, so we need primes up to 1000.
    # There are 168 primes up to 1000.
    # We can map prime to bit index.
    
    # Sieve for primes up to 1000
    primes = []
    is_prime = [True] * 1001
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, 1001):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 1001, i):
                is_prime[j] = False
                
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    def get_support_mask(x):
        mask = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << prime_to_bit[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is a prime factor
            if temp in prime_to_bit:
                mask |= (1 << prime_to_bit[temp])
            else:
                # This shouldn't happen if primes list is complete up to sqrt(limit)
                # But limit is 10^6, sqrt is 1000.
                # If x is a prime > 1000, it can't divide a powerful number <= 10^6 
                # unless the powerful number is that prime itself? No, powerful numbers have exponents >= 2.
                # So any prime factor of a powerful number <= 10^6 must be <= 1000.
                pass
        return mask

    # Store (value, mask)
    powerful_data = []
    for val in powerful_list:
        mask = get_support_mask(val)
        powerful_data.append((val, mask))
        
    # Sort by value descending to allow pruning
    powerful_data.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Precompute is_squarefree
    is_sqfree = [True] * (limit + 1)
    is_sqfree[0] = False # 0 is not squarefree? Definition: squarefree if no square > 1 divides it.
    # Usually 1 is squarefree.
    for p in primes:
        p2 = p * p
        if p2 > limit:
            break
        for j in range(p2, limit + 1, p2):
            is_sqfree[j] = False
            
    # 4. Search
    best_val = 0
    best_n = -1
    n_pow = len(powerful_data)
    
    for i in range(n_pow):
        A, mask_A = powerful_data[i]
        # If A * max_B <= best_val, break
        if A * powerful_data[0][0] <= best_val:
            break
            
        for j in range(n_pow):
            B, mask_B = powerful_data[j]
            if A * B <= best_val:
                break
                
            if mask_A & mask_B:
                continue
                
            # Solve n = A*k, n+1 = B*m
            # A*k == -1 mod B
            # k == -A^-1 mod B
            try:
                inv_A = pow(A, -1, B)
            except ValueError:
                # Should not happen if masks are disjoint
                continue
                
            k_base = (-inv_A) % B
            n_base = A * k_base
            step = A * B
            
            # Find k range such that 10000 <= n_base + k*step <= 1000000
            # k >= (10000 - n_base) / step
            # k <= (1000000 - n_base) / step
            
            k_min = math.ceil((10000 - n_base) / step)
            k_max = math.floor((1000000 - n_base) / step)
            
            if k_min > k_max:
                continue
                
            # Iterate k
            # We can optimize by checking if any n in this arithmetic progression has squarefree parts
            # But step can be large, so number of terms is small.
            for k in range(k_min, k_max + 1):
                n = n_base + k * step
                S1 = n // A
                S2 = (n + 1) // B
                
                if is_sqfree[S1] and is_sqfree[S2]:
                    val = A * B
                    if val > best_val:
                        best_val = val
                        best_n = n
                        
    return best_n
