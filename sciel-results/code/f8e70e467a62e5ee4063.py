import math

def construct():
    N = 10000000
    
    # Step 1: Sieve primes up to sqrt(N) + some margin
    # sqrt(10^7) is ~3162. Let's use 10000 to be safe.
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def sigma(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                # Factor out p
                power_sum = 1
                power_val = 1
                while temp % p == 0:
                    temp //= p
                    power_val *= p
                    power_sum += power_val
                result *= power_sum
        if temp > 1:
            # temp is prime
            result *= (1 + temp)
        return result

    best_error_num = float('inf')
    best_error_den = 1
    best_a = None
    best_b = None

    # Iterate b downwards
    # We can skip even b if we find a good odd b solution?
    # No, just iterate all.
    
    # Optimization: 
    # If best_error is 1/(2*b_prev), we only need to check b > b_prev/2? No.
    # 1/(2b) < best_error => b > 1/(2*best_error).
    
    for b in range(N, 0, -1):
        # Check if this b can possibly beat the current best
        # Min error for this b is 1/(2b)
        # If 1/(2b) >= best_error, then no a for this b can beat best_error.
        # So we can break if b is small enough.
        # best_error is stored as num/den.
        # 1/(2b) >= num/den  => den >= 2b * num => b <= den / (2*num)
        
        if best_a is not None:
            # Check if 1/(2b) >= best_error_num / best_error_den
            # 1/(2b) >= N/D => D >= 2bN => b <= D/(2N)
            max_b_allowed = best_error_den // (2 * best_error_num)
            if b <= max_b_allowed:
                break
        
        s_b = sigma(b)
        
        # Determine candidates for a
        if b % 2 == 0:
            # a = 1.5b is integer, but 2a-3b=0, invalid.
            # Nearest are 1.5b - 1 and 1.5b + 1
            a_mid = (3 * b) // 2
            candidates = [a_mid - 1, a_mid + 1]
        else:
            # 1.5b is k.5
            # Nearest are floor and ceil
            a_low = (3 * b - 1) // 2
            a_high = (3 * b + 1) // 2
            candidates = [a_low, a_high]
        
        for a in candidates:
            if a < 1 or a > N or a == b:
                continue
            
            s_a = sigma(a)
            if s_a == s_b:
                # Calculate error
                err_num = abs(2 * a - 3 * b)
                err_den = 2 * b
                
                if err_num == 0:
                    continue
                
                # Compare err_num/err_den < best_error_num/best_error_den
                if best_a is None or err_num * best_error_den < best_error_num * err_den:
                    best_error_num = err_num
                    best_error_den = err_den
                    best_a = a
                    best_b = b

    if best_a is None:
        # Fallback
        return [11, 6]
    
    return [int(best_a), int(best_b)]
