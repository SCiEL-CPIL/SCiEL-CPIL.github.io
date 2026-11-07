import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin for 90s limit
    N = 10_000_000

    # Known good fallback
    best_a, best_b = 122, 80  # sigma(122)=186, sigma(80)=186, |122/80-1.5| = 1/40
    best_err_num, best_err_den = 1, 40

    def check_deadline():
        return time.time() - start_time > deadline

    def factorize(n):
        """Return list of (prime, exponent) pairs."""
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt = 0
                while n % d == 0:
                    n //= d
                    cnt += 1
                factors.append((d, cnt))
            d += 1 if d == 2 else 2
        if n > 1:
            factors.append((n, 1))
        return factors

    def sigma_from_factors(factors):
        """Compute sigma from factorization (prime, exp) list."""
        result = 1
        for p, e in factors:
            result *= (p**(e+1) - 1) // (p - 1)
        return result

    def sigma(n):
        if n <= 0:
            return 0
        return sigma_from_factors(factorize(n))

    # Strategy: search for pairs where a/b is close to 3/2
    # a/b = 3/2 + eps, where eps = k/(2b) for integer k
    # So a = (3b + k) / 2, requiring 3b+k to be even
    # Error = k/(2b), minimize this
    
    # For small k, search large b values
    # We'll search k from 1 upward, and for each k, search b from large to small
    
    max_b = (2 * N) // 3  # a = (3b+k)/2 <= N => b <= (2N-k)/3 < 2N/3
    
    # Precompute a small set of known sigma-equal pairs for quick reference
    # by checking small candidates near ratio 3/2
    
    # Search k = 1, 2, 3, ... 
    # For each k, we need b such that:
    #   1. (3b + k) is even
    #   2. a = (3b+k)/2 <= N
    #   3. sigma(a) == sigma(b)
    #   4. k/(2b) is minimized => maximize b
    
    # To speed up, we can check b values in descending order for each k
    # and stop when we find a match (largest b gives smallest error for that k)
    # But we also need to check if a smaller k with slightly smaller b could beat it
    
    # Better approach: for each k, find the LARGEST b that works
    # Then compare k/(2*b) across all k values
    
    # Limit k search to reasonable range
    max_k = 200
    
    for k in range(1, max_k + 1):
        if check_deadline():
            break
        
        # b must satisfy: 3b + k is even
        # 3b + k even => b and k have same parity (3 is odd)
        # Also b >= 1, a = (3b+k)/2 >= 1, a <= N
        # b <= (2N - k) / 3
        
        b_max = (2 * N - k) // 3
        if b_max < 1:
            break
            
        # Start from largest b and go down
        # But checking every b is too slow. 
        # Instead, we'll sample strategically or use a smarter approach.
        
        # For this k, the error at b_max is k/(2*b_max)
        # If this is already worse than best_err, we can skip this k
        # (since smaller b gives larger error for same k)
        current_k_min_err = k / (2 * b_max)
        if current_k_min_err > best_err_num / best_err_den:
            # Even the best b for this k can't beat current best
            # But wait - we should still check if there's a match
            # because maybe a larger k with even larger b could work
            # Actually no - for fixed k, larger b is better, and b_max is the largest
            # So if k/(2*b_max) > best_err, no b for this k can help
            # But we should still try smaller k values first... 
            # Actually k is increasing, so this k is worse than previous k's best case
            # We can break if k is getting large enough
            # But let's not break yet, just skip this k
            # Actually, since k is increasing and b_max is roughly constant,
            # k/(2*b_max) is increasing, so once this condition holds,
            # all larger k will also fail. We can break.
            break
        
        # Search for b from b_max downward
        # To make this tractable, we'll check a limited number of b values
        # or use a smarter strategy
        
        # Check b values: start from b_max, step down
        # We'll check up to some number of candidates
        # But factorization is expensive, so we need to be selective
        
        # Strategy: check b values where (3b+k) is even
        # and try to find sigma((3b+k)/2) == sigma(b)
        
        # For efficiency, let's check a batch of b values
        # We'll check from b_max downward, but limit the number of checks
        
        max_checks = 5000  # per k value
        checked = 0
        
        # Determine starting parity
        # 3b + k must be even
        # If k is even, b must be even
        # If k is odd, b must be odd
        if k % 2 == 0:
            b_start = b_max if b_max % 2 == 0 else b_max - 1
        else:
            b_start = b_max if b_max % 2 == 1 else b_max - 1
        
        if b_start < 1:
            continue
            
        b = b_start
        while b >= 1 and checked < max_checks:
            if check_deadline():
                break
                
            a = (3 * b + k) // 2
            if a > N or a < 1:
                b -= 2
                checked += 1
                continue
                
            sa = sigma(a)
            sb = sigma(b)
            
            if sa == sb:
                err_num = k
                err_den = 2 * b
                # Compare err_num/err_den < best_err_num/best_err_den
                # i.e., err_num * best_err_den < best_err_num * err_den
                if err_num * best_err_den < best_err_num * err_den:
                    best_err_num = err_num
                    best_err_den = err_den
                    best_a = a
                    best_b = b
                    
            b -= 2
            checked += 1
    
    # Also do a focused search around the known good pair and nearby
    # Check a window of b values around 80 (the known good b)
    if not check_deadline():
        for b in range(1, min(5000, N) + 1):
            if check_deadline():
                break
            for k in range(1, 20):
                if (3 * b + k) % 2 != 0:
                    continue
                a = (3 * b + k) // 2
                if a > N or a < 1:
                    continue
                sa = sigma(a)
                sb = sigma(b)
                if sa == sb:
                    err_num = k
                    err_den = 2 * b
                    if err_num * best_err_den < best_err_num * err_den:
                        best_err_num = err_num
                        best_err_den = err_den
                        best_a = a
                        best_b = b
    
    return [int(best_a), int(best_b)]
