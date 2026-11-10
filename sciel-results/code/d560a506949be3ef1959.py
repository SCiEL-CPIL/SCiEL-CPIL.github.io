import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    MAX = 10_000_000
    
    # 1. Sieve Smallest Prime Factor (SPF)
    spf = list(range(MAX + 1))
    for i in range(2, int(MAX**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute sigma(n) using SPF
    sigma = [0] * (MAX + 1)
    sigma[1] = 1
    p_power = [1] * (MAX + 1)
    sigma_p_power = [1] * (MAX + 1)
    
    for i in range(2, MAX + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            # i = p^k * m', m = p^(k-1) * m'
            p_power[i] = p_power[m] * p
            sigma_p_power[i] = sigma_p_power[m] + p_power[i]
            # sigma(i) = sigma(m') * sigma(p^k)
            # m' = m // p_power[m]
            sigma[i] = sigma[m // p_power[m]] * sigma_p_power[i]
        else:
            # i = p * m, gcd(p, m) = 1
            p_power[i] = p
            sigma_p_power[i] = 1 + p
            sigma[i] = sigma[m] * (1 + p)
        
        if time.time() > deadline:
            # Fallback if sieve takes too long (unlikely)
            return [11, 6]

    # 3. Search for best pair
    best_error_num = None
    best_error_den = None
    best_a = None
    best_b = None
    
    # We want to minimize |2a - 3b| / (2b)
    # Iterate b from MAX down to 1
    # For each b, check a near 1.5b
    
    for b in range(MAX, 0, -1):
        if time.time() > deadline:
            break
            
        sb = sigma[b]
        
        # Ideal a = 1.5 * b
        # Check a such that |2a - 3b| is small
        # Min non-zero |2a - 3b| is 1 if b is odd, 2 if b is even.
        
        # Candidates for a:
        # a = floor(1.5b) and a = ceil(1.5b)
        # If 1.5b is integer (b even), a = 1.5b gives error 0 (invalid).
        # So check a = 1.5b - 1, 1.5b + 1
        
        # Let's just check a few integers around 1.5b
        # a_min = max(1, int(1.5 * b) - 10)
        # a_max = min(MAX, int(1.5 * b) + 10)
        
        # More efficient:
        # Check specific candidates that minimize |2a - 3b|
        
        if b % 2 == 0:
            # 3b is even, 1.5b is integer
            a_target = 3 * b // 2
            # Candidates: a_target - 1, a_target + 1
            # Error for a_target-1: |2(a_target-1) - 3b| = |2a_target - 2 - 3b| = |-2| = 2
            # Error for a_target+1: |2(a_target+1) - 3b| = |2| = 2
            # Both give error 2/(2b) = 1/b
            candidates = [a_target - 1, a_target + 1]
        else:
            # 3b is odd, 1.5b is x.5
            # a1 = (3b - 1) // 2, a2 = (3b + 1) // 2
            # Error for a1: |2a1 - 3b| = |3b - 1 - 3b| = 1
            # Error for a2: |2a2 - 3b| = |3b + 1 - 3b| = 1
            # Both give error 1/(2b)
            candidates = [(3 * b - 1) // 2, (3 * b + 1) // 2]
            
        for a in candidates:
            if a < 1 or a > MAX:
                continue
            if sigma[a] == sb:
                # Calculate error
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue # Invalid
                
                err_num = diff
                err_den = 2 * b
                
                # Compare with best
                if best_error_num is None:
                    best_error_num = err_num
                    best_error_den = err_den
                    best_a = a
                    best_b = b
                else:
                    # Compare err_num/err_den < best_num/best_den
                    # err_num * best_den < best_num * err_den
                    if err_num * best_error_den < best_error_num * err_den:
                        best_error_num = err_num
                        best_error_den = err_den
                        best_a = a
                        best_b = b
        
        # Early termination?
        # The minimum possible error for any b' < b is >= 1 / (2 * b')
        # If we have found an error E, and E <= 1 / (2 * b), then no smaller b can beat it?
        # Wait. If b is even, min error is 1/b.
        # If we found error 1/b (b even), can a smaller odd b' give 1/(2b') < 1/b?
        # 1/(2b') < 1/b => b < 2b' => b' > b/2.
        # So if we are at b, and we found error 1/b, we can't stop immediately because
        # a b' = b/2 + 1 (odd) could give 1/(2(b/2+1)) which is approx 1/b.
        # Actually 1/(2(b/2)) = 1/b.
        # So we should not stop early unless we are sure.
        # Given MAX=10^7, the loop is fast enough to run to completion or until deadline.
        
    if best_a is None:
        # Fallback
        return [11, 6]
        
    return [int(best_a), int(best_b)]
