"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def sigma(n):
    """Compute sigma(n) efficiently."""
    if n <= 0:
        return 0
    res = 1
    i = 2
    while i * i <= n:
        if n % i == 0:
            # i is a prime factor
            count = 0
            while n % i == 0:
                n //= i
                count += 1
            # Sum of powers: 1 + i + ... + i^count = (i^(count+1) - 1) / (i - 1)
            term = (i ** (count + 1) - 1) // (i - 1)
            res *= term
        i += 1
    if n > 1:
        res *= (n + 1)
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    # We want to minimize |a/b - 3/2|.
    # This is equivalent to minimizing |2a - 3b| / (2b).
    # For a fixed small integer k = 2a - 3b, the error is |k| / (2b).
    # To minimize error, we want:
    # 1. Small |k| (preferably 1, 2, 3, ...)
    # 2. Large b (subject to a, b <= 10^7)
    
    # We iterate k from small to large.
    # For a given k, we have 2a - 3b = k.
    # Since gcd(2,3)=1, solutions exist for any integer k.
    # General solution:
    # a = a0 + 3t
    # b = b0 + 2t
    # We need to find a particular solution (a0, b0) for each k.
    
    # Let's find a0, b0 such that 2a0 - 3b0 = k.
    # We can pick b0 such that 3b0 ≡ -k (mod 2).
    # 3b0 ≡ b0 (mod 2). So b0 ≡ -k (mod 2).
    # If k is even, b0 can be even. If k is odd, b0 can be odd.
    # Let's just solve for b0 in a simple way.
    # 2a = 3b + k. So 3b + k must be even.
    # If k is even, b can be anything (parity of 3b doesn't matter if k is even? No. 3b+k even => 3b even => b even).
    # If k is odd, 3b must be odd => b odd.
    
    # Actually, it's easier to just iterate b for a given k?
    # No, for a fixed k, b can be very large.
    # b = (2a - k)/3.
    # We want large b. Max b is 10^7.
    # a = (3b + k)/2.
    # Since a <= 10^7, (3b + k)/2 <= 10^7 => 3b + k <= 2*10^7 => b <= (2*10^7 - k)/3.
    # Also b >= 1.
    # And 3b + k must be even.
    
    # Strategy:
    # Iterate k from 1 upwards.
    # For each k, find the largest valid b such that:
    # 1. b <= 10^7
    # 2. a = (3b + k)/2 is an integer (so 3b+k is even)
    # 3. a <= 10^7
    # 4. a, b are positive integers.
    
    # If we find such a pair, check sigma(a) == sigma(b).
    # If yes, update best if error is smaller.
    # Note: For a fixed k, error is constant |k|/(2b). So larger b means smaller error.
    # So for each k, we only need to check the LARGEST valid b (and maybe a few others if the largest fails sigma check? 
    # No, if the largest b fails, a smaller b with same k will have LARGER error, so it can't beat the current best 
    # unless the current best is worse. But we are iterating k in increasing order. 
    # Wait, error for k=1 with b=10^6 is 1/(2*10^6). Error for k=100 with b=10^7 is 100/(2*10^7) = 5/10^6.
    # So small k is generally better, but large b is better.
    # We should prioritize small k and large b.
    
    # Let's iterate k from 1 to some limit.
    # For each k, find the max b.
    # Check sigma.
    # If we find a match, we have a candidate.
    # We should also consider that for a specific k, maybe the max b doesn't work, but a slightly smaller b does.
    # However, checking all b for a fixed k is expensive if k is small (many solutions).
    # But we can just check the top few largest b for each k.
    
    # Let's refine:
    # For a given k, valid b must satisfy:
    # b <= (2*10^7 - k) / 3
    # b >= 1
    # (3b + k) % 2 == 0
    
    max_k_to_check = 1000  # Small k is most likely to yield the best ratio
    
    for k in range(1, max_k_to_check + 1):
        if time.time() > deadline:
            break
            
        # Find max b
        # 3b + k <= 20000000
        # b <= (20000000 - k) / 3
        max_b_possible = (20000000 - k) // 3
        
        if max_b_possible < 1:
            continue
        
        # Determine parity condition for b
        # (3b + k) % 2 == 0
        # 3b % 2 == b % 2
        # So (b + k) % 2 == 0
        # b % 2 == (-k) % 2 == k % 2
        
        # So b must have the same parity as k.
        
        # Start from max_b_possible and go down, adjusting parity if needed.
        if max_b_possible % 2 != k % 2:
            b_start = max_b_possible - 1
        else:
            b_start = max_b_possible
            
        # Check a few candidates from the top
        # How many? If we check too many, we might timeout.
        # Let's check the top 100 for each k?
        # Or just the top 1?
        # If top 1 fails sigma, the next one has slightly larger error.
        # Given the sparsity of sigma collisions, we might need to check several.
        # But checking sigma is fast.
        
        checked = 0
        for b in range(b_start, max(0, b_start - 1000), -2): # Step by 2 to maintain parity
            if time.time() > deadline:
                break
                
            a = (3 * b + k) // 2
            
            # Verify constraints
            if a < 1 or a > 10000000:
                continue
            if b < 1 or b > 10000000:
                continue
            
            # Verify integer division was exact (should be by construction)
            if 2 * a != 3 * b + k:
                continue
                
            # Check sigma
            sa = sigma(a)
            sb = sigma(b)
            
            if sa == sb:
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_a, best_b = a, b
                    best_err = err
                    # If we found a very good one, we might stop?
                    # No, smaller k might yield even better.
                    # But for this k, we found the best b (largest b).
                    # So we can break the inner loop for this k?
                    # Yes, because for this k, error decreases as b increases.
                    # We started from max b. So this is the best for this k.
                    break
        
        # If we found a very small error, we can break outer loop?
        # Theoretical minimum error for k=1 is 1/(2*10^7) approx 5e-8.
        # If we get close to that, it's hard to beat.
        # But let's keep searching for smaller k.
        
    # Return as Python integers
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
