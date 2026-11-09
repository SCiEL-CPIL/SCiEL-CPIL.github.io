import math
import time

MAX_N = 10_000_000
TIME_LIMIT = 80.0  # Safety margin for 90s limit

def construct():
    t_start = time.time()
    
    # 1. Compute sigma(n) for all n <= MAX_N
    sigma = [0] * (MAX_N + 1)
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d
            
    # 2. Search for b from MAX_N down to 1
    # We want |2a - 3b| = 1
    # This implies 2a = 3b +/- 1
    # Since 2a is even, 3b must be odd, so b must be odd.
    # a = (3b +/- 1) / 2
    
    best_pair = None
    best_err = float('inf')
    
    # Iterate b downwards to maximize b (minimize error 1/(2b))
    # Step by -2 to only check odd b
    # Start from largest odd <= MAX_N
    start_b = MAX_N if MAX_N % 2 == 1 else MAX_N - 1
    
    for b in range(start_b, 0, -2):
        if time.time() > t_start + TIME_LIMIT - 5.0:
            break
            
        # Candidate a values
        # Case 1: 2a = 3b - 1 => a = (3b - 1) / 2
        a1 = (3 * b - 1) // 2
        if 1 <= a1 <= MAX_N:
            if sigma[a1] == sigma[b]:
                # Error is |a1/b - 1.5| = |(3b-1)/(2b) - 3/2| = |(3b-1-3b)/(2b)| = |-1/(2b)| = 1/(2b)
                # This is > 0.
                return [int(a1), int(b)]
                
        # Case 2: 2a = 3b + 1 => a = (3b + 1) / 2
        a2 = (3 * b + 1) // 2
        if 1 <= a2 <= MAX_N:
            if sigma[a2] == sigma[b]:
                # Error is 1/(2b)
                return [int(a2), int(b)]

    # If no solution with |2a-3b|=1 found, try |2a-3b|=2
    # This implies a = 3b/2 +/- 1
    # If b is even, a is integer.
    # Error is 1/b.
    # This is a weaker solution (higher error), but better than nothing.
    
    # However, the loop above already took most of the time.
    # If we didn't find anything, it's unlikely we will find a |2a-3b|=2 solution quickly 
    # unless we restart the search.
    # Given the time spent in sieve and first loop, we might be close to limit.
    
    # Let's check if we have time for a second pass.
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 10.0:
         # Search for |2a-3b|=2
         # a = (3b +/- 2) / 2
         # b must be even for a to be integer? 
         # If b is even, 3b is even, 3b+2 is even.
         # If b is odd, 3b is odd, 3b+2 is odd.
         # So b must be even.
         
         start_b2 = MAX_N if MAX_N % 2 == 0 else MAX_N - 1
         for b in range(start_b2, 0, -2):
             if time.time() > t_start + TIME_LIMIT - 2.0:
                 break
             
             a1 = (3 * b - 2) // 2
             if 1 <= a1 <= MAX_N:
                 if sigma[a1] == sigma[b]:
                     return [int(a1), int(b)]
             
             a2 = (3 * b + 2) // 2
             if 1 <= a2 <= MAX_N:
                 if sigma[a2] == sigma[b]:
                     return [int(a2), int(b)]

    # Fallback: return a valid pair with sigma(a)=sigma(b)
    # e.g., a=1, b=1 is invalid (err 0.5? No, 1/1 - 1.5 = 0.5. Wait.
    # If a=1, b=1, sigma(1)=1, sigma(1)=1. err = |1 - 1.5| = 0.5.
    # This is valid and non-zero.
    # But we want to minimize error.
    # If we found nothing better, we return [1,1]?
    # Or we can try to find any pair.
    
    # A simple valid pair: 
    # sigma(2) = 1+2 = 3. sigma(3) = 1+3 = 4. No.
    # sigma(4) = 1+2+4 = 7. sigma(6) = 1+2+3+6 = 12. No.
    # sigma(1) = 1. sigma(2) = 3.
    # Is there any n with sigma(n)=1? Only n=1.
    # So (1,1) is the only pair with sigma=1.
    # (1,1) gives err 0.5.
    
    # Let's just return [1, 1] as a safe fallback if the search fails.
    # It's valid: int, <= 10^7, sigma(1)=sigma(1), err=0.5 != 0.
    return [1, 1]
