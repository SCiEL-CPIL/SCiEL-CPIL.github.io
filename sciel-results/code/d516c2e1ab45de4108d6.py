"""Optimized constructor for distinct three-term unit-fraction reps of 4/n."""
import time
import math

DENOM_BOUND = 10**12

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s kill
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        x_min = n // 4 + 1
        found = None
        
        # Try x values in a strategic order: first small x (near n/4), then larger
        # For each x, the y-range is approximately n/4 in width
        # Total work per n is roughly O(num_x * n)
        
        # We'll try up to 500 x values, but break early if found
        max_x_tries = 500
        x_limit = x_min + max_x_tries
        
        for x_attempt in range(x_min, x_limit):
            if time.time() > deadline:
                break
            A = 4 * x_attempt - n
            if A <= 0:
                continue
            B = n * x_attempt
            
            # y must satisfy: B/A < y <= 2B/A and y > x_attempt (for x < y < z ordering)
            y_lo = B // A + 1
            y_hi = (2 * B) // A
            if y_hi < y_lo:
                continue
            if y_lo <= x_attempt:
                y_lo = x_attempt + 1
            if y_lo > y_hi:
                continue
            
            range_len = y_hi - y_lo + 1
            
            # For small ranges, direct iteration is fast
            # For large ranges, we can use the divisor trick or just iterate with time checks
            # Range length is at most ~n^2/16 for x near n/4, which for n=256 is ~4096
            # So direct iteration with a cap of 20000 should cover all cases
            
            A_local = A
            B_local = B
            x_local = x_attempt
            bound = DENOM_BOUND
            
            # Cap y iterations to avoid getting stuck on one x value
            y_cap = min(y_hi + 1, y_lo + 20000)
            
            for y in range(y_lo, y_cap):
                if (y - y_lo) % 4000 == 0 and time.time() > deadline:
                    break
                denom = A_local * y - B_local
                if denom <= 0:
                    continue
                if (B_local * y) % denom == 0:
                    z = (B_local * y) // denom
                    if z > y and z != x_local and z <= bound:
                        if x_local != y and y != z and x_local != z:
                            found = [x_local, y, z]
                            break
            
            if found:
                break
        
        answers[i] = found
    
    return answers
