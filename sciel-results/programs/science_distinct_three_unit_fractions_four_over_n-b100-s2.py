"""Optimized constructor for distinct three-term unit-fraction reps of 4/n."""
import time

DENOM_BOUND = 10**12

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin under 90s kill
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        x_min = n // 4 + 1
        found = None
        # Try a larger range of x values to find solutions with larger minimal denominators
        # The range of y is approx n/4, so total work per x is O(n). 
        # With 200 x values, work per n is O(200 * n).
        # Sum over n=3..256 is roughly 200 * (256^2/2) ~ 6.5 million iterations.
        # Each iteration is simple integer arithmetic. Should fit in 80s easily.
        for x_attempt in range(x_min, x_min + 200):
            if time.time() > deadline:
                break
            A = 4 * x_attempt - n
            if A <= 0:
                continue
            B = n * x_attempt
            # y must satisfy: B/A < y <= 2B/A and y > x_attempt
            y_lo = B // A + 1
            y_hi = (2 * B) // A
            if y_hi < y_lo:
                continue
            # Ensure y > x_attempt for ordering and distinctness
            if y_lo <= x_attempt:
                y_lo = x_attempt + 1
            if y_lo > y_hi:
                continue
            
            # Iterate y
            # Limit iterations per x to avoid pathological cases, though range is naturally bounded
            # Range length is y_hi - y_lo + 1 <= B/A = n*x/(4x-n)
            # For x close to n/4, this is large. For x >> n/4, it's ~ n/4.
            # We cap it to be safe, but 5000 is generous for n<=256 (max range ~ 256*256/4 ~ 16k? No, max x is x_min+200)
            # If x = n/4 + 1, A=4, B=n(n/4+1) ~ n^2/4. Range ~ n^2/16. For n=256, range ~ 4096.
            # So 5000 is enough.
            
            # Local variables for speed
            A_local = A
            B_local = B
            x_local = x_attempt
            bound = DENOM_BOUND
            
            # Check time every 1000 iterations to reduce overhead
            y_limit = min(y_hi + 1, y_lo + 5000)
            for y in range(y_lo, y_limit):
                if (y - y_lo) % 1000 == 0 and time.time() > deadline:
                    break
                denom = A_local * y - B_local
                if denom <= 0:
                    continue
                if (B_local * y) % denom == 0:
                    z = (B_local * y) // denom
                    # z > y is guaranteed by y <= 2B/A construction if z is integer?
                    # 1/y + 1/z = A/B. If y <= 2B/A, then 1/y >= A/2B => 1/z <= A/2B => z >= 2B/A >= y.
                    # If z == y, then 2/y = A/B => y = 2B/A. This is an integer solution but not distinct.
                    # We require distinct denominators.
                    if z > y and z != x_local and z <= bound:
                        # Check distinctness of all three
                        if x_local != y and y != z and x_local != z:
                            found = [x_local, y, z]
                            break
            if found:
                break
        answers[i] = found
    
    return answers
