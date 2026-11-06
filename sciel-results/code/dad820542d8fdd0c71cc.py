"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses for all elements in [1, 1008]
    # inv[i] = i^(-1) mod 1009
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Two-pointer approach to find the longest valid interval
    # L is the start of the current valid window
    # R is the end of the current window
    L = 1
    best_len = 0
    best_start = 1
    
    for R in range(1, N + 1):
        # 1. If R is a self-inverse element (1 or 1008), it cannot be in any valid interval.
        #    So we must start the next potential interval after R.
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        # 2. Ensure 1 is not in [L, R]
        if L <= 1 <= R:
            L = 2
            
        # 3. Ensure 1008 is not in [L, R]
        if L <= 1008 <= R:
            L = 1009

        # 4. Handle conflict with R's inverse
        #    If inv[R] is in [L, R-1], then both R and inv[R] are in the interval, which is invalid.
        #    To fix, we must move L to be greater than inv[R].
        if L <= inv[R] <= R - 1:
            L = inv[R] + 1
            
        # 5. Check if the current interval [L, R] is valid and non-empty
        if L <= R:
            current_len = R - L + 1
            if current_len > best_len:
                best_len = current_len
                best_start = L
                
        # Safety check for time
        if time.time() > deadline:
            break

    # Fallback if no valid interval found
    if best_len == 0:
        best_start = 2
        best_len = 1

    # Verify the result and try to find a better one with a refined check
    # The two-pointer should be correct, but let's verify and also check
    # if we can extend the best interval found
    if time.time() < deadline - 1.0:
        # Verify best interval is valid
        def is_valid_interval(s, ln):
            e = s + ln - 1
            if s < 1 or e > N:
                return False
            for x in range(s, e + 1):
                if (x * x) % MOD == 1:
                    return False
            for x in range(s, e + 1):
                ix = inv[x]
                if s <= ix <= e and ix != x:
                    return False
            return True
        
        if not is_valid_interval(best_start, best_len):
            # Fallback to a known valid interval
            best_start = 2
            best_len = 1
        
        # Try to extend the best interval in both directions
        # Try extending to the left
        s, ln = best_start, best_len
        while s > 1:
            ns = s - 1
            if (ns * ns) % MOD == 1:
                break
            if ns == 1 or ns == 1008:
                break
            ix = inv[ns]
            if ns + 1 <= ix <= ns + ln:  # ix in [ns+1, ns+ln]
                break
            s = ns
            ln += 1
        
        # Try extending to the right
        e = s + ln - 1
        while e < N:
            ne = e + 1
            if (ne * ne) % MOD == 1:
                break
            if ne == 1 or ne == 1008:
                break
            ix = inv[ne]
            if s <= ix <= ne - 1:  # ix in [s, ne-1]
                break
            e = ne
            ln += 1
        
        if ln > best_len:
            best_len = ln
            best_start = s

    return [best_start, best_len]

# EVOLVE-BLOCK-END
