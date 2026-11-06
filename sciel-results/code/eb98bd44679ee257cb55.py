import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    # inv[i] is the modular inverse of i modulo MOD
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # Forbidden elements are those x where x*x % MOD == 1
    # These are 1 and MOD-1 (1008)
    forbidden = {1, MOD - 1}

    best_len = 0
    best_start = 1

    # Forward pass: Two-pointer sliding window
    s = 1
    for e in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # If e is a self-inverse, it cannot be in the interval.
        # Move s past e.
        if e in forbidden:
            s = max(s, e + 1)
            continue
            
        ie = inv[e]
        # If the inverse of e is already in the window [s, e-1],
        # we have a conflict (e * ie % MOD == 1).
        # We must remove ie from the window by moving s to ie + 1.
        # Note: ie < e is guaranteed if ie is in [s, e-1].
        # Also, if ie == e, it would be a self-inverse, handled above.
        if s <= ie < e:
            s = ie + 1
            
        # Now [s, e] should be valid.
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    if best_len == 0:
        # Fallback: Any single element that is not forbidden is valid.
        # e.g., 2. 2*2=4 != 1.
        return [2, 1]
        
    # Final verification
    def is_valid(start, length):
        end = start + length - 1
        if start < 1 or end > N or length <= 0:
            return False
        for x in range(start, end + 1):
            # Check self-inverse
            if (x * x) % MOD == 1:
                return False
            # Check distinct pair
            ix = inv[x]
            if ix != x and start <= ix <= end:
                return False
        return True

    if not is_valid(best_start, best_len):
        # If verification fails (should not happen with correct two-pointer),
        # fallback to exhaustive search for small N if time permits, or return best found so far.
        # Given N=1008, exhaustive is O(N^2 * L) which might be slow if L is large.
        # But the two-pointer is O(N). If it fails, there's a logic bug.
        # For safety, we can try a few smaller intervals around best_start.
        # Or just return the best found, assuming the check is the only issue.
        # However, the task requires a valid result.
        # Let's do a quick check for any valid interval of length 1.
        for i in range(2, N + 1):
            if i not in forbidden:
                return [i, 1]
                
    return [best_start, best_len]
