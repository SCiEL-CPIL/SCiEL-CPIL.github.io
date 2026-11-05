"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses in [1, 1008]
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Two-pointer sliding window approach (O(N))
    # Maintain window [left, right] that is valid.
    # For each right, advance left as needed to maintain validity.
    # A window is valid iff no element is the inverse of another element in the window.
    
    best_len = 0
    best_start = 1
    left = 1
    
    for right in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # If right is self-inverse (right * right % p == 1), we can't include it
        if (right * right) % p == 1:
            left = right + 1
            continue
        
        # If inv[right] is in [left, right-1], we must exclude it
        ie = inv[right]
        if left <= ie < right:
            left = ie + 1
        
        # Now [left, right] should be valid
        if left <= right:
            length = right - left + 1
            if length > best_len:
                best_len = length
                best_start = left
    
    if best_len == 0:
        # Fallback: return a small valid interval
        return [2, 1]
    
    # Verify the result using brute force
    s = best_start
    e = s + best_len - 1
    valid = True
    for x in range(s, e + 1):
        if (x * x) % p == 1:
            valid = False
            break
        for y in range(s, e + 1):
            if (x * y) % p == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Brute force search for the true maximum
        best_len_bf = 0
        best_start_bf = 1
        for start in range(1, n + 1):
            if time.time() > deadline:
                break
            for length in range(1, n - start + 2):
                end = start + length - 1
                if end > n:
                    break
                # Check if interval [start, end] is valid
                ok = True
                for x in range(start, end + 1):
                    if (x * x) % p == 1:
                        ok = False
                        break
                    if not ok:
                        break
                    for y in range(start, end + 1):
                        if (x * y) % p == 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    if length > best_len_bf:
                        best_len_bf = length
                        best_start_bf = start
                else:
                    # No need to check longer intervals starting at same start
                    break
        if best_len_bf > 0:
            return [best_start_bf, best_len_bf]
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
