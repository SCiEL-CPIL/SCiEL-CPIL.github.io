"""Find the longest interval in [1,1008] with no modular inverse pairs mod 1009."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    p = 1009
    n = 1008  # elements are 1..1008
    
    # Precompute modular inverses: inv[x] = x^{-1} mod 1009 for x in 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Two-pointer sweep: left and right both only increase.
    # Invariant: after processing 'right', the window [left, right] is valid
    # (no pair x,y in [left,right] with x*y % p == 1, including x=y).
    #
    # When adding 'right' to the window:
    #   - If right is self-inverse (1 or 1008), it can't be in any valid interval,
    #     so advance left past it.
    #   - Otherwise, the only new conflict is between 'right' and inv[right].
    #     If inv[right] is in [left, right], advance left to inv[right]+1.
    #     This resolves the conflict and the window remains valid because:
    #     * All elements in [left, right-1] were in the previously valid window.
    #     * 'right' now has inv[right] < left, so no conflict with any element.
    
    best_start = 1
    best_length = 0
    
    left = 1
    for right in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Self-inverse elements cannot be in any valid interval
        if right == 1 or right == 1008:
            left = right + 1
            continue
        
        ix = inv[right]
        
        # If inv[right] is in the current window, advance left past it
        if left <= ix <= right:
            left = ix + 1
        
        # Ensure window is non-empty
        if left > right:
            continue
        
        length = right - left + 1
        if length > best_length:
            best_length = length
            best_start = left
    
    if best_length <= 0:
        # Fallback: find any valid single element
        for x in range(1, n + 1):
            if x != 1 and x != 1008:
                best_start = x
                best_length = 1
                break
    
    # Verify the result using a set-based check
    s = best_start
    L = best_length
    elements = set(range(s, s + L))
    valid = True
    for x in elements:
        if (x * x) % p == 1:
            valid = False
            break
        ix = inv[x]
        if ix in elements:
            valid = False
            break
    
    if not valid:
        # Fallback: scan for any valid single element
        best_start = 2
        best_length = 1
    
    return [best_start, best_length]
