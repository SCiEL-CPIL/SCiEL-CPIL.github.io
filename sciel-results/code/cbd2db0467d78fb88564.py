import time

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # values are 1..1008
    
    # Compute modular inverse for each x in 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Forbidden single values: 1 and 1008 (self-inverse)
    # So interval must be within [2, 1007]
    lo_bound = 2
    hi_bound = 1007
    
    best_start = lo_bound
    best_len = 1
    
    # Two-pointer / sliding window approach
    # For each left endpoint L, find the maximum R such that [L, R] is valid.
    # A window [L, R] is valid iff for every x in [L, R], inv[x] is NOT in [L, R].
    # This means: for all x in [L, R], either inv[x] < L or inv[x] > R.
    
    # Key insight: if we fix L and increase R, the window is valid up to the
    # first R where some x in [L, R] has inv[x] in [L, R].
    # 
    # More efficiently: for a fixed L, the maximum valid R is the minimum over
    # all x >= L of (inv[x] - 1) if inv[x] >= L, otherwise infinity.
    # But we need inv[x] > R for validity, so if inv[x] >= L, we need R < inv[x].
    # So max R = min over x in [L, ...] where inv[x] >= L of (inv[x] - 1).
    # But we also need R >= L, and R <= hi_bound.
    
    # Let's use a different approach: for each L, scan R from L upward.
    # The window [L, R] is valid iff for all x in [L, R], inv[x] < L or inv[x] > R.
    # As R increases, the condition "inv[x] > R" becomes harder to satisfy.
    # The first violation occurs when there exists x in [L, R] with inv[x] in [L, R].
    
    # For each L, let's find the smallest R >= L such that there exists x in [L, R]
    # with inv[x] in [L, R]. Then the max valid R for this L is R-1.
    
    # Alternatively, we can precompute for each position x, the "blocking" effect.
    # For a given L, x blocks the window if inv[x] >= L. In that case, R must be < inv[x].
    # So for fixed L, the maximum valid R is:
    #   min(hi_bound, min{inv[x] - 1 : x >= L, inv[x] >= L} )
    # But x must also be <= R, so x is in [L, R]. This is circular.
    
    # Let's just do the direct scan but with early termination.
    # For each L, scan R from L to hi_bound.
    # Keep track of whether any x in [L, R] has inv[x] in [L, R].
    
    # Actually, let's use the insight from inspiration 2:
    # When adding R to window [L, R-1]:
    # - If inv[R] < L: safe (inverse is outside window to the left)
    # - If inv[R] > R: safe for now, but note that if we later extend to inv[R], 
    #   then R and inv[R] would both be in the window. So we need to track the 
    #   minimum "future inverse" among elements in the window.
    # - If inv[R] in [L, R-1]: conflict!
    
    # Let's implement this carefully.
    
    for L in range(lo_bound, hi_bound + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Track the minimum inv[x] for x in [L, R] where inv[x] > x
        # This helps detect when a future R' = inv[x] would cause a conflict.
        min_future_inv = float('inf')
        max_R = L - 1
        
        for R in range(L, hi_bound + 1):
            ir = inv[R]
            
            if ir == R:
                # Self-inverse, cannot include R
                break
            elif ir < L:
                # Inverse is to the left of L, safe
                max_R = R
            elif ir < R:
                # Inverse is in [L, R-1] or [1, L-1]
                if ir >= L:
                    # Conflict: ir is in [L, R-1]
                    break
                else:
                    # ir < L, safe
                    max_R = R
            else:  # ir > R
                # Inverse is to the right, safe for now
                # But if we extend to ir, then R and ir would both be in window
                if ir < min_future_inv:
                    min_future_inv = ir
                # Check if current R already conflicts with a previous element's future inverse
                if R >= min_future_inv:
                    break
                max_R = R
        
        current_len = max_R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
    
    # Verify the result
    final_start = best_start
    final_end = best_start + best_len - 1
    valid = True
    for x in range(final_start, final_end + 1):
        if final_start <= inv[x] <= final_end:
            valid = False
            break
    
    if not valid:
        # Fallback: find a valid interval
        # Try small intervals
        for s in range(lo_bound, hi_bound + 1):
            for l in range(1, hi_bound - s + 2):
                e = s + l - 1
                if e > hi_bound:
                    break
                ok = True
                for x in range(s, e + 1):
                    if s <= inv[x] <= e:
                        ok = False
                        break
                if ok:
                    if l > best_len:
                        best_len = l
                        best_start = s
                    break
    
    return [best_start, best_len]
