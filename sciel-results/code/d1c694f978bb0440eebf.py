"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008  # interval is contained in [1, 1008]
    
    # Precompute modular inverses
    # inv[x] is the y in [1, p-1] such that x*y % p == 1
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x such that x*x % p == 1
    # For p=1009, these are x=1 and x=p-1=1008
    # These cannot appear in any valid interval
    
    # Use two-pointer sliding window to find the longest valid interval
    # A window [L, R] is valid if:
    #   1. No element in [L,R] is a self-inverse (1 or 1008)
    #   2. No pair (x, inv[x]) both exist in [L,R]
    
    best_len = 0
    best_start = 1
    
    L = 1
    for R in range(1, n + 1):
        # If R is a self-inverse, it cannot be in any valid interval
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is in the current window [L, R-1], we have a conflict
        # Move L past invR to resolve the conflict
        if L <= invR < R:
            L = invR + 1
        
        # Also need to ensure L doesn't point to a self-inverse
        # (already handled by the above logic since we skip self-inverses)
        
        # Check if the window [L, R] is non-empty
        if R >= L:
            # Verify that no self-inverse is in [L, R]
            # 1 and 1008 are the only self-inverses
            # If L <= 1 <= R, then 1 is in the window - but we skip R=1 above
            # Actually, let's be more careful. If 1 is in [L,R], it's invalid.
            # If 1008 is in [L,R], it's invalid.
            # Since R never equals 1 or 1008 (we skip them), 
            # but L could be 1 and R > 1, so 1 could be in [L,R].
            # Similarly, L could be <= 1008 and R >= 1008, but R < 1009 so R <= 1008.
            # Wait, R ranges from 1 to 1008. If R=1008, we skip it.
            # So R is never 1 or 1008.
            # But L could be 1, and then 1 is in [L, R] for any R >= 1.
            # We need to handle this.
            
            # If 1 is in [L, R], the window is invalid
            if L <= 1 <= R:
                # Move L past 1
                L = 2
                # Re-check invR condition after moving L
                if L <= invR < R:
                    L = invR + 1
                # Re-check if 1 is still in [L,R] - it shouldn't be since L >= 2 now
            
            # If 1008 is in [L, R], the window is invalid
            if L <= 1008 <= R:
                # Move L past 1008
                L = 1009  # This makes the window empty
                # Actually, since R <= 1008, if 1008 is in [L,R], then R >= 1008
                # But R=1008 is skipped, so R < 1008 when we get here? No.
                # R ranges 1..1008, we skip R=1 and R=1008.
                # So R is in 2..1007. Thus 1008 > R always, so 1008 cannot be in [L,R].
                # So this check is unnecessary.
                pass
            
            # Now check window validity
            if R >= L:
                curr_len = R - L + 1
                if curr_len > best_len:
                    best_len = curr_len
                    best_start = L
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
