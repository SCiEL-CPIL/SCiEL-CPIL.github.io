"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses: inv[i] = i^{-1} mod 1009
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Two-pointer sweep to find longest valid interval
    # An interval [L, R] is valid iff for all x in [L, R], inv[x] is not in [L, R]
    # (This automatically excludes self-inverse elements 1 and 1008 since inv[1]=1, inv[1008]=1008)
    
    L = 1
    best_len = 0
    best_start = 1
    
    for R in range(1, N + 1):
        # If R is self-inverse, it cannot be in any valid interval
        if inv[R] == R:
            L = R + 1
            continue
        
        # Ensure L is at least 2 (since 1 is self-inverse and inv[1]=1, 
        # but we handle it via the conflict check below)
        # Actually, the conflict check handles it: if L <= 1 <= R, then inv[1]=1 is in [L,R]
        # So we need L > 1. But let's let the conflict check handle it naturally.
        
        # Handle conflict: if inv[R] is in [L, R-1], move L past it
        # Also need to check: for all x in [L, R], inv[x] not in [L, R]
        # The key insight: as R increases by 1, only the new element R can create a new conflict.
        # If inv[R] >= L, then we have a conflict, and we must set L = inv[R] + 1.
        # But wait - we also need to ensure that for all existing x in [L, R-1], inv[x] is still not in [L, R].
        # Since we maintained validity for [L, R-1], the only new conflict is from R itself.
        # If inv[R] is in [L, R-1], we move L to inv[R]+1.
        # After moving L, we need to verify the new interval [L, R] is valid.
        # Moving L right only removes elements, so previously valid pairs remain valid.
        # The only thing to check is that the new L doesn't create issues, but since we're
        # removing elements from the left, no new conflicts are introduced.
        
        if L <= inv[R] <= R - 1:
            L = inv[R] + 1
        
        # Also handle the case where R itself is self-inverse (already handled above)
        # And ensure L is valid (L <= R)
        if L <= R:
            current_len = R - L + 1
            if current_len > best_len:
                best_len = current_len
                best_start = L
        
        # Safety check for time
        if time.time() > deadline:
            break
    
    # Brute-force verification: check all intervals to confirm optimality
    # This is O(N^3) worst case but N=1008 so ~1 billion ops - too slow
    # Instead, do a smarter verification: for each start s, find max valid end e
    # This is O(N^2) which is ~1M ops - fast enough
    
    verified_best_len = 0
    verified_best_start = 1
    
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # s must not be self-inverse
        if inv[s] == s:
            continue
        
        # Greedily extend from s
        e = s
        while e < N:
            next_e = e + 1
            # Check if next_e can be added
            if inv[next_e] == next_e:
                break
            # Check if inv[next_e] is in [s, e]
            if s <= inv[next_e] <= e:
                break
            # Also check: is there any x in [s, e] with inv[x] == next_e?
            # That would mean x == inv[next_e], which is the same check above
            e = next_e
        
        # Verify [s, e] is fully valid
        valid = True
        for x in range(s, e + 1):
            if s <= inv[x] <= e:
                valid = False
                break
        
        if valid:
            length = e - s + 1
            if length > verified_best_len:
                verified_best_len = length
                verified_best_start = s
    
    # Use the verified result if it's better or equal
    if verified_best_len >= best_len:
        best_start = verified_best_start
        best_len = verified_best_len
    
    # Fallback
    if best_len == 0:
        best_start = 2
        best_len = 1
    
    return [best_start, best_len]

# EVOLVE-BLOCK-END
