"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # values are 1..1008
    
    # Compute modular inverse for each x in 1..1008
    # x * inv[x] ≡ 1 (mod 1009)
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse values: x^2 ≡ 1 mod 1009 => x = 1 or x = 1008
    # These cannot be in any valid interval (since x*x % 1009 = 1)
    # So we search in [2, 1007]
    
    lo_bound = 2
    hi_bound = 1007
    n = hi_bound - lo_bound + 1  # 1006
    
    # For a window [L, R] to be valid: for all x in [L, R], inv[x] < L or inv[x] > R
    # 
    # Key insight: For a fixed L, the maximum valid R can be computed as:
    #   max_R(L) = min over x in [L, ...] of (inv[x] - 1) if inv[x] >= L
    # More precisely, for each x >= L, if inv[x] >= L, then R must be < inv[x].
    # So max_R(L) = min(inv[x] - 1 for x in [L, hi_bound] if inv[x] >= L)
    # But we also need R >= L, and the window must be non-empty.
    #
    # Actually, let me think more carefully. For a window [L, R]:
    #   For every x in [L, R], we need inv[x] < L or inv[x] > R.
    #   Equivalently: for every x in [L, R] with inv[x] >= L, we need inv[x] > R.
    #   So R < inv[x] for all x in [L, R] with inv[x] >= L.
    #   This means R <= min(inv[x] - 1 for x in [L, R] if inv[x] >= L)
    #
    # This is circular because R appears on both sides. But we can compute it iteratively:
    # Start with R = L. Then check all x in [L, R]: if any has inv[x] in [L, R], shrink R.
    #
    # Better approach: For each L, compute the "bottleneck" - the minimum inv[x] among all x in [L, hi_bound] where inv[x] >= L.
    # Let bottleneck(L) = min(inv[x] for x in [L, hi_bound] if inv[x] >= L), or infinity if no such x.
    # Then max_R(L) = min(bottleneck(L) - 1, hi_bound).
    # But wait, this isn't quite right either, because x ranges over [L, R], not [L, hi_bound].
    # However, if we take x over [L, hi_bound], the constraint is stronger, so it gives a valid (possibly suboptimal) R.
    # Actually no - if x > R, then x is not in the window, so its inv doesn't matter.
    #
    # Let me use a different approach: precompute for each L the maximum valid R using a sweep.
    # 
    # Alternative O(n log n) approach:
    # For each pair (x, y) where y = inv[x] and x < y (to avoid double counting),
    # this pair forbids any window that contains both x and y.
    # A window [L, R] contains both x and y iff L <= x and y <= R (assuming x < y).
    # So for each such pair, it forbids windows where L <= x AND R >= y.
    #
    # For a fixed L, the forbidden R values are those R >= y for any pair (x, y) with x >= L and x < y.
    # So max_R(L) = min(y - 1 for pairs (x,y) with x >= L and x < y) - but we need the min y, so max_R = min_y - 1.
    # Wait, that's not right. We need R < y for ALL such pairs. So R < min(y for all applicable pairs).
    # So max_R(L) = min(y for pairs (x,y) with x >= L, x < y) - 1.
    # But also R >= L, and R <= hi_bound.
    #
    # Let me build this properly:
    # Create list of pairs (x, y) where y = inv[x], x < y, both in [2, 1007].
    # For each L, find min y among pairs where x >= L.
    # This is a suffix minimum query!
    
    # Build the list of conflict pairs
    # For x in [2, 1007], y = inv[x]. If x < y, add pair (x, y).
    # Note: if x > y, the pair is already counted as (y, x) when we process y.
    # If x == y, it's self-inverse (only 1 and 1008, already excluded).
    
    # For each x in [2, 1007], store the minimum y (where y = inv[x] and y > x) or infinity
    min_y_for_x = [float('inf')] * (MOD + 1)
    for x in range(lo_bound, hi_bound + 1):
        y = inv[x]
        if y > x and lo_bound <= y <= hi_bound:
            min_y_for_x[x] = y
    
    # Now for each L, we need: min_y(L) = min(min_y_for_x[x] for x in [L, hi_bound])
    # This is a suffix minimum.
    
    suffix_min_y = [float('inf')] * (MOD + 2)
    for x in range(hi_bound, lo_bound - 1, -1):
        suffix_min_y[x] = min(min_y_for_x[x], suffix_min_y[x + 1])
    
    # For each L in [lo_bound, hi_bound]:
    #   max_R = min(suffix_min_y[L] - 1, hi_bound) if suffix_min_y[L] < inf else hi_bound
    #   But we also need max_R >= L (non-empty window)
    #   length = max_R - L + 1
    
    best_start = lo_bound
    best_len = 0
    
    for L in range(lo_bound, hi_bound + 1):
        if suffix_min_y[L] < float('inf'):
            max_R = min(suffix_min_y[L] - 1, hi_bound)
        else:
            max_R = hi_bound
        
        if max_R < L:
            # No valid window starting at L? That shouldn't happen since single element is always valid.
            # Wait - single element [L, L] is valid iff inv[L] != L, which is true for L in [2, 1007].
            # So max_R should be at least L. Let me check: if suffix_min_y[L] = L+1, then max_R = L. OK.
            # If suffix_min_y[L] = L, that would mean there's a pair (x, L) with x >= L and x < L, impossible.
            # Actually suffix_min_y[L] is min y where y > x >= L, so y > L always. So max_R >= L.
            continue
        
        length = max_R - L + 1
        if length > best_len:
            best_len = length
            best_start = L
        
        # Deadline check (should be fast, but just in case)
        if time.time() - start_time > DEADLINE:
            break
    
    # Verify the result
    final_start = best_start
    final_end = best_start + best_len - 1
    valid = True
    for x in range(final_start, final_end + 1):
        if final_start <= inv[x] <= final_end:
            valid = False
            break
    
    if not valid:
        # Fallback: find a valid interval by brute force
        for L in range(lo_bound, hi_bound + 1):
            for R in range(hi_bound, L - 1, -1):
                ok = True
                for x in range(L, R + 1):
                    if L <= inv[x] <= R:
                        ok = False
                        break
                if ok:
                    final_start = L
                    best_len = R - L + 1
                    valid = True
                    break
            if valid:
                break
    
    if not valid:
        # Last resort: single element
        final_start = lo_bound
        best_len = 1
    
    return [final_start, best_len]

# EVOLVE-BLOCK-END
