"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute modular inverses for all elements in [1, p-1]
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Find the maximum interval [start, start+length-1] within [1, p-1]
    # such that for all x, y in the interval, x*y % p != 1
    # This means no element in the interval can have its inverse also in the interval
    
    # An element x is "forbidden" in an interval if inv[x] is also in the interval
    # We need to find the longest contiguous range where no such pair exists
    
    best_len = 0
    best_start = 1
    
    # For each possible start position, find the maximum valid end
    for s in range(1, p):
        # If even reaching the end can't beat current best, stop
        if (p - 1) - s + 1 <= best_len:
            break
        
        # Find the maximum e such that [s, e] is valid
        # An interval [s, e] is invalid if there exists x in [s,e] with inv[x] also in [s,e]
        # i.e., if inv[x] >= s and inv[x] <= e for some x in [s,e]
        # Equivalently, the interval is valid iff for all x in [s,e], inv[x] < s or inv[x] > e
        
        # Let's find the minimum "conflict" - for each x, if inv[x] >= s, then we need e < inv[x]
        # So the maximum valid e is min over all x in [s, ...] where inv[x] >= s of (inv[x] - 1)
        
        max_e = p - 1
        for x in range(s, p):
            ix = inv[x]
            if ix >= s:
                # If inv[x] is in [s, p-1], then e must be < ix to avoid having both x and inv[x]
                # But wait, we also need x itself to be in [s, e]
                # If x > e, then x isn't in the interval, so no conflict
                # So for a given s, the constraint from x is: if x <= e and ix <= e, conflict
                # Rearranging: for the interval [s, e], conflict exists iff there's x in [s,e] with inv[x] in [s,e]
                
                # For fixed s, as we extend e, the first conflict occurs at the smallest
                # e such that some x in [s,e] has inv[x] in [s,e]
                # This happens when e >= min(x, inv[x]) for some x where both x >= s and inv[x] >= s
                # Actually, let me think differently.
                
                # For fixed s, define f(e) = whether [s,e] is valid
                # [s,e] is valid iff for all x in [s,e], inv[x] not in [s,e]
                # i.e., for all x in [s,e], inv[x] < s or inv[x] > e
                
                # The interval becomes invalid at the smallest e >= s such that
                # there exists x in [s,e] with s <= inv[x] <= e
                # This is equivalent to: there exists x in [s,e] with inv[x] in [s,e]
                # Which means: min over x in [s,e] of max(x, inv[x])... no.
                
                # Let me just check: for fixed s, what's the maximum e?
                # The constraint is: for all x in [s,e], inv[x] < s or inv[x] > e
                # Equivalently: for all x in [s,e] with inv[x] >= s, we need inv[x] > e
                # So e < min{inv[x] : x in [s,e], inv[x] >= s}
                # But this is circular because e appears on both sides.
                
                # Better approach: for fixed s, the interval [s,e] is valid iff
                # no pair (x, inv[x]) with x != inv[x] has both x and inv[x] in [s,e],
                # and no self-inverse element is in [s,e].
                
                # Self-inverses: x^2 ≡ 1 (mod p), so x = 1 or x = p-1 = 1008
                # So if s <= 1 <= e or s <= 1008 <= e, invalid.
                # Since s >= 1, if s = 1, then 1 is in the interval, invalid.
                # If e >= 1008, then 1008 is in the interval, invalid.
                
                # For pairs: if x in [s,e] and inv[x] in [s,e] and x != inv[x], invalid.
                # WLOG assume x < inv[x]. Then we need x in [s,e] and inv[x] in [s,e].
                # Since x < inv[x], this means s <= x and inv[x] <= e.
                # So e >= inv[x] for some x in [s,e] with x < inv[x] and x >= s.
                
                # For fixed s, the maximum valid e is:
                # e_max = min(1007, min{inv[x] - 1 : x in [s, 1007], x < inv[x], inv[x] >= s})
                # But we also need x <= e_max, which is automatically satisfied if we take the min properly.
                
                # Actually, let's just iterate e and check validity incrementally.
                pass
        
        # Simpler: for each s, find the largest e such that [s,e] is valid
        # Start with e = s, check if valid, then try to extend
        valid_end = s - 1
        for e in range(s, p):
            # Check if [s, e] is valid
            # Need to check: is there any x in [s,e] such that inv[x] in [s,e]?
            # And no self-inverse in [s,e]
            
            # Quick check for self-inverses
            if s <= 1 <= e or s <= (p-1) <= e:
                break
            
            # Check if any x in [s,e] has inv[x] in [s,e]
            # To do this efficiently, we can precompute for each e the minimum "problem"
            # But for now, let's just check directly for small ranges
            conflict = False
            for x in range(s, e + 1):
                ix = inv[x]
                if s <= ix <= e:
                    conflict = True
                    break
            if conflict:
                break
            
            valid_end = e
        
        if valid_end >= s:
            length = valid_end - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
