import time

def construct():
    start_time = time.time()
    p = 1009
    max_val = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Self-inverse elements: x where x^2 ≡ 1 (mod p), i.e., x = 1 or x = p-1
    self_inv = {1, p - 1}
    
    # For each element x, record the minimum partner in its inverse pair
    # If inv[x] < x, then the "lower" partner is inv[x]
    # If inv[x] > x, then x is the lower partner (but we only need to track when x is the upper)
    # Actually, let's think differently: for a pair (a, b) with a < b and a*b ≡ 1 (mod p),
    # both a and b cannot be in the interval. So for position b, the constraint is that
    # the interval start must be > a (i.e., start >= a+1) to avoid including a.
    # Equivalently, if we fix the right endpoint at b, the left endpoint must be > max_a_for_b[b].
    
    # Let's define: for each b, min_valid_start[b] = max(a+1 for all pairs (a,b) with a<b)
    # Actually, simpler: for each b, the smallest start that avoids all conflicts ending at b
    # is 1 + max{a : (a,b) is a conflict pair, a < b}
    
    # Let's use a different approach: two pointers
    # For a valid interval [s, e]:
    #   - s >= 2 and e <= p-2 (to avoid self-inverse 1 and p-1)
    #   - For every x in [s, e], inv[x] is NOT in [s, e]
    #     This means: for every x in [s, e], either inv[x] < s or inv[x] > e
    
    # Using two pointers: fix s, find max e such that [s, e] is valid
    # Or: fix e, find min s such that [s, e] is valid
    
    # Let's precompute for each x, what is the "other" element in its inverse pair
    # For x in [2, p-2], partner[x] = inv[x] (which is also in [2, p-2] unless x is self-inverse)
    # Self-inverse in [2, p-2]: none, since only 1 and p-1 are self-inverse
    
    # Two-pointer approach:
    # Start with s=2, e=2
    # Expand e as far as possible while maintaining validity
    # When we can't expand e, increment s and try again
    
    # Actually, let's use the constraint: for interval [s, e] to be valid,
    # for all x in [s, e], inv[x] must be outside [s, e]
    # This is equivalent to: max{inv[x] : x in [s,e], inv[x] in [s,e]} does not exist
    # Or: min{inv[x] : x in [s,e]} < s or max{inv[x] : x in [s,e]} > e... no that's not right
    
    # Better: for each x, if inv[x] is in [s,e], then we have a conflict.
    # So we need: for all x in [s,e], inv[x] < s or inv[x] > e
    
    # Precompute: for each x in [2, p-2], let low[x] = min(x, inv[x]), high[x] = max(x, inv[x])
    # A pair (low[x], high[x]) conflicts if both are in [s,e]
    # So for a given [s,e], it's valid iff for all x in [s,e], high[x] > e or low[x] < s
    # Equivalently: there is no x in [s,e] such that low[x] >= s and high[x] <= e
    
    # Let's precompute: for each possible right endpoint e, what is the maximum low[x] 
    # among all x in [2, e] such that high[x] <= e?
    # Then [s, e] is valid iff s > max_low_for_e[e]
    
    # Actually, let's just do it directly with two pointers and a constraint array
    
    # For each position i, define constraint[i] = the minimum start s such that 
    # if i is in the interval, s must be > some value to avoid the partner of i
    # If inv[i] < i, then to include i, we need s > inv[i] (i.e., s >= inv[i]+1)
    # If inv[i] > i, then to include i, we need e < inv[i] (i.e., e <= inv[i]-1)
    # But we're fixing e and varying s, so the constraint from i (when inv[i] > i) is on e, not s
    
    # Let me reconsider. For a fixed e, find the minimum valid s.
    # [s, e] is valid iff:
    #   1. s >= 2, e <= p-2
    #   2. For all x in [s, e]: inv[x] not in [s, e]
    #      This breaks into:
    #      a. For all x in [s, e] with inv[x] < x: inv[x] < s  (i.e., s > inv[x])
    #      b. For all x in [s, e] with inv[x] > x: inv[x] > e  (i.e., e < inv[x])
    #      
    # Condition (b) depends only on e: for a fixed e, condition (b) is satisfied iff
    #   for all x in [2, e] with inv[x] > x: inv[x] > e
    #   i.e., max{inv[x] : x in [2,e], inv[x] > x} > e... no, we need ALL such inv[x] > e
    #   i.e., min{inv[x] : x in [2,e], inv[x] > x} > e
    #   If this min is <= e, then there's some x <= e with inv[x] in [x+1, e], causing a conflict
    #   Wait, no. If x is in [s,e] and inv[x] > x, we need inv[x] > e.
    #   So condition (b) for fixed e: for all x in [2, e] (since x must be >= s >= 2), 
    #   if inv[x] > x, then inv[x] > e.
    #   Equivalently: min{inv[x] : x in [2, e], inv[x] > x} > e, or the set is empty.
    #   
    # Condition (a) for fixed e and variable s: for all x in [s, e] with inv[x] < x:
    #   inv[x] < s, i.e., s > inv[x].
    #   So s > max{inv[x] : x in [s, e], inv[x] < x}
    #   But this is circular since the set depends on s.
    #   
    # Alternative: s must be greater than max{inv[x] : x in [2, e], inv[x] < x, AND x >= s}
    # This is still circular.
    
    # Let's go back to the original approach but optimize it.
    # For each s, find the maximum e such that [s, e] is valid.
    # As e increases, the constraint is: for all b in [s+1, e], max_a_for_b[b] < s
    # where max_a_for_b[b] = max{a : (a,b) is a conflict pair, a < b}
    # 
    # So for fixed s, the maximum valid e is the smallest b in [s+1, p-2] such that 
    # max_a_for_b[b] >= s, minus 1. If no such b exists, e = p-2.
    #
    # We can precompute max_a_for_b, then for each s, we need to find the first b >= s+1 
    # where max_a_for_b[b] >= s.
    #
    # This is still O(n²) in the worst case, but n=1006 is small enough.
    # However, we can optimize: as s increases, the valid e can only decrease or stay the same?
    # No, actually as s increases, the constraint max_a_for_b[b] >= s becomes harder to satisfy,
    # so e can only increase or stay the same. Wait, no:
    # For larger s, we're starting later, so we might avoid more conflicts, allowing larger e.
    # But we also have less room (e <= p-2).
    
    # Let's just do the O(n²) approach since n is small (~1006), but with early termination.
    
    max_a_for_b = [0] * (p + 1)
    for i in range(2, p - 1):  # i in [2, 1007]
        j = inv[i]
        if j == i:
            continue
        a = min(i, j)
        b = max(i, j)
        if a > max_a_for_b[b]:
            max_a_for_b[b] = a
    
    best_len = 0
    best_start = 2
    
    # For each start s in [2, p-2], find max valid e
    for s in range(2, p - 1):
        # Early termination: even if we go to p-2, can we beat best?
        max_possible_len = (p - 2) - s + 1
        if max_possible_len <= best_len:
            break
        
        # Find the first b in [s+1, p-2] where max_a_for_b[b] >= s
        valid_e = p - 2
        for b in range(s + 1, p - 1):  # b in [s+1, p-2]
            if max_a_for_b[b] >= s:
                valid_e = b - 1
                break
        
        length = valid_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
        
        # Time guard
        if time.time() - start_time > 85:
            break
    
    return [best_start, best_len]
