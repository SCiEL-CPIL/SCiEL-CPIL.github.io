"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    # For each possible start position, find the maximum valid end
    for a in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Skip self-inverse residues: 1 and 1008 (since 1*1 % 1009 = 1, 1008*1008 % 1009 = 1)
        if a == 1 or a == 1008:
            continue
        
        # Find the maximum b >= a such that for all x, y in [a, b], x*y % 1009 != 1
        # The constraint is: for every x in [a,b], inv(x) must NOT be in [a,b]
        # (since if inv(x) is in [a,b], then x * inv(x) % 1009 = 1)
        # Also need to check x=x: x*x % 1009 != 1, which means x != 1 and x != 1008 (already handled)
        
        # We need to find the smallest b >= a such that the interval [a, b] contains
        # both x and inv(x) for some x, OR contains a self-inverse element.
        # Actually, we want the LARGEST valid interval starting at a.
        
        # An interval [a, b] is invalid if there exists x in [a, b] such that inv(x) is also in [a, b].
        # So we need to find the first position where adding it would create a conflict.
        
        # Let's think differently: for a fixed start a, what is the maximum end b?
        # The interval [a, b] is valid iff for all x in [a, b], inv(x) is NOT in [a, b].
        # This means: for all x in [a, b], either inv(x) < a or inv(x) > b.
        
        # So the constraint on b is: b < min{inv(x) : x in [a, b] and inv(x) > b} -- this is circular.
        
        # Better approach: for a fixed a, the maximum valid b is the minimum over all x >= a of:
        #   - if inv(x) >= a, then b must be < inv(x) (since if b >= inv(x), both x and inv(x) are in [a,b])
        #   - but also x itself must be <= b
        # So for each x >= a, if inv(x) >= a, then b must be < inv(x).
        # The tightest such constraint gives us the max b.
        
        # But we also need x to be in [a, b], so x <= b.
        # And we need to check that for the chosen b, no x in [a,b] has inv(x) in [a,b].
        
        # Let's compute: for start a, find the minimum inv(x) over all x >= a where inv(x) >= a.
        # Then b_max = that minimum - 1.
        # But we also need b_max >= a (otherwise the interval is just [a] which might be valid if a*a != 1).
        
        # Wait, we also need to make sure that for all x in [a, b_max], inv(x) is not in [a, b_max].
        # If b_max = min{inv(x) : x >= a, inv(x) >= a} - 1, then for any x in [a, b_max]:
        #   if inv(x) >= a, then inv(x) >= min{inv(x') : x' >= a, inv(x') >= a} > b_max
        #   so inv(x) > b_max, hence not in [a, b_max].
        #   if inv(x) < a, then inv(x) is not in [a, b_max] either.
        # So this works!
        
        # But we need to be careful: the min should be over x in [a, ...] where inv(x) >= a.
        # If no such x exists, then there's no upper bound from this constraint, and b can be as large as n.
        
        min_inv_above_a = n + 1  # infinity
        for x in range(a, n + 1):
            if inv[x] >= a:
                if inv[x] < min_inv_above_a:
                    min_inv_above_a = inv[x]
        
        if min_inv_above_a <= n:
            b_max = min_inv_above_a - 1
        else:
            b_max = n
        
        # Ensure b_max >= a
        if b_max < a:
            continue
        
        length = b_max - a + 1
        if length > best_len:
            best_len = length
            best_start = a
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
