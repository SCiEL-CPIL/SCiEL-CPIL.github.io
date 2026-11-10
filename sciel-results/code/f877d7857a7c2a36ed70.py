"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 90.0  # seconds, with safety margin
    
    p = 1009
    # For each x in [1, 1008], compute its modular inverse mod p
    # x * inv(x) ≡ 1 (mod p)
    # We need to find the longest interval [s, s+l-1] in [1, 1008]
    # such that for all x, y in the interval, x*y % p != 1
    
    # First, let's identify which pairs (x, y) are "forbidden" - i.e., x*y ≡ 1 (mod p)
    # For each x, there's exactly one y = x^(-1) mod p such that x*y ≡ 1 (mod p)
    # Note: x=1 has inverse 1, x=1008 has inverse 1008 (since 1008 ≡ -1, (-1)^2 = 1)
    
    # Build a set of forbidden pairs
    # For an interval to be valid, it must not contain any forbidden pair
    # A forbidden pair is (x, y) where x*y ≡ 1 (mod 1009)
    
    # For each x in [1, 1008], compute inv_x
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Now, for each x, the "partner" that conflicts with it is inv[x]
    # An interval is valid if for all x in the interval, inv[x] is NOT in the interval
    
    # Let's precompute for each position i (1 to 1008), which positions are its conflicts
    # conflict[i] = inv[i] (the unique partner)
    
    # Now we need the longest contiguous interval [s, e] such that:
    # For all x in [s, e], inv[x] is not in [s, e]
    # Equivalently: the interval [s, e] contains no pair {x, inv[x]} for any x
    
    # This is equivalent to: no x in [s, e] has inv[x] also in [s, e]
    
    # Let's use a two-pointer/sliding window approach
    # Or we can do a direct search
    
    # For each starting position s, find the maximum ending position e
    # such that the interval [s, e] is valid
    
    best_start = 1
    best_len = 0
    
    # For each s from 1 to 1008, extend e as far as possible
    for s in range(1, 1009):
        if time.time() - start_time > deadline - 5:
            break
        # Try to extend the interval starting at s
        e = s
        valid = True
        while e <= 1008:
            # Check if adding e to the interval [s, e-1] keeps it valid
            # We need to check: is inv[e] in [s, e-1]? And is e in [s, inv[e]]?
            # Actually, we need to check that for all x in [s, e], inv[x] is not in [s, e]
            # When we add e, we need to verify:
            # 1. inv[e] is not in [s, e-1] (i.e., inv[e] < s or inv[e] > e-1)
            # 2. For all x in [s, e-1], inv[x] != e (i.e., e is not the inverse of any x in [s, e-1])
            # But condition 2 is equivalent to: e is not in {inv[x] : x in [s, e-1]}
            # Which means: for all x in [s, e-1], x != inv[e]... no wait
            
            # Let me reconsider. The condition is: for all x, y in [s, e], x*y % p != 1
            # This means: there is no pair (x, y) in [s, e] × [s, e] such that x*y ≡ 1 (mod p)
            # For each x, there's exactly one y = inv[x] that conflicts
            # So the interval is invalid if there exists x in [s, e] such that inv[x] is also in [s, e]
            
            # When extending from [s, e-1] to [s, e], we need to check:
            # - Is inv[e] in [s, e]? If yes, invalid.
            # - But also, for existing x in [s, e-1], is inv[x] == e? If so, invalid.
            # The second condition is: is there x in [s, e-1] with inv[x] = e?
            # This is equivalent to: is inv[e] in [s, e-1]? Because inv[inv[e]] = e
            # So both conditions reduce to: is inv[e] in [s, e]?
            
            if s <= inv[e] <= e:
                valid = False
                break
            e += 1
        
        if not valid:
            e -= 1  # e was incremented past the last valid position
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
