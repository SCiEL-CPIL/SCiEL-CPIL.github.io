"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008
    # Precompute modular inverses in [1, 1008]
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    # Try all possible intervals
    for start in range(1, n + 1):
        # For each start, find the maximum end such that no x,y in [start, end] have x*y % p == 1
        # We need to check all pairs. But we can be smarter.
        # For each x in [start, ...], if inv[x] is also in the interval, it's invalid.
        # So the interval must not contain both x and inv[x] for any x.
        # Also x != inv[x] except for self-inverses (1 and p-1=1008), which are forbidden.
        
        # Let's build the interval greedily or just check all ends.
        # Since n is small (1008), we can just iterate end from start to n and check validity.
        # But checking all pairs is O(len^2) which is 1008^2 * 1008 = too slow.
        
        # Better approach: for a given interval [s, e], it's valid iff for all x in [s,e], inv[x] is NOT in [s,e].
        # Equivalently, for all x in [s,e], inv[x] < s or inv[x] > e.
        
        # Let's iterate start, and for each start, find the maximum end.
        # As end increases, the condition becomes harder to satisfy.
        # Actually, if [s, e] is invalid, then [s, e+1] might be valid? No, because if [s,e] has a bad pair, adding more elements won't fix it.
        # So for fixed s, valid e's form a prefix (from s up to some max_e).
        
        # Wait, that's not right. If [s, e] is valid, [s, e-1] is also valid. So valid e's for fixed s form an initial segment.
        # So we can binary search or just linearly find the max valid e.
        
        # Let's just iterate end from start to n and check if [start, end] is valid.
        # To check if [start, end] is valid, we need to ensure no x in [start, end] has inv[x] in [start, end].
        # We can precompute for each start, the "blocking" elements.
        
        # Alternative: for each pair (x, inv[x]) with x < inv[x], they cannot both be in the interval.
        # So the interval must not contain any complete pair (x, inv[x]).
        # Also must not contain 1 or 1008.
        
        # Let's just do it straightforwardly for n=1008.
        # For each start (1 to 1008), for each end (start to 1008), check validity.
        # Validity check: for all x in [start, end], if inv[x] >= start and inv[x] <= end, return False.
        # This is O(n^3) in worst case = 1008^3 ≈ 1e9, too slow.
        
        # Let's think differently.
        # For a fixed start s, what's the maximum valid end e?
        # An element x in [s, e] is "bad" if inv[x] is also in [s, e].
        # So e must be < min{ inv[x] for x in [s, e] if inv[x] >= s }
        # This is circular. Let's just iterate e and maintain the constraint.
        
        # Actually, let's precompute for each possible start s, the list of "forbidden ends" caused by each x >= s.
        # For x in [s, 1008]:
        #   if inv[x] >= s and inv[x] != x: then x and inv[x] form a pair. If both are in [s, e], it's invalid.
        #   So if x is in [s, e], then e must be < inv[x] (if inv[x] > x) or... 
        #   Actually, if x < inv[x], then if x is in the interval, e must be < inv[x].
        #   If x > inv[x], then if x is in the interval, e must be < x is automatic, but we need start <= inv[x] to be false, i.e., start > inv[x]. But start is fixed.
        
        # Let me just do O(n^2) with early termination.
        # For each start s:
        #   For each end e from s to n:
        #     Check if [s, e] is valid by scanning x from s to e.
        #     If invalid, break (since larger e will also be invalid? No, that's not true. Wait, if [s,e] is invalid, [s,e+1] is also invalid because the bad pair is still there. Yes!)
        #     So we can break at the first invalid e.
        
        # Wait, is that true? If [s,e] is invalid, there exists x,y in [s,e] with x*y%1009==1. Then for any e' > e, x,y are still in [s,e'], so [s,e'] is also invalid.
        # Yes! So for fixed s, the valid e's form a prefix: s, s+1, ..., e_max.
        
        # So for each s, we find the largest e such that [s,e] is valid.
        # We can do this by iterating e from s to n, and checking validity incrementally.
        
        # To check validity of [s, e] given that [s, e-1] was valid:
        #   We only need to check if the new element e forms a bad pair with any element in [s, e-1].
        #   I.e., check if inv[e] is in [s, e-1].
        #   Also check if e is a self-inverse (e*e % 1009 == 1), which means e=1 or e=1008.
        
        # So:
        for s in range(1, n + 1):
            # Start with e = s
            # Check if [s, s] is valid: s*s % 1009 != 1
            if (s * s) % p == 1:
                continue  # s is self-inverse, can't be in any valid interval
            
            e = s
            valid = True
            # We'll extend e as long as possible
            for e_candidate in range(s + 1, n + 1):
                # Check if adding e_candidate to [s, e_candidate-1] creates a bad pair
                # Bad if inv[e_candidate] is in [s, e_candidate-1]
                # Also, e_candidate itself might be self-inverse
                if (e_candidate * e_candidate) % p == 1:
                    break  # e_candidate is self-inverse, can't add it
                inv_ec = inv[e_candidate]
                if inv_ec >= s and inv_ec < e_candidate:
                    break  # inv[e_candidate] is already in the interval
                e = e_candidate
            
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
