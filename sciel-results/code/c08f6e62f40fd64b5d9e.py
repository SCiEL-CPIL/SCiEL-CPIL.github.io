"""Program for maximizing 3-AP differences in a 32-element set within 1..512."""

def construct():
    # A set designed to cover many AP differences.
    # We combine several arithmetic progressions with different common differences.
    # For a 3-term AP with difference d, we need a, a+d, a+2d in the set.
    
    # Strategy: Use a base set that is a union of intervals and specific points
    # to cover many small d values, and extend to larger values.
    
    # For small d (1 to ~16), a contiguous block of length L covers d up to L-2.
    # If we take {1, 2, ..., 32}, it covers d = 1, 2, ..., 30.
    # But we only have 32 elements.
    
    # Can we do better than 30? The maximum possible d is limited by the range.
    # If we have elements up to 512, d can be up to 255.
    # To cover d, we need three elements spaced by d.
    
    # A known result: for n elements, the maximum number of 3-APs is related to the 
    # structure of the set. But we want distinct differences d, not number of APs.
    
    # Let's try a set that is a union of several blocks.
    # Block 1: {1, 2, ..., 16} -> covers d=1..14
    # Block 2: {32, 33, ..., 47} -> covers d=1..14 (same as before, no new d)
    # This doesn't help for distinct d.
    
    # To cover larger d, we need elements spread out.
    # For d=16, we need a, a+16, a+32. 
    # If we have {1, 17, 33}, that covers d=16.
    # {1, 2, ..., 32} already contains 1, 17, 33? No, 33 is not in 1..32.
    # So {1..32} covers d=1..30. d=16 is covered by (1,17,33)? No, 33 is not there.
    # (1,17,33) requires 33. 
    # (2,18,34) requires 34.
    # ...
    # (1,17,33) not in set. 
    # But (1,17,33) is a 3-AP with d=16.
    # In {1..32}, do we have any 3-AP with d=16?
    # a, a+16, a+32. 
    # If a=1, need 33. No.
    # If a=0, need 32. 0 not in set.
    # So d=16 is NOT covered by {1..32}.
    # Let's check: a=1, a+d=17, a+2d=33. 33 not in set.
    # a=2, a+d=18, a+2d=34. 34 not in set.
    # ...
    # a=1, a+32=33. No.
    # The largest d covered by {1..n} is n-2, with (1, 1+d, 1+2d) where 1+2d <= n.
    # 1+2d <= 32 => 2d <= 31 => d <= 15.
    # So {1..32} covers d=1..15. That's 15 differences.
    # Wait, d=15: (1,16,31). Yes, 31 <= 32.
    # d=16: (1,17,33). 33 > 32. No.
    # So {1..32} has score 15.
    
    # Can we improve?
    # Let's try to extend the set to cover more d values.
    # Replace some elements to get larger d.
    # If we keep {1..30} and add {32, 64, ...}? 
    # We need 32 elements.
    
    # A better construction: 
    # Use {1, 2, ..., 16} union {32, 33, ..., 47} union {64, 65, ..., 79} union ...
    # Each block of 16 covers d=1..14.
    # But we need distinct d across the whole set.
    # Cross-block APs can give larger d.
    
    # Let's try a specific set:
    # S = {1, 2, 3, ..., 16}  # 16 elements
    # S += {32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}  # 16 elements
    # Total 32 elements.
    
    # d=1..14 covered by first block.
    # d=1..14 covered by second block (same d).
    # Cross: 1, 1+16=17? No. 1, 1+32=33, 1+64=65? No.
    # 16, 16+16=32, 16+32=48? 48 not in set.
    # 1, 33, 65? No.
    # 2, 34, 66? No.
    # This doesn't seem to add much.
    
    # Let's go back to basics. The seed {1..32} has score 15.
    # Can we get 16 or more?
    # If we take {1..31} and add 64:
    # d=1..14 covered.
    # d=15: (1,16,31) yes.
    # d=32: (1,33,65)? No. (2,34,66)? No. (32, 64, 96)? 96 no.
    # 32, 64, 96. No.
    # What about d=31? (1,32,63)? 63 no. (33,64,95)? No.
    
    # It seems hard to beat 15 with simple sets.
    # However, there might be a clever set.
    # For now, return the seed set {1..32}.
    
    return list(range(1, 33))
