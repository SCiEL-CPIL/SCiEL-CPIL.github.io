def is_bad(c0, c1, c2, c3):
    # Check if distinct count < 3
    # This means max distinct is 2.
    # So at least 2 elements are equal to some other element.
    # Or simply:
    if c0 == c1 and c0 == c2 and c0 == c3: return True
    if c0 == c1 and c0 == c2: return True # 3 same
    if c0 == c1 and c0 == c3: return True
    if c0 == c2 and c0 == c3: return True
    if c1 == c2 and c1 == c3: return True
    if c1 == c2 and c1 == c0: return True # covered
    # Check for 2 pairs or 3+1
    # If not 3+1, then must be 2+2 to be bad (size 2).
    # Or 2+1+1 is size 3 (good).
    # So bad if:
    # 1. All 4 same
    # 2. 3 same, 1 diff
    # 3. 2 pairs (2,2)
    
    # Let's just use the set. It's C-optimized.
    return len({c0, c1, c2, c3}) < 3
