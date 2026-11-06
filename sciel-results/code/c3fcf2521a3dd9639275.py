def is_invalid_colors(c1, c2, c3, c4):
    # Check if all same
    if c1 == c2 == c3 == c4: return True
    # Check if exactly 2 colors
    # This happens if one color appears 3 times and other 1, or 2 and 2.
    # Faster check:
    # If c1 != c2 and c1 != c3 and c1 != c4, then we have at least 3 distinct (c1, c2, c3)
    # Wait, if c1=0, c2=1, c3=2, c4=0 -> {0,1,2} size 3. Valid.
    # If c1=0, c2=1, c3=0, c4=1 -> {0,1} size 2. Invalid.
    
    # Simple check:
    # Count distinct
    # We can do:
    if c1 == c2:
        if c3 == c4:
            # 0,0,1,1 or 0,0,0,0
            if c1 == c3: return True
            else: return True # 0,0,1,1 -> 2 colors
        else:
            # 0,0,1,2 or 0,0,1,1 (handled)
            # If c3 != c4, and c1==c2, then we have c1, c3, c4.
            # If c3 == c1, then c1,c1,c1,c4 -> 2 colors.
            # If c3 != c1, then c1,c1,c3,c4. If c3!=c4, we have 3 distinct?
            # c1, c3, c4. If c3!=c1 and c4!=c1 and c3!=c4, then 3 distinct.
            # If c3==c1, invalid.
            # If c4==c1, invalid.
            # If c3==c4, invalid (0,0,1,1).
            if c3 == c1 or c4 == c1 or c3 == c4: return True
            else: return False
    else:
        # c1 != c2
        if c1 == c3:
            # 0,1,0,?
            # If c4 == 0 or c4 == 1, then 2 colors.
            # If c4 != 0 and c4 != 1, then 3 colors.
            if c4 == c1 or c4 == c2: return True
            else: return False
        elif c1 == c4:
            # 0,1,?,0
            # If c3 == 0 or c3 == 1, 2 colors.
            if c3 == c1 or c3 == c2: return True
            else: return False
        elif c2 == c3:
            # 0,1,1,?
            # If c4 == 0 or c4 == 1, 2 colors.
            if c4 == c1 or c4 == c2: return True
            else: return False
        elif c2 == c4:
            # 0,1,?,1
            if c3 == c1 or c3 == c2: return True
            else: return False
        elif c3 == c4:
            # 0,1,?,? (c3==c4)
            # If c3 == 0 or c3 == 1, 2 colors.
            if c3 == c1 or c3 == c2: return True
            else: return False
        else:
            # All distinct or 3 distinct?
            # c1!=c2, c1!=c3, c1!=c4, c2!=c3, c2!=c4, c3!=c4
            # Then 4 distinct. Valid.
            return False
