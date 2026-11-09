def is_bad_fast(c1, c2, c3, c4):
    # 1 distinct
    if c1 == c2 and c2 == c3 and c3 == c4:
        return True
    # 2 distinct
    # 3 same
    if c1 == c2 and c2 == c3 and c3 != c4: return True
    if c1 == c2 and c2 == c4 and c1 != c3: return True
    if c1 == c3 and c3 == c4 and c1 != c2: return True
    if c2 == c3 and c3 == c4 and c2 != c1: return True
    # 2-2 split
    if c1 == c2 and c3 == c4 and c1 != c3: return True
    if c1 == c3 and c2 == c4 and c1 != c2: return True
    if c1 == c4 and c2 == c3 and c1 != c2: return True
    return False
