def can_add(x):
    if x in S: return False
    for y in S:
        s = x + y
        if counts.get(s, 0) + 1 > 2:
            return False
    s = 2 * x
    if counts.get(s, 0) + 1 > 2:
        return False
    return True
