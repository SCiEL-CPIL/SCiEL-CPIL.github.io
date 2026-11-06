def can_add(x, S_list, in_set):
    # x is middle
    for a in S_list:
        c = 2 * x - a
        if 1 <= c <= 256 and in_set[c]:
            return False
    # x is end
    for b in S_list:
        a = 2 * b - x
        if 1 <= a <= 256 and in_set[a]:
            return False
    # x is start
    for b in S_list:
        c = 2 * b - x
        if 1 <= c <= 256 and in_set[c]:
            return False
    return True
