def construct():
    S = set()
    # 8 blocks of 4
    # Spacing 64
    for k in range(8):
        base = 1 + k * 64
        S.add(base)
        S.add(base + 1)
        S.add(base + 2)
        S.add(base + 3)
    return sorted(S)
