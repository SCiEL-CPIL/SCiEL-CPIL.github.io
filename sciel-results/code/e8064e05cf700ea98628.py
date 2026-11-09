def construct():
    return [(i + j) % 13 for i in range(16) for j in range(i + 1, 16)]
