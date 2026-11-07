def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 4 * j) % n for j in range(n)] for i in range(n)]
    return [L1, L2]
