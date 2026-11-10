def construct():
    n = 10
    return [[[(i + j) % n for j in range(n)] for i in range(n)]]
