# EVOLVE-BLOCK-START
def construct():
    n = 10
    squares = []
    for k in range(1, 5):
        sq = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            r2 = i // 5
            r5 = i % 5
            for j in range(n):
                c2 = j // 5
                c5 = j % 5
                u = (r5 + k * c5) % 5
                v = c2
                sq[i][j] = u + 5 * v
        squares.append(sq)
    return squares
# EVOLVE-BLOCK-END
