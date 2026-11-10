"""Construct 5 mutually orthogonal Latin squares of order 10 via Kronecker product."""
# EVOLVE-BLOCK-START

def construct():
    n = 10
    m = 5
    # 2x2 Latin square
    L2 = [[0, 1],
          [1, 0]]
    # 4 MOLS of order 5: L_k[r][c] = (c + k*r) % 5 for k in 1..4
    # plus the "trivial" k=0 square for the L2 component
    squares = []
    for k in range(0, 5):  # k=0 -> L2 part; k=1..4 -> order-5 part
        S = [[0] * n for _ in range(n)]
        for i in range(2):
            for j in range(2):
                base = 5 * L2[i][j]
                for r in range(m):
                    for c in range(m):
                        if k == 0:
                            within = 0
                        else:
                            within = (c + k * r) % m
                        S[i * m + r][j * m + c] = base + within
        squares.append(S)
    # Convert to pure Python lists of ints (already ints, but ensure list type)
    return [[[int(v) for v in row] for row in sq] for sq in squares]

# EVOLVE-BLOCK-END
