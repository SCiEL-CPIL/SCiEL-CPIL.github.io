def construct():
    import time
    deadline = time.time() + 60.0

    # Kronecker product construction:
    # 1 MOLS of order 2 (L2) combined with 4 MOLS of order 5 (finite field F5)
    # gives 4 mutually orthogonal Latin squares of order 10.
    
    # The 2x2 Latin square
    L2 = [[0, 1], [1, 0]]
    
    # 4 MOLS of order 5 using finite field F5
    # L_k[i][j] = (k * i + j) % 5 for k in 1,2,3,4
    n5 = 5
    squares_5 = []
    for k in range(1, 5):
        sq = [[(k * i + j) % n5 for j in range(n5)] for i in range(n5)]
        squares_5.append(sq)
    
    # Construct 10x10 squares via Kronecker product
    # For each of the 4 order-5 squares, form a 10x10 square
    n10 = 10
    result = []
    for idx, sq5 in enumerate(squares_5):
        sq10 = [[0]*n10 for _ in range(n10)]
        for r in range(n10):
            for c in range(n10):
                # Position in L2: (r//5, c//5)
                lr, lc = r // 5, c // 5
                # Position in sq5: (r%5, c%5)
                r5, c5 = r % 5, c % 5
                # Symbol: 5 * L2[lr][lc] + sq5[r5][c5]
                sq10[r][c] = 5 * L2[lr][lc] + sq5[r5][c5]
        # Ensure all values are Python ints
        sq10 = [[int(x) for x in row] for row in sq10]
        result.append(sq10)
    
    return result
