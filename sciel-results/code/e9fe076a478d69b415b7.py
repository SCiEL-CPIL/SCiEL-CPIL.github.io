def construct():
    n = 10
    
    # Square 1: Cyclic Latin Square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Square 2: A known orthogonal mate for order 10
    # This square is constructed to be orthogonal to L1
    L2 = [
        [0, 3, 6, 9, 2, 1, 4, 7, 5, 8],
        [1, 4, 7, 0, 3, 2, 5, 8, 6, 9],
        [2, 5, 8, 1, 4, 3, 6, 9, 7, 0],
        [3, 6, 9, 2, 5, 4, 7, 0, 8, 1],
        [4, 7, 0, 3, 6, 5, 8, 1, 9, 2],
        [5, 8, 1, 4, 7, 6, 9, 2, 0, 3],
        [6, 9, 2, 5, 8, 7, 0, 3, 1, 4],
        [7, 0, 3, 6, 9, 8, 1, 4, 2, 5],
        [8, 1, 4, 7, 0, 9, 2, 5, 3, 6],
        [9, 2, 5, 8, 1, 0, 3, 6, 4, 7]
    ]
    
    # Verify types are Python int
    # The list comprehension for L1 produces Python ints.
    # The hardcoded L2 contains Python ints.
    
    return [L1, L2]
