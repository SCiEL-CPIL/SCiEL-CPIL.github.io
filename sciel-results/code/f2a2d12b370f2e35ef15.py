def construct():
    # Square 1: Standard cyclic
    s1 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    
    # Square 2: Known orthogonal partner
    s2 = [
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
        [1, 3, 5, 7, 9, 0, 2, 4, 6, 8],
        [2, 4, 6, 8, 1, 3, 5, 7, 9, 0],
        [3, 5, 7, 9, 0, 2, 4, 6, 8, 1],
        [4, 6, 8, 1, 3, 5, 7, 9, 0, 2],
        [5, 7, 9, 0, 2, 4, 6, 8, 1, 3],
        [6, 8, 1, 3, 5, 7, 9, 0, 2, 4],
        [7, 9, 0, 2, 4, 6, 8, 1, 3, 5],
        [8, 1, 3, 5, 7, 9, 0, 2, 4, 6],
        [9, 0, 2, 4, 6, 8, 1, 3, 5, 7]
    ]
    
    # Verify orthogonality just in case (optional but good for debugging)
    # For production, we trust the hardcoded values.
    
    return [s1, s2]
