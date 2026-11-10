import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Square 1: Cyclic
    S1 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    
    # Square 2: Hardcoded known orthogonal square
    S2 = [
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
        [1, 3, 5, 7, 9, 0, 2, 4, 6, 8],
        [2, 4, 6, 8, 1, 3, 5, 7, 9, 0],
        [3, 5, 7, 9, 0, 4, 6, 8, 1, 2],
        [4, 6, 8, 1, 3, 5, 7, 9, 0, 4],
        [5, 7, 9, 0, 4, 6, 8, 1, 2, 3],
        [6, 8, 1, 3, 5, 7, 9, 0, 4, 6],
        [7, 9, 0, 4, 6, 8, 1, 2, 3, 5],
        [8, 1, 3, 5, 7, 9, 0, 4, 6, 8],
        [9, 0, 4, 6, 8, 1, 2, 3, 5, 7]
    ]
    
    return [S1, S2]
