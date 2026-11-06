import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Schläfli graph: n=27, triangle-free, alpha=9
    # Vertices 0..26
    # This is the point graph of GQ(2,4)
    # Hardcoded edges for Schläfli graph
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 6], [1, 7], [1, 8], [1, 9], [1, 10],
        [2, 11], [2, 12], [2, 13], [2, 14], [2, 15],
        [3, 16], [3, 17], [3, 18], [3, 19], [3, 20],
        [4, 21], [4, 22], [4, 23], [4, 24], [4, 25],
        [5, 26], [5, 11], [5, 12], [5, 13], [5, 14],
        [6, 15], [6, 16], [6, 17], [6, 18], [6, 19],
        [7, 20], [7, 21], [7, 22], [7, 23], [7, 24],
        [8, 25], [8, 26], [8, 11], [8, 12], [8, 13],
        [9, 14], [9, 15], [9, 16], [9, 17], [9, 18],
        [10, 19], [10, 20], [10, 21], [10, 22], [10, 23],
        [11, 24], [11, 25], [11, 26], [11, 12], [11, 13],
        [12, 14], [12, 15], [12, 16], [12, 17], [12, 18],
        [13, 19], [13, 20], [13, 21], [13, 22], [13, 23],
        [14, 24], [14, 25], [14, 26], [14, 15], [14, 16],
        [15, 17], [15, 18], [15, 19], [15, 20], [15, 21],
        [16, 22], [16, 23], [16, 24], [16, 25], [16, 26],
        [17, 18], [17, 19], [17, 20], [17, 21], [17, 22],
        [18, 23], [18, 24], [18, 25], [18, 26], [18, 19],
        [19, 20], [19, 21], [19, 22], [19, 23], [19, 24],
        [20, 25], [20, 26], [20, 21], [20, 22], [20, 23],
        [21, 24], [21, 25], [21, 26], [21, 22], [21, 23],
        [22, 23], [22, 24], [22, 25], [22, 26], [22, 23],
        [23, 24], [23, 25], [23, 26], [23, 24], [23, 25],
        [24, 25], [24, 26], [24, 25], [24, 26], [25, 26]
    ]
    # Note: The above hardcoded list is a placeholder and likely incorrect/incomplete.
    # I need a correct generation method.
    
    # Correct generation: Schläfli graph is the complement of the line graph of K_{6,6}? No.
    # Let's use the Andrásfai graph A_3(9) which is n=26.
    # It is safer to return a known valid graph.
    
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    
    return {'n': n, 'edges': edges}
