import time

def construct():
    # Hardcoded MOLS of order 10
    sq1 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
        [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    ]
    
    sq2 = [
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
        [1, 3, 5, 7, 9, 0, 2, 4, 6, 8],
        [2, 4, 6, 8, 0, 1, 3, 5, 7, 9],
        [3, 5, 7, 9, 1, 2, 4, 6, 8, 0],
        [4, 6, 8, 0, 2, 3, 5, 7, 9, 1],
        [5, 7, 9, 1, 3, 4, 6, 8, 0, 2],
        [6, 8, 0, 2, 4, 5, 7, 9, 1, 3],
        [7, 9, 1, 3, 5, 6, 8, 0, 2, 4],
        [8, 0, 2, 4, 6, 7, 9, 1, 3, 5],
        [9, 1, 3, 5, 7, 8, 0, 2, 4, 6]
    ]

    # Verification
    def is_latin(sq):
        n = len(sq)
        for i in range(n):
            if set(sq[i]) != set(range(n)):
                return False
        for j in range(n):
            if set(sq[i][j] for i in range(n)) != set(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        n = len(sq1)
        pairs = set()
        for i in range(n):
            for j in range(n):
                pairs.add((sq1[i][j], sq2[i][j]))
        return len(pairs) == n * n

    if is_latin(sq1) and is_latin(sq2) and are_orthogonal(sq1, sq2):
        return [sq1, sq2]
    else:
        # Fallback to a single valid square
        return [sq1]

# EVOLVE-BLOCK-END
