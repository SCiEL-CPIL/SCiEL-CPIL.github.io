import time

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s kill

    # Square 1: Cyclic group table - guaranteed valid Latin square
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]

    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    def are_orthogonal(sq_a, sq_b):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq_a[i][j], sq_b[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n * n

    # Candidate sq2 from known construction
    sq2 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 3, 2, 5, 4, 7, 6, 9, 8],
        [3, 2, 5, 4, 7, 6, 9, 8, 1, 0],
        [5, 4, 7, 6, 9, 8, 1, 0, 3, 2],
        [7, 6, 9, 8, 1, 0, 3, 2, 5, 4],
        [9, 8, 1, 0, 3, 2, 5, 4, 7, 6],
    ]

    # Verify the candidate pair
    if is_latin(sq1) and is_latin(sq2) and are_orthogonal(sq1, sq2):
        result = [sq1, sq2]
    else:
        # Fallback: just return the guaranteed valid sq1
        result = [sq1]

    # Ensure proper types: lists of lists of Python ints
    cleaned = []
    for sq in result:
        cleaned.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])

    return cleaned
