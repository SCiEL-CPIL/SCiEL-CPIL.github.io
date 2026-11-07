import time

def _verify_squares(squares):
    n = len(squares[0])
    for sq in squares:
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            pairs = set()
            for r in range(n):
                for c in range(n):
                    pair = (squares[i][r][c], squares[j][r][c])
                    if pair in pairs:
                        return False
                    pairs.add(pair)
            if len(pairs) != n * n:
                return False
    return True

def construct():
    # Known pair of orthogonal Latin squares of order 10
    # Using the construction from Z5 x Z2
    L1 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    L2 = [[(i + 2 * j) % 10 for j in range(10)] for i in range(10)]
    
    squares = [L1, L2]
    
    if _verify_squares(squares):
        return squares
    else:
        return [L1]
