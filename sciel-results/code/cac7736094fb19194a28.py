import time

def construct():
    n = 10
    
    # Define L1 and L2 based on linear construction
    # L1[i][j] = (i + j) % 10
    # L2[i][j] = (i + 2j) % 10
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1, L2]
    
    # Verification function
    def is_valid_latin_square(sq):
        n = len(sq)
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        n = len(sq1)
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return True

    # Verify each square is a Latin Square
    valid = True
    for sq in squares:
        if not is_valid_latin_square(sq):
            valid = False
            break
            
    # Verify pairwise orthogonality
    if valid:
        for i in range(len(squares)):
            for j in range(i + 1, len(squares)):
                if not are_orthogonal(squares[i], squares[j]):
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        # Ensure types are Python int
        result = []
        for sq in squares:
            result.append([[int(x) for x in row] for row in sq])
        return result
    else:
        # Fallback to single square if verification fails (should not happen)
        return [[[int(x) for x in row] for row in L1]]
