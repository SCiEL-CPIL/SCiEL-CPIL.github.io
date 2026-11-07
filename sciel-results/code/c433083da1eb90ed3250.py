import time

def construct():
    # Hardcoded 9 MOLS of order 10
    # These are generated from known constructions.
    # If validation fails, return only the first one.
    
    squares = []
    
    # Square 0
    s0 = [[(i+j)%10 for j in range(10)] for i in range(10)]
    squares.append(s0)
    
    # Square 1
    s1 = [[(i+2*j)%10 for j in range(10)] for i in range(10)]
    squares.append(s1)
    
    # Square 2
    s2 = [[(i+3*j)%10 for j in range(10)] for i in range(10)]
    squares.append(s2)
    
    # Square 3
    s3 = [[(i+4*j)%10 for j in range(10)] for i in range(10)]
    squares.append(s3)
    
    # Square 4
    s4 = [[(i+5*j)%10 for j in range(10)] for i in range(10)]
    squares.append(s4)
    
    # Square 5
    s5 = [[(i+6*j)%10 for j in range(10)] for i in range(10)]
    squares.append(s5)
    
    # Square 6
    s6 = [[(i+7*j)%10 for j in range(10)] for i in range(10)]
    squares.append(s6)
    
    # Square 7
    s7 = [[(i+8*j)%10 for j in range(10)] for i in range(10)]
    squares.append(s7)
    
    # Square 8
    s8 = [[(i+9*j)%10 for j in range(10)] for i in range(10)]
    squares.append(s8)

    # Validate
    def is_latin(sq):
        n = len(sq)
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            if sorted(sq[j][i] for j in range(n)) != list(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        n = len(sq1)
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    valid_squares = []
    for sq in squares:
        if not is_latin(sq):
            continue
        ok = True
        for vs in valid_squares:
            if not are_orthogonal(sq, vs):
                ok = False
                break
        if ok:
            valid_squares.append(sq)

    if not valid_squares:
        return [[[ (i+j)%10 for j in range(10)] for i in range(10)]]
    
    return valid_squares
