import time

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, keep margin
    
    # Mapping 0..9 to (Z5, Z2)
    # i -> (i % 5, i // 5)
    def to_coords(i):
        return (i % 5, i // 5)
        
    def from_coords(c):
        return c[1] * 5 + c[0]
        
    squares = []
    
    # Generate candidate squares
    # S_a(i, j) = (i1 + a*j1 % 5, i2 + 1*j2 % 2)
    # We vary 'a' from 1 to 4. 'b' is fixed at 1.
    for a in range(1, 5):
        sq = [[0]*n for _ in range(n)]
        for i in range(n):
            i1, i2 = to_coords(i)
            for j in range(n):
                j1, j2 = to_coords(j)
                val1 = (i1 + a * j1) % 5
                val2 = (i2 + 1 * j2) % 2
                sq[i][j] = from_coords((val1, val2))
        squares.append(sq)
        
    # Verify Latin property
    def is_latin(sq):
        for i in range(n):
            if len(set(sq[i])) != n:
                return False
            col = [sq[r][i] for r in range(n)]
            if len(set(col)) != n:
                return False
        return True
        
    # Verify Orthogonality
    def is_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True
        
    # Filter valid squares
    valid_squares = []
    for sq in squares:
        if not is_latin(sq):
            continue
        # Check against all previously accepted squares
        ok = True
        for vs in valid_squares:
            if not is_orthogonal(vs, sq):
                ok = False
                break
        if ok:
            valid_squares.append(sq)
            
    if not valid_squares:
        # Fallback to the single known square
        return [[[(i+j)%n for j in range(n)] for i in range(n)]]
        
    return valid_squares
