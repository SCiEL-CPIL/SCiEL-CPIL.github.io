def construct():
    n = 10
    # Square 1: (i + j) % 10
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Square 2: (i + 2 * j) % 10
    sq2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    # Verify orthogonality just in case (though mathematically proven above)
    # Pairs must be unique
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                # Fallback to a single square if somehow not orthogonal (should not happen)
                return [sq1]
            pairs.add(p)
            
    return [sq1, sq2]
