def construct():
    p = 67
    # Compute quadratic residues mod p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Build orientation: one bit per pair (i, j) with i < j
    # 0 means i -> j, 1 means j -> i
    # i -> j iff (j - i) % p is in squares
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    
    return {'n': p, 'orientation': orientation}
