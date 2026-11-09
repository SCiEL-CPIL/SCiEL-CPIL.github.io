def construct():
    n = 29
    # Paley coloring: edge (i, j) is color 0 if (j-i) % n is a quadratic residue mod n
    # n=29 is prime, 29 % 4 == 1, so Paley graph is defined.
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
