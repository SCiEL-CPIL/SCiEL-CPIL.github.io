def construct():
    n = 29
    # Paley coloring: edge (i, j) is color 0 if (j-i) % n is a quadratic residue mod n
    # n=29 is prime, 29 % 4 == 1, so Paley graph is defined.
    # Use a boolean array for O(1) lookup without hash overhead
    is_residue = [False] * n
    for x in range(1, n):
        is_residue[(x * x) % n] = True
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if is_residue[diff]:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
