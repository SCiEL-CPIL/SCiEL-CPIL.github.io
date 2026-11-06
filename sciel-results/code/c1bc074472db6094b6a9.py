def construct():
    n = 30
    p = 31
    # Calculate quadratic residues mod 31
    squares = { (x * x) % p for x in range(1, p) }
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Map i, j to 1..30 for the Paley construction
            # Effectively, we are looking at the subgraph induced by 1..30 in P_31
            # The difference (j - i) mod 31 is the same for (i+1, j+1)
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
