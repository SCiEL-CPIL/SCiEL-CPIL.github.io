"""Rewritten program using Paley graph construction for n=29."""

def construct():
    # Use Paley graph of order 29
    # Vertices are 0..28
    # Edge (i, j) is color 0 if (j - i) % 29 is a quadratic residue mod 29
    # Otherwise color 1
    
    p = 29
    n = p
    
    # Calculate quadratic residues mod p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
