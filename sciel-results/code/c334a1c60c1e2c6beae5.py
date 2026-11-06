"""Rewritten program to maximize n for the Ramsey number R(K2,2,2)."""

def construct():
    # The Ramsey number R(K2,2,2, K2,2,2) is known to be 30.
    # This means the maximum n for a valid 2-coloring is 29.
    # The Paley graph on 29 vertices provides such a coloring.
    # 29 is a prime p congruent to 1 mod 4, so the Paley graph is self-complementary.
    n = 29
    
    # Generate the set of quadratic residues modulo 29
    # These correspond to the edges of color 0.
    # The non-residues correspond to the edges of color 1.
    # Since the graph is self-complementary, both color classes are isomorphic
    # and thus both are K2,2,2-free if one is.
    
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
