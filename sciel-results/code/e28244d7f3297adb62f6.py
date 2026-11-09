"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Paley colouring of K29: colour 0 on pairs whose difference is a nonzero square mod 29.
    # This is a known valid Ramsey coloring for K2,2,2, achieving n=29.
    n = 29
    squares = {x * x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            colors.append(0 if diff in squares else 1)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
