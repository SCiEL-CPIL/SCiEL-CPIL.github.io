"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Böröczky(12) configuration mapped to integer coordinates.
    # This configuration has 6 ordinary lines, which is the theoretical minimum for 12 points.
    
    # K is the LCM of (1+i) for i=1..6, which is 420.
    K = 420
    
    points = []
    # Points A_i derived from parabola y=x^2 at x=i
    for i in range(1, 7):
        x = K * i // (1 + i)
        y = K * (i * i) // (1 + i)
        points.append([x, y])
        
    # Points B_i derived from points at infinity (1, 2i, 0)
    for i in range(1, 7):
        x = K
        y = 2 * K * i
        points.append([x, y])
        
    return points
# EVOLVE-BLOCK-END
