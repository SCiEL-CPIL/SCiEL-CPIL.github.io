"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deterministic Paley(29) coloring.
    # This guarantees a valid coloring for n=29 without any risk of invalidity or timeout.
    # The score will be 29.
    
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin within 120s limit
    
    n = 29
    # Compute quadratic residues mod 29
    squares = {x * x % 29 for x in range(1, 29)}
    
    # Generate colors in lexicographic order of (i, j) for i < j
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % 29
            colors.append(0 if diff in squares else 1)
    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
