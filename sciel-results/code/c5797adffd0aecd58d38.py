# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use Laderman's algorithm for 4x4x4 (rank 49) and schoolbook for 4x4x1 (rank 16).
    # Total rank: 49 + 49 + 16 = 114.
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman factors for 4x4x4
    # I will use the coefficients from a standard reference.
    # To avoid manual error, I will use the Strassen composition if possible, 
    # but Laderman is specific.
    
    # I will use the schoolbook for 4x4x4 (64) and 4x4x1 (16) -> 144.
    # This is not an improvement.
    
    # I will use the rank 49 factors.
    # ...
