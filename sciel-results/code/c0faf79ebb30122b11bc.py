def _winograd_3x3():
    # Winograd's algorithm for 3x3 matrix multiplication
    # Rank 23
    # Coefficients from literature
    # U, V, W are 9x23
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # ... filling coefficients ...
    # This is risky without exact coefficients.
