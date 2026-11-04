<<<<<<< SEARCH
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
=======
    # New decomposition strategy:
    # Split A (2x4) into A1 (2x2), A2 (2x1), A3 (2x1)
    # C = A1 B1 + A2 B2 + A3 B3
    # B1 is 2x7, B2 is 1x7, B3 is 1x7
    # R(2x1x7) = 1
    # R(2x2x7) <= R(2x2x4) + R(2x2x3) = 14 + 11 = 25
    # Total Rank = 25 + 1 + 1 = 27
    
    rank = 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
>>>>>>> REPLACE
