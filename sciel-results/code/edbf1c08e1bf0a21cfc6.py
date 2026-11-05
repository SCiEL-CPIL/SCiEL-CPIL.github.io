def construct():
    n, m, p = 4, 5, 5
    rank = 57
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Part 1: A1×B11 (4×4 × 4×4 → 4×4), rank 49
    # Uses Strassen
    # ... (existing Strassen code)
    
    # Part 2: A1×B12 (4×4 × 4×1 → 4×1), rank 4
    # C[i,4] += sum_j A[i,j] * B[j,4] for j=0-3
    col = 49
    for i in range(4):
        U[i*5 + 0, col] = 1  # Wait, this isn't right
        ...
