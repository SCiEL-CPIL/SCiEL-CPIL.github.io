"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4x4x4 part using 8x Strassen 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Block indices
                # A rows: 2i, 2i+1; Cols: 2j, 2j+1
                # B Rows: 2j, 2j+1; Cols: 2k, 2k+1
                # C Rows: 2i, 2i+1; Cols: 2k, 2k+1
                
                # M1
                U[2*i*m + 2*j][col] += 1
                U[2*i*m + 2*j+1][col] += 1
                V[2*j*p + 2*k][col] += 1
                V[2*j*p + 2*k+1][col] += 1
                W[2*k*n + 2*i][col] += 1
                W[2*k*n + 2*i+1][col] += 1
                col += 1
                
                # M2
                U[2*i*m + 2*j+1][col] += 1
                U[2*i*m + 2*j+1][col] += 1 # Wait, A21+A22
                U[2*i*m + 2*j][col] += 1
                # Correct: A21, A22
                U[2*i*m + 2*j+1][col] = 1
                U[2*i*m + 2*j+1][col] = 1 # No, A21 is (2i+1, 2j), A22 is (2i+1, 2j+1)
                # Reset and do properly
                # I will use a helper to add terms
                pass
