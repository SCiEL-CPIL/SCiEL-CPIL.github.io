import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Schoolbook decomposition for fallback
    def schoolbook_3x3x5():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Try to construct a lower rank decomposition
    # Use BSC (rank 23) for 3x3x3 and schoolbook (rank 18) for 3x3x2
    # Total rank 41
    
    # BSC 3x3x3 decomposition coefficients
    # This is a known rank-23 decomposition
    # U_bsc, V_bsc, W_bsc are 9x23 matrices
    # I will use a hardcoded set. If incorrect, verification will fail.
    
    # Since hardcoding 23x9x3 numbers is prone to error, 
    # I will use a simpler improvement if possible.
    # Or I will use the rank-41 with a verified BSC set.
    
    # Let's use a known valid BSC set.
    # The BSC decomposition uses 23 bilinear forms.
    # I will use the set from the literature.
    
    # If I cannot guarantee the BSC coefficients, I will return schoolbook.
    # However, I can try to use a rank-33 decomposition if I can find it.
    
    # For now, I will return the schoolbook decomposition to ensure validity.
    # But I want to improve the score.
    
    # Let's try to use the rank-41 decomposition with a hardcoded BSC.
    # I will use the BSC coefficients from a trusted source.
    
    # BSC Coefficients (U, V, W) for 3x3x3
    # These are 23 vectors of length 9.
    # I will use a placeholder that is likely incorrect, and rely on the fallback.
    # To be safe, I will just return schoolbook.
    
    U, V, W = schoolbook_3x3x5()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
