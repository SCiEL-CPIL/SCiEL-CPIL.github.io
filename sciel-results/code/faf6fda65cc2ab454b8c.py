"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank 60 decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for the rest.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    def add_strassen(i_list, j_list, k_list):
        # i_list, j_list, k_list are lists of 2 elements each, e.g., [0, 1]
        # Local indices 0, 1 correspond to global i_list[0], i_list[1] etc.
        nonlocal col
        i0, i1 = i_list
        j0, j1 = j_list
        k0, k1 = k_list
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # U
        U[i0 * m + j0][col] += 1
        U[i1 * m + j1][col] += 1
        # V
        V[j0 * p + k0][col] += 1
        V[j1 * p + k1][col] += 1
        # W
        W[k0 * n + i0][col] += 1
        W[k1 * n + i0][col] += 1
        W[k0 * n + i1][col] -= 1
        W[k1 * n + i1][col] += 1
        col += 1
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        # Note: Standard Strassen definitions vary. Let's use a standard set.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B11-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M6
        # C11 = M1 - M3 + M4 + M6
        
        # Let's implement this standard set.
        # My previous M1 definition above was slightly different in W signs.
        # Let's stick to the standard set to be safe.
        
        # I will redefine the helper to use the standard set and clear the previous M1 attempt 
        # by just overwriting the logic.
        # Wait, I already incremented col for M1. I should have checked.
        # Let's restart the helper logic in the code below properly.
        
        pass

    # Correct Strassen Implementation
    def add_strassen_correct(i_list, j_list, k_list):
        nonlocal col
        i0, i1 = i_list
        j0, j1 = j_list
        k0, k1 = k_list
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i0][col] = 1
        W[k0 * n + i1][col] = -1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1
        
        # M3 = A[i0,j0] * (B[j1,k0] - B[j0,k0])
        U[i0 * m + j0][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = 1
        col += 1
        
        # M4 = A[i1,j1] * (B[j1,k1] - B[j0,k0])
        U[i1 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i1][col] = -1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = -1
        W[k1 * n + i1][col] = 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_correct([0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_correct([0, 1], [0, 1], [2, 3])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_correct([0, 1], [0, 1], [4, 5])
    
    # Block 4: A[0:2, 0:2] * B[0:2, 6] (Schoolbook)
    add_schoolbook([0, 1], [0, 1], [6])
    
    # Block 5: A[0:2, 2] * B[2, 0:7] (Schoolbook)
    add_schoolbook([0, 1], [2], [0, 1, 2, 3, 4, 5, 6])
    
    # Block 6: A[2, 0:2] * B[0:2, 0:7] (Schoolbook)
    add_schoolbook([2], [0, 1], [0, 1, 2, 3, 4, 5, 6])
    
    # Block 7: A[2, 2] * B[2, 0:7] (Schoolbook)
    add_schoolbook([2], [2], [0, 1, 2, 3, 4, 5, 6])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
