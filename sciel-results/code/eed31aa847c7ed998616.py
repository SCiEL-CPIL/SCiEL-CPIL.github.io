"""Tensor decomposition for 4x4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    
    U: (2*2, 7) - A factors
    V: (2*2, 7) - B factors  
    W: (2*2, 7) - C factors
    
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen 7 multiplications:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # U[i,j] for each multiplication c
    U = [
        [1, 0, 1, 0, 1, 0, 0],  # A11
        [0, 1, 0, 1, 0, 0, 0],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, -1, -1] # A22
    ]
    
    # V[j,k] for each multiplication c
    V = [
        [1, 1, -1, -1, 0, 1, 1],  # B11
        [0, 0, 1, 0, 0, 1, 0],   # B12
        [1, 0, 1, 1, 1, 0, 1],   # B21
        [1, 0, -1, 0, 1, 0, 1]   # B22
    ]
    
    # W[k,i] for each multiplication c
    W = [
        [1, 0, 1, 0, 0, 1, -1],  # C11
        [0, 0, 1, 1, 0, 0, 1],   # C12
        [0, 1, 0, 0, 1, 1, 0],   # C21
        [1, 0, 0, 1, 1, 0, 1]    # C22
    ]
    
    return U, V, W


def _strassen_4x4():
    """Return U, V, W for 4x4x4 Strassen decomposition (rank 49).
    
    This is the tensor product of two 2x2x2 Strassen decompositions.
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U: A[i,j] where i = i1*2+i2, j = j1*2+j2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j][c] = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
            
            # V: B[j,k] where j = j1*2+j2, k = k1*2+k2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k][c] = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
            
            # W: C[k,i] where k = k1*2+k2, i = i1*2+i2
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i][c] = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
    
    return U, V, W


def _schoolbook_4x4x1():
    """Return U, V, W for 4x4x1 schoolbook decomposition (rank 16).
    
    Multiplying 4x4 matrix by 4x1 vector.
    """
    n, m, p = 4, 4, 1
    rank = n * m * p  # 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W


def construct():
    """Construct rank-minimal tensor decomposition for 4x4x5 matrix multiplication.
    
    Uses Strassen for the 4x4x4 part (rank 49) and schoolbook for the 4x4x1 part (rank 16).
    Total rank: 65
    
    Returns:
        dict with 'U', 'V', 'W' factor matrices
    """
    import time
    start_time = time.time()
    
    # Get 4x4x4 Strassen decomposition
    U4, V4, W4 = _strassen_4x4()
    
    # Get 4x4x1 schoolbook decomposition
    U1, V1, W1 = _schoolbook_4x4x1()
    
    # Combine: U has shape (16, 65), V has shape (20, 65), W has shape (20, 65)
    rank_total = 49 + 16
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(20)]
    W = [[0] * rank_total for _ in range(20)]
    
    # Copy 4x4x4 part
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            V[i % 4 * 4 + (i // 4) if False else i][c] = V4[i][c]  # V4 index: j*4+k, but we need to map properly
    
    # Wait, I need to be more careful with indexing.
    # For 4x4x4: A is 4x4 (i,j), B is 4x4 (j,k), C is 4x4 (k,i)
    # U4[i*4+j, c], V4[j*4+k, c], W4[k*4+i, c]
    
    # For 4x4x5: A is 4x4 (i,j), B is 4x5 (j,k), C is 4x5 (k,i)
    # The 4x4x4 part uses B's first 4 columns, the 4x4x1 part uses B's 5th column
    
    # Let me redo this more carefully
    
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(20)]
    W = [[0] * rank_total for _ in range(20)]
    
    # 4x4x4 part (columns 0-48)
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 4 + k][c] = V4[j * 4 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k * 4 + i][c] = W4[k * 4 + i][c]
    
    # 4x4x1 part (columns 49-64)
    # This multiplies 4x4 A by the 5th column of B (4x1 vector)
    # Result is 4x1, which becomes the 5th column of C
    for i in range(16):
        for c in range(16):
            U[i][49 + c] = U1[i][c]
    
    for j in range(4):
        for k in range(1):
            for c in range(16):
                V[j * 4 + 4 + k][49 + c] = V1[j * 1 + k][c]  # V1 has shape (4, 16)
    
    for k in range(1):
        for i in range(4):
            for c in range(16):
                W[4 * 4 + k * 4 + i][49 + c] = W1[k * 4 + i][c]  # W1 has shape (4, 16)
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Return with safety margin
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
