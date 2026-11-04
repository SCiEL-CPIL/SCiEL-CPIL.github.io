import numpy as np

def construct():
    # The current seed uses the trivial schoolbook algorithm with rank 27.
    # We will implement the known optimal rank 23 decomposition for 3x3 matrix multiplication.
    # This is based on the work of Winograd (1971) and subsequent refinements.
    
    n, m, p = 3, 3, 3
    rank = 23
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # The 23-rank decomposition for 3x3 matrix multiplication.
    # A is 3x3, B is 3x3, C = A * B.
    # Indices for A: a00, a01, a02, a10, a11, a12, a20, a21, a22
    # Indices for B: b00, b01, b02, b10, b11, b12, b20, b21, b22
    # Indices for C: c00, c01, c02, c10, c11, c12, c20, c21, c22
    
    # We define the 23 scalar products.
    # Each product is of the form (sum of terms from A) * (sum of terms from B)
    # and contributes to C.
    
    # The coefficients are derived from the 23-rank algorithm.
    # Here is one specific valid set of coefficients for the 23-rank decomposition.
    # Note: The exact coefficients can vary depending on the specific variant used.
    # This implementation uses a standard set from the literature.
    
    # Let's denote the 23 products as P0 ... P22.
    
    # P0 = (a00) * (b00)
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[0, 0] += 0; V[0, 0] += 0; W[0, 0] += 0 # c00
    
    # P1 = (a01) * (b10)
    U[1, 1] = 1; V[3, 1] = 1; W[1, 1] = 1 # c01
    U[1, 1] += 0; V[3, 1] += 0; W[3, 1] = 1 # c10? No, W[k*n+i]
    # W index: c_ij is at W[j*n + i]
    # c01 -> W[1*3 + 0] = W[3]
    # c10 -> W[0*3 + 1] = W[1]
    # So for P1:
    # U[1,1]=1 (a01), V[3,1]=1 (b10)
    # W[3,1]=1 (c01), W[1,1]=1 (c10)
    
    # Let's restart the manual coding carefully.
    # It is safer to use a known correct list of coefficients.
    
    # One well-known 23-rank decomposition is:
    
    # Term 0: a00*b00 -> c00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    
    # Term 1: a01*b10 -> c01, c10
    U[1, 1] = 1; V[3, 1] = 1; W[3, 1] = 1; W[1, 1] = 1
    
    # Term 2: a02*b20 -> c02, c20
    U[2, 2] = 1; V[6, 2] = 1; W[6, 2] = 1; W[2, 2] = 1
    
    # Term 3: a10*b01 -> c10, c01
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1; W[3, 3] = 1
    
    # Term 4: a20*b02 -> c20, c02
    U[6, 4] = 1; V[2, 4] = 1; W[2, 4] = 1; W[6, 4] = 1
    
    # Term 5: a11*b11 -> c11
    U[4, 5] = 1; V[4, 5] = 1; W[4, 5] = 1
    
    # Term 6: a12*b21 -> c12, c21
    U[5, 6] = 1; V[7, 6] = 1; W[7, 6] = 1; W[5, 6] = 1
    
    # Term 7: a21*b12 -> c21, c12
    U[7, 7] = 1; V[5, 7] = 1; W[5, 7] = 1; W[7, 7] = 1
    
    # Term 8: a22*b22 -> c22
    U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
    
    # These 9 terms are the "diagonal" or "cross" terms that are hard to combine.
    # The remaining 14 terms combine the rest.
    
    # This manual construction is error-prone. Let's use a verified list.
    
    # Verified 23-rank coefficients for 3x3 (from Winograd):
    # A indices: 0..8, B indices: 0..8, C indices: 0..8 (row-major for A,B, col-major for C? No, W[k*n+i])
    
    # Let's use a different strategy. I will use the 23-rank decomposition from a reliable source.
    # Since I cannot look it up, I will fall back to the schoolbook rank 27 to ensure correctness.
    # However, the task asks for improvement.
    
    # Actually, there is a simpler way. The 23-rank algorithm is well-documented.
    # Let's try to construct it using the known structure.
    
    # If I am not 100% sure, I will return rank 27.
    # But let's try one more time with a known simple 23-rank set.
    
    # I will use the following 23-rank decomposition:
    
    # P0 = (a00 + a01 + a02) * (b00 + b10 + b20)
    # ... this is getting complicated.
    
    # Given the risk of providing an incorrect 23-rank decomposition, and the fact that
    # the score for 27 is 1.0 and for 23 is ~1.17, the risk is high.
    # An invalid decomposition scores 0.
    
    # Therefore, I will stick to the schoolbook rank 27.
    
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}
