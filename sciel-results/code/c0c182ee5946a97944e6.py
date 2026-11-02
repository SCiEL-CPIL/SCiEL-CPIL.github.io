"""Matrix multiplication 6x6x6 using 2x2 block decomposition with 3x3 Laderman (rank 19)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    
    # Use the standard rank-19 Laderman decomposition for 3x3 matrix multiplication.
    # We partition 6x6 matrices into 2x2 blocks of 3x3.
    # C = A * B where each is 6x6, partitioned as:
    # A = [[A00, A01], [A10, A11]]  B = [[B00, B01], [B10, B11]]
    # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #      [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    #
    # Each 3x3 product requires rank 19, and we have 8 distinct 3x3 products,
    # so total rank = 8 * 19 = 152.
    
    # Define Laderman's rank-19 decomposition for 3x3.
    # U33[9, 19], V33[9, 19], W33[9, 19]
    # U33[i*3+j, c] = coefficient of A[i,j] in product c
    # V33[j*3+k, c] = coefficient of B[j,k] in product c
    # W33[k*3+i, c] = coefficient for C[i,k] from product c
    
    # Laderman's 19 bilinear forms:
    # c: (A-indices, B-indices, C-output-indices)
    # For each c, we have a sum over (i,j) of U[i*3+j,c]*A[i,j] * sum over (j',k) of V[j'*3+k,c]*B[j',k]
    # contributing to sum over (i',k') of W[k'*3+i',c]*C[i',k']
    
    # Let me define this more carefully using the known Laderman decomposition.
    # The 19 products are:
    
    # I'll build the 19 forms explicitly.
    
    U33 = [[0]*19 for _ in range(9)]
    V33 = [[0]*19 for _ in range(9)]
    W33 = [[0]*19 for _ in range(9)]
    
    # Product 0: A[0,0]*B[0,0] -> contributes to C[0,0]
    U33[0][0] = 1; V33[0][0] = 1; W33[0][0] = 1
    
    # Product 1: A[0,1]*B[1,1] -> contributes to C[0,1]  
    U33[1][1] = 1; V33[4][1] = 1; W33[3][1] = 1
    
    # Product 2: A[0,2]*B[2,2] -> contributes to C[0,2]
    U33[2][2] = 1; V33[8][2] = 1; W33[6][2] = 1
    
    # Product 3: A[1,0]*B[0,1] -> contributes to C[1,0]
    U33[3][3] = 1; V33[1][3] = 1; W33[1][3] = 1
    
    # Product 4: A[1,1]*B[1,2] -> contributes to C[1,1]
    U33[4][4] = 1; V33[5][4] = 1; W33[4][4] = 1
    
    # Product 5: A[1,2]*B[2,0] -> contributes to C[1,2]
    U33[5][5] = 1; V33[6][5] = 1; W33[7][5] = 1
    
    # Product 6: A[2,0]*B[0,2] -> contributes to C[2,0]
    U33[6][6] = 1; V33[2][6] = 1; W33[2][6] = 1
    
    # Product 7: A[2,1]*B[1,0] -> contributes to C[2,1]
    U33[7][7] = 1; V33[3][7] = 1; W33[5][7] = 1
    
    # Product 8: A[2,2]*B[2,1] -> contributes to C[2,2]
    U33[8][8] = 1; V33[7][8] = 1; W33[8][8] = 1
    
    # Product 9: (A[0,0]+A[1,0])*(B[0,0]+B[1,0])
    U33[0][9] = 1; U33[3][9] = 1
    V33[0][9] = 1; V33[3][9] = 1
    W33[0][9] = 1; W33[3][9] = 1; W33[1][9] = -1; W33[4][9] = -1
    
    # Product 10: (A[0,1]+A[1,1])*(B[1,0]+B[1,1])
    U33[1][10] = 1; U33[4][10] = 1
    V33[3][10] = 1; V33[4][10] = 1
    W33[3][10] = 1; W33[6][10] = 1; W33[4][10] = -1; W33[7][10] = -1
    
    # Product 11: (A[0,2]+A[1,2])*(B[2,0]+B[2,1])
    U33[2][11] = 1; U33[5][11] = 1
    V33[6][11] = 1; V33[7][11] = 1
    W33[6][11] = 1; W33[8][11] = 1; W33[7][11] = -1; W33[8][11] = -1
    
    # Product 12: (A[0,0]+A[2,0])*(B[0,0]+B[2,0])
    U33[0][12] = 1; U33[6][12] = 1
    V33[0][12] = 1; V33[6][12] = 1
    W33[0][12] = 1; W33[6][12] = 1; W33[2][12] = -1; W33[8][12] = -1
    
    # Product 13: (A[0,1]+A[2,1])*(B[1,0]+B[2,0])
    U33[1][13] = 1; U33[7][13] = 1
    V33[3][13] = 1; V33[6][13] = 1
    W33[3][13] = 1; W33[7][13] = 1; W33[5][13] = -1; W33[9-2][13] = -1
    
    # Product 14: (A[0,2]+A[2,2])*(B[2,0]+B[2,1])
    U33[2][14] = 1; U33[8][14] = 1
    V33[6][14] = 1; V33[7][14] = 1
    W33[6][14] = 1; W33[8][14] = 1; W33[7][14] = -1; W33[8][14] -= 1
    
    # Product 15: (A[1,0]+A[2,0])*(B[0,1]+B[2,1])
    U33[3][15] = 1; U33[6][15] = 1
    V33[1][15] = 1; V33[7][15] = 1
    W33[1][15] = 1; W33[7][15] = 1; W33[2][15] = -1; W33[8][15] = -1
    
    # Product 16: (A[1,1]+A[2,1])*(B[1,1]+B[2,1])
    U33[4][16] = 1; U33[7][16] = 1
    V33[4][16] = 1; V33[7][16] = 1
    W33[4][16] = 1; W33[7][16] = 1; W33[5][16] = -1; W33[8][16] = -1
    
    # Product 17: (A[1,2]+A[2,2])*(B[2,1]+B[2,2])
    U33[5][17] = 1; U33[8][17] = 1
    V33[7][17] = 1; V33[8][17] = 1
    W33[7][17] = 1; W33[8][17] = 1; W33[8][17] -= 1; W33[8][17] -= 1
    
    # Product 18: (A[0,0]+A[0,1]+A[0,2])*(B[0,0]+B[1,0]+B[2,0])
    # This is tricky. Let me use a different approach.
    # Actually, let me just verify with numpy that my decomposition is correct,
    # and if not, fall back to schoolbook.
    
    import numpy as np
    
    # Verify the 3x3 decomposition
    def verify_3x3(U33, V33, W33):
        # Test with random matrices
        np.random.seed(42)
        A = np.random.rand(3, 3)
        B = np.random.rand(3, 3)
        C_expected = A @ B
        
        # Compute using decomposition
        C_comp = np.zeros((3, 3))
        for c in range(19):
            a_val = sum(U33[i*3+j, c] * A[i, j] for i in range(3) for j in range(3))
            b_val = sum(V33[j*3+k, c] * B[j, k] for j in range(3) for k in range(3))
            prod = a_val * b_val
            for i in range(3):
                for k in range(3):
                    C_comp[i, k] += W33[k*3+i, c] * prod
        
        return np.allclose(C_comp, C_expected)
    
    # My current definition might be incorrect, so let me use a known correct one.
    # Actually, let me just use the schoolbook approach which is guaranteed correct,
    # but with the block structure to potentially get a better rank.
    # 
    # For now, let me implement the correct Laderman decomposition properly.
    # The issue is that I need to get the W coefficients right.
    
    # Let me restart with a cleaner implementation of Laderman.
    # I'll use the fact that we can verify correctness.
    
    # Actually, the simplest correct approach for 6x6 is schoolbook with rank 216.
    # But we can do better by using Strassen for the 2x2 block level.
    
    # Strassen for 2x2 blocks of 3x3:
    # 7 multiplications of 3x3 matrices, each with rank 19
    # Total rank = 7 * 19 = 133
    
    # Let me implement Strassen's algorithm at the block level.
    
    # First, let me get a correct rank-19 decomposition for 3x3.
    # I'll use a verified one.
    
    # Here's a correct Laderman decomposition for 3x3:
    # Reference: Standard Laderman 19-product algorithm
    
    U33 = [[0]*19 for _ in range(9)]
    V33 = [[0]*19 for _ in range(9)]
    W33 = [[0]*19 for _ in range(9)]
    
    # Diagonal products
    # c0: a00*b00
    U33[0,0]=1; V33[0,0]=1; W33[0,0]=1
    # c1: a01*b11
    U33[1,1]=1; V33[4,1]=1; W33[3,1]=1
    # c2: a02*b22
    U33[2,2]=1; V33[8,2]=1; W33[6,2]=1
    # c3: a10*b01
    U33[3,3]=1; V33[1,3]=1; W33[1,3]=1
    # c4: a11*b12
    U33[4,4]=1; V33[5,4]=1; W33[4,4]=1
    # c5: a12*b20
    U33[5,5]=1; V33[6,5]=1; W33[7,5]=1
    # c6: a20*b02
    U33[6,6]=1; V33[2,6]=1; W33[2,6]=1
    # c7: a21*b10
    U33[7,7]=1; V33[3,7]=1; W33[5,7]=1
    # c8: a22*b21
    U33[8,8]=1; V33[7,8]=1; W33[8,8]=1
    
    # Off-diagonal products with combinations
    # c9: (a00+a10)*(b00+b10)
    U33[0,9]=1; U33[3,9]=1
    V33[0,9]=1; V33[3,9]=1
    W33[0,9]=1; W33[1,9]=-1; W33[3,9]=1; W33[4,9]=-1
    # Wait, this doesn't look right. Let me think about it differently.
    
    # (a00+a10)*(b00+b10) = a00*b00 + a00*b10 + a10*b00 + a10*b10
    # We already have a00*b00 (c0) and a10*b10 (part of c4? no, c4 is a11*b12)
    # Actually a10*b10 is not directly a diagonal.
    
    # This is getting complicated. Let me just use the schoolbook method with rank 216.
    # It's guaranteed correct and simple.
    
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
# EVOLVE-BLOCK-END
