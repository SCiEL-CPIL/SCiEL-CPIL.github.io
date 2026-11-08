"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Lader 3x3x3x3 rank-23 algorithm
    # Returns U, V, W for 3x3 * 3x3 with rank 23
    def lader_3x3():
        # The 23 rank-1 terms for 3x3x3x3
        # Each term is (u_vec, v_vec, w_vec) of length 9 each
        # U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
        
        # Lader's 23-rank algorithm for 3x3x3x3
        # Coefficients from Lader (1977)
        
        # Define the 23 terms
        # Each term t: U[:,t], V[:,t], W[:,t]
        # Such that sum_t U[:,t] * V[:,t] * W[:,t] = multiplication tensor
        
        # I'll use the known Lader coefficients
        # The algorithm computes C = A*B where A,B are 3x3
        
        # Let me define using the standard Lader construction
        # This is based on the 23-multiplication algorithm
        
        # U indices: i*m+j for i,j in 0..2 (row i, col j of A)
        # V indices: j*p+k for j,k in 0..2 (row j, col k of B) 
        # W indices: k*n+i for k,i in 0..2 (row k, col i of C)
        
        U = [[0]*23 for _ in range(9)]
        V = [[0]*23 for _ in range(9)]
        W = [[0]*23 for _ in range(9)]
        
        # Lader's algorithm - 23 rank-1 tensors
        # I'll encode the known solution
        
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        
        # Term 1: A[0,1]*B[1,0] -> C[0,0]
        U[3][1] = 1; V[3][1] = 1; W[0][1] = 1
        
        # Term 2: A[0,2]*B[2,0] -> C[0,0]
        U[6][2] = 1; V[6][2] = 1; W[0][2] = 1
        
        # Term 3: A[1,0]*B[0,1] -> C[1,0]
        U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
        
        # Term 4: A[1,1]*B[1,1] -> C[1,1]
        U[4][4] = 1; V[4][4] = 1; W[4][4] = 1
        
        # Term 5: A[1,2]*B[2,1] -> C[1,1]
        U[7][5] = 1; V[7][5] = 1; W[4][5] = 1
        
        # Term 6: A[2,0]*B[0,2] -> C[2,0]
        U[2][6] = 1; V[2][6] = 1; W[6][6] = 1
        
        # Term 7: A[2,1]*B[1,2] -> C[2,1]
        U[5][7] = 1; V[5][7] = 1; W[7][7] = 1
        
        # Term 8: A[2,2]*B[2,2] -> C[2,2]
        U[8][8] = 1; V[8][8] = 1; W[8][8] = 1
        
        # This is getting complicated. Let me use a verified Lader implementation.
        
        # Actually, let me use a simpler but still better approach.
        # Use Strassen 2x2x2x2 (rank 7) applied to overlapping 2x2 blocks of 3x3.
        
        # For 3x3 x 3x3, I can use 4 overlapping 2x2 blocks:
        # Block 1: rows 0-1, cols 0-1 of A; rows 0-1, cols 0-1 of B
        # Block 2: rows 0-1, cols 1-2 of A; rows 0-1, cols 1-2 of B  
        # Block 3: rows 1-2, cols 0-1 of A; rows 1-2, cols 0-1 of B
        # Block 4: rows 1-2, cols 1-2 of A; rows 1-2, cols 1-2 of B
        
        # Each 2x2x2x2 uses rank 7, total 28. But with careful combination, 
        # we can reduce. However, this still gives rank 28 > 23.
        
        # Let me just use rank 27: 3 Strassen 2x2x2x2 computations
        # Split 3x3 into:
        # - Use rows 0,1 and cols 0,1 (2x2)
        # - Use rows 1,2 and cols 0,1 (2x2) 
        # - Use rows 0,1 and cols 1,2 (2x2)
        # - Use rows 1,2 and cols 1,2 (2x2)
        # That's 4 blocks x 7 = 28
        
        # Alternatively, just use schoolbook for 3x3x3x3: rank 9
        # And for 3x3x3x6: rank 18
        # Total: 2 * 18 = 36? No, schoolbook for 3x3x3x3 is 9, for 3x3x3x6 is 18.
        # Two 3x3x3x3: 2*9 = 18. But that's just schoolbook!
        
        # OK let me just do schoolbook for the full 3x3x3x6: rank 54
        # Or two 3x3x3x3 schoolbook: rank 18
        
        # Wait, schoolbook for 3x3 * 3x3 is 3*3*3 = 27 scalar mults, rank 27.
        # For 3x3 * 3x6 it's 3*3*6 = 54.
        # Two separate 3x3*3x3: 2*27 = 54. Same.
        
        # Let me try: compute A*B where B is 3x6.
        # Split B into B1 (3x3) and B2 (3x3).
        # C1 = A*B1 (3x3), C2 = A*B2 (3x3)
        # Use Strassen for each: rank 7 each, total 14? No, Strassen is for 2x2.
        
        # I'll just return schoolbook rank 54 for now, which is the baseline.
        
        U = [[0]*27 for _ in range(9)]
        V = [[0]*27 for _ in range(9)]
        W = [[0]*27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W
    
    # For 3x3 * 3x6, use two 3x3*3x3 computations
    # But schoolbook gives 54 either way.
    
    # Let me try a rank-36 approach using 3x3 Strassen-like
    # Actually, let me just implement the best I can.
    
    # Use the fact that we can compute 3x3 * 3x6 with rank 27 
    # by noting that each row of A (1x3) times B (3x6) gives 1x6
    # And 1x3 * 3x6 can be done with 3 multiplications per output column
    # 3 rows * 6 cols * 3 = 54. No improvement.
    
    # Let me try rank 45: use 5 Strassen 2x2x2x2 blocks
    # This is getting too complex without a clear algorithm.
    
    # Just return schoolbook rank 54
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
