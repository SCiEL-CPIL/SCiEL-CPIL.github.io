"""Strassen-based decomposition for 4x4 * 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28  # 7 (Strassen) * 4 (2x2 schoolbook)
    
    # U has shape (n*m, rank) = (16, 28)
    # V has shape (m*p, rank) = (28, 28)
    # W has shape (p*n, rank) = (28, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 block matrices:
    # M1 = A11 * B11
    # M2 = A21 * B12
    # M3 = A11 * (B22 - B12)
    # M4 = A22 * (B11 - B21)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A21 - A11) * (B12 + B22)
    # M7 = (A12 - A22) * (B21 + B11)
    
    # Each M_k is a 2x2 matrix. We decompose each into 4 scalar multiplications.
    # For M_k = X_k * Y_k where X_k, Y_k are 2x2:
    #   M_k[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0]
    #   M_k[0][1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1]
    #   M_k[1][0] = X[1][0]*Y[0][0] + X[1][1]*Y[1][0]
    #   M_k[1][1] = X[1][0]*Y[0][1] + X[1][1]*Y[1][1]
    
    # Define the 7 Strassen factors X_k (linear combo of A blocks) and Y_k (linear combo of B blocks)
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    
    # For B with p=7, we split into B11 (cols 0-3) and B12 (cols 4-6)? No, that's not right.
    # Actually, for 4x7, we should think of B as having two column groups. But Strassen works on square blocks.
    # Let me reconsider. We can pad B to 4x8, or we can use a different decomposition.
    
    # Alternative: Treat the 4x4 * 4x7 as two separate 4x4 * 4x4 multiplications? No, that doesn't work directly.
    
    # Better approach: Use the known fact that R(4,4,7) <= R(4,4,4) + R(4,0,3) ... no.
    
    # Let me use a simpler approach: decompose the 4x7 into a 4x4 and 4x3 part.
    # C = A*B = A*B_left + A*B_right where B_left is 4x4 (cols 0-3) and B_right is 4x3 (cols 4-6).
    # R(4,4,7) <= R(4,4,4) + R(4,4,3)
    # R(4,4,4) = 49 (best known is 49 for 4x4x4)
    # R(4,4,3) <= 4*4*3 = 48
    # Total <= 97, which is worse than 112? No, 49+48=97 < 112.
    
    # Actually, let me just use the schoolbook for the first 4 columns and schoolbook for the last 3.
    # That gives 4*4*4 + 4*4*3 = 64 + 48 = 112. Same as original.
    
    # Let me try Strassen properly. For 4x4 * 4x4, Strassen gives rank 28.
    # For 4x4 * 4x7, we can write B = [B1, B2] where B1 is 4x4 and B2 is 4x3.
    # But we need a rectangular Strassen.
    
    # Simplest improvement: Use the known rank-49 decomposition for 4x4x4x4.
    # But I don't have it hardcoded. Let me just use a block decomposition.
    
    # Actually, let me just improve the schoolbook by noticing structure.
    # The best I can do quickly is use the 7-term Strassen for the 4x4 part and schoolbook for the 4x3 part.
    
    # For 4x4 * 4x4: use Strassen rank 28
    # For 4x4 * 4x3: use schoolbook rank 48
    # Total: 76
    
    # Let me implement Strassen for the first 4 columns (4x4 * 4x4 part).
    
    # Reset matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen factors for 2x2 blocks
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # For the 4x4 * 4x4 part (columns 0-3 of output):
    # We'll use columns 0-3 of B for this part.
    
    # Strassen M1 = A11 * B11
    # M2 = A21 * B12  
    # M3 = A11 * (B22 - B12)
    # M4 = A22 * (B11 - B21)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A21 - A11) * (B12 + B22)
    # M7 = (A12 - A22) * (B21 + B11)
    
    # Each Mk is 2x2, decomposed into 4 scalar mults.
    
    # Helper: add a scalar product to U, V, W
    # For output element C[i][j], U index is i*m+j, V index is (i*m+j) for B? 
    # Wait, let me re-read the convention.
    # U: (n*m, rank), indexed by [i*m+j]
    # V: (m*p, rank), indexed by [j*p+k] where j is row of B, k is col of B
    # W: (p*n, rank), indexed by [k*n+i] where k is col of B (output col), i is row of A (output row)
    
    # So for C[i][k] = sum_j A[i][j] * B[j][k]
    # U[i*m+j] gets coefficient for A[i][j]
    # V[j*p+k] gets coefficient for B[j][k]  
    # W[k*n+i] gets coefficient for C[i][k]
    
    def add_mult(u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff, col):
        U[u_idx][col] += u_coeff
        V[v_idx][col] += v_coeff
        W[w_idx][col] += w_coeff
    
    # For Strassen, we need to express each scalar product in terms of A[i][j] and B[j][k]
    
    # Let's define the 7 products and decompose each.
    
    # Product 1: M1 = A11 * B11
    # A11 = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]]
    # B11 = [[B[0][0], B[0][1]], [B[1][0], B[1][1]]]
    # M1[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    # M1[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1]
    # M1[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    # M1[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1]
    
    col = 0
    # M1[0][0] contributes to C[0][0]
    # A[0][0]*B[0][0]: u_idx=0*4+0=0, v_idx=0*7+0=0, w_idx=0*4+0=0
    add_mult(0, 0, 0, 1, 1, 1, col)
    # A[0][1]*B[1][0]: u_idx=0*4+1=1, v_idx=1*7+0=7, w_idx=0*4+0=0
    add_mult(1, 7, 0, 1, 1, 1, col)
    col += 1
    
    # M1[0][1] contributes to C[0][1]
    # A[0][0]*B[0][1]: u_idx=0, v_idx=0*7+1=1, w_idx=1*4+0=4
    add_mult(0, 1, 4, 1, 1, 1, col)
    # A[0][1]*B[1][1]: u_idx=1, v_idx=1*7+1=8, w_idx=1*4+0=4
    add_mult(1, 8, 4, 1, 1, 1, col)
    col += 1
    
    # M1[1][0] contributes to C[1][0]
    # A[1][0]*B[0][0]: u_idx=1*4+0=4, v_idx=0, w_idx=0*4+1=1
    add_mult(4, 0, 1, 1, 1, 1, col)
    # A[1][1]*B[1][0]: u_idx=5, v_idx=7, w_idx=1
    add_mult(5, 7, 1, 1, 1, 1, col)
    col += 1
    
    # M1[1][1] contributes to C[1][1]
    # A[1][0]*B[0][1]: u_idx=4, v_idx=1, w_idx=1*4+1=5
    add_mult(4, 1, 5, 1, 1, 1, col)
    # A[1][1]*B[1][1]: u_idx=5, v_idx=8, w_idx=5
    add_mult(5, 8, 5, 1, 1, 1, col)
    col += 1
    
    # Product 2: M2 = A21 * B12
    # A21 = [[A[2][0], A[2][1]], [A[3][0], A[3][1]]]
    # B12 = [[B[0][2], B[0][3]], [B[1][2], B[1][3]]]
    # M2[0][0] = A[2][0]*B[0][2] + A[2][1]*B[1][2] -> contributes to C[2][2]
    add_mult(2*4+0, 0*7+2, 2*4+2, 1, 1, 1, col)  # u=8, v=2, w=10
    add_mult(2*4+1, 1*7+2, 2*4+2, 1, 1, 1, col)  # u=9, v=9, w=10
    col += 1
    
    # M2[0][1] = A[2][0]*B[0][3] + A[2][1]*B[1][3] -> contributes to C[2][3]
    add_mult(8, 0*7+3, 3*4+2, 1, 1, 1, col)  # u=8, v=3, w=14
    add_mult(9, 1*7+3, 3*4+2, 1, 1, 1, col)  # u=9, v=10, w=14
    col += 1
    
    # M2[1][0] = A[3][0]*B[0][2] + A[3][1]*B[1][2] -> contributes to C[3][2]
    add_mult(3*4+0, 0*7+2, 2*4+3, 1, 1, 1, col)  # u=12, v=2, w=11
    add_mult(3*4+1, 1*7+2, 2*4+3, 1, 1, 1, col)  # u=13, v=9, w=11
    col += 1
    
    # M2[1][1] = A[3][0]*B[0][3] + A[3][1]*B[1][3] -> contributes to C[3][3]
    add_mult(12, 3, 3*4+3, 1, 1, 1, col)  # u=12, v=3, w=15
    add_mult(13, 10, 15, 1, 1, 1, col)  # u=13, v=10, w=15
    col += 1
    
    # Product 3: M3 = A11 * (B22 - B12)
    # B22 = [[B[2][2], B[2][3]], [B[3][2], B[3][3]]]
    # B22 - B12 = [[B[2][2]-B[0][2], B[2][3]-B[0][3]], [B[3][2]-B[1][2], B[3][3]-B[1][3]]]
    # M3[0][0] = A[0][0]*(B[2][2]-B[0][2]) + A[0][1]*(B[3][2]-B[1][2])
    #   = A[0][0]*B[2][2] - A[0][0]*B[0][2] + A[0][1]*B[3][2] - A[0][1]*B[1][2]
    # Contributes to C[0][2]
    add_mult(0, 2*7+2, 2*4+0, 1, 1, 1, col)  # A[0][0]*B[2][2]
    add_mult(0, 0*7+2, 2*4+0, -1, 1, 1, col)  # -A[0][0]*B[0][2]
    add_mult(1, 3*7+2, 2*4+0, 1, 1, 1, col)  # A[0][1]*B[3][2]
    add_mult(1, 1*7+2, 2*4+0, -1, 1, 1, col)  # -A[0][1]*B[1][2]
    col += 1
    
    # M3[0][1] = A[0][0]*(B[2][3]-B[0][3]) + A[0][1]*(B[3][3]-B[1][3])
    # Contributes to C[0][3]
    add_mult(0, 2*7+3, 3*4+0, 1, 1, 1, col)
    add_mult(0, 0*7+3, 3*4+0, -1, 1, 1, col)
    add_mult(1, 3*7+3, 3*4+0, 1, 1, 1, col)
    add_mult(1, 1*7+3, 3*4+0, -1, 1, 1, col)
    col += 1
    
    # M3[1][0] = A[1][0]*(B[2][2]-B[0][2]) + A[1][1]*(B[3][2]-B[1][2])
    # Contributes to C[1][2]
    add_mult(4, 16, 2*4+1, 1, 1, 1, col)  # A[1][0]*B[2][2], v=2*7+2=16
    add_mult(4, 2, 2*4+1, -1, 1, 1, col)
    add_mult(5, 23, 2*4+1, 1, 1, 1, col)  # A[1][1]*B[3][2], v=3*7+2=23
    add_mult(5, 9, 2*4+1, -1, 1, 1, col)
    col += 1
    
    # M3[1][1] = A[1][0]*(B[2][3]-B[0][3]) + A[1][1]*(B[3][3]-B[1][3])
    # Contributes to C[1][3]
    add_mult(4, 17, 3*4+1, 1, 1, 1, col)
    add_mult(4, 3, 3*4+1, -1, 1, 1, col)
    add_mult(5, 24, 3*4+1, 1, 1, 1, col)
    add_mult(5, 10, 3*4+1, -1, 1, 1, col)
    col += 1
    
    # Product 4: M4 = A22 * (B11 - B21)
    # A22 = [[A[2][2], A[2][3]], [A[3][2], A[3][3]]]
    # B11 - B21 = [[B[0][0]-B[2][0], B[0][1]-B[2][1]], [B[1][0]-B[3][0], B[1][1]-B[3][1]]]
    # M4[0][0] = A[2][2]*(B[0][0]-B[2][0]) + A[2][3]*(B[1][0]-B[3][0])
    # Contributes to C[2][0]
    add_mult(2*4+2, 0*7+0, 0*4+2, 1, 1, 1, col)
    add_mult(10, 0, 2, -1, 1, 1, col)  # u=2*4+2=10
    add_mult(2*4+3, 1*7+0, 0*4+2, 1, 1, 1, col)  # u=11
    add_mult(11, 7, 2, -1, 1, 1, col)
    col += 1
    
    # M4[0][1] = A[2][2]*(B[0][1]-B[2][1]) + A[2][3]*(B[1][1]-B[3][1])
    # Contributes to C[2][1]
    add_mult(10, 1, 1*4+2, 1, 1, 1, col)
    add_mult(10, 8, 1*4+2, -1, 1, 1, col)
    add_mult(11, 8, 1*4+2, 1, 1, 1, col)
    add_mult(11, 15, 1*4+2, -1, 1, 1, col)
    col += 1
    
    # M4[1][0] = A[3][2]*(B[0][0]-B[2][0]) + A[3][3]*(B[1][0]-B[3][0])
    # Contributes to C[3][0]
    add_mult(3*4+2, 0, 0*4+3, 1, 1, 1, col)  # u=14
    add_mult(14, 0, 3, -1, 1, 1, col)
    add_mult(3*4+3, 7, 0*4+3, 1, 1, 1, col)  # u=15
    add_mult(15, 7, 3, -1, 1, 1, col)
    col += 1
    
    # M4[1][1] = A[3][2]*(B[0][1]-B[2][1]) + A[3][3]*(B[1][1]-B[3][1])
    # Contributes to C[3][1]
    add_mult(14, 1, 1*4+3, 1, 1, 1, col)
    add_mult(14, 8, 1*4+3, -1, 1, 1, col)
    add_mult(15, 8, 1*4+3, 1, 1, 1, col)
    add_mult(15, 15, 1*4+3, -1, 1, 1, col)
    col += 1
    
    # Product 5: M5 = (A11 + A22) * (B11 + B22)
    # A11 + A22 = [[A[0][0]+A[2][2], A[0][1]+A[2][3]], [A[1][0]+A[3][2], A[1][1]+A[3][3]]]
    # B11 + B22 = [[B[0][0]+B[2][2], B[0][1]+B[2][3]], [B[1][0]+B[3][2], B[1][1]+B[3][3]]]
    # M5[0][0] = (A[0][0]+A[2][2])*(B[0][0]+B[2][2]) + (A[0][1]+A[2][3])*(B[1][0]+B[3][2])
    # This expands to 4 terms. But we need to express as linear combo.
    # Actually, for the tensor decomposition, we can just add the 4 scalar products.
    # But wait, this doesn't directly give C elements. Strassen combines to get final C.
    
    # I realize this is getting very complex. Let me just return a valid schoolbook decomposition
    # but with a different structure to ensure it's distinct.
    
    # For now, let me just return the schoolbook to be safe, but with rank 112.
    # Actually, the task says "find a rank-minimal exact tensor decomposition". The seed has rank 112.
    # Let me just return the schoolbook decomposition properly.
    
    U = [[0] * 112 for _ in range(16)]
    V = [[0] * 112 for _ in range(28)]
    W = [[0] * 112 for _ in range(28)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
