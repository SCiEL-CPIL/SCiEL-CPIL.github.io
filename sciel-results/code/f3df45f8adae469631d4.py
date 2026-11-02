"""Matrix multiplication 2,2,10 using Strassen-based decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    
    # We partition the 2x10 matrix into five 2x2 blocks.
    # Let A be 2x2, B be 2x10 = [B1 B2 B3 B4 B5] where each Bj is 2x2.
    # C = A * B = [A*B1, A*B2, A*B3, A*B4, A*B5]
    
    # For each 2x2 block multiplication, we use Strassen's algorithm (rank 7).
    # Strassen's algorithm for 2x2 matrices:
    # Given P = [[p11, p12], [p21, p22]] and Q = [[q11, q12], [q21, q22]]
    # R = P * Q
    # m1 = (p11 + p22) * (q11 + q22)
    # m2 = (p21 + p22) * q11
    # m3 = p11 * (q12 - q22)
    # m4 = p22 * (q21 - q11)
    # m5 = (p11 + p12) * q22
    # m6 = (p21 - p11) * (q11 + q12)
    # m7 = (p12 - p22) * (q21 + q22)
    # R11 = m1 + m4 - m5 + m7
    # R12 = m3 + m5
    # R21 = m2 + m4
    # R22 = m1 - m2 + m3 + m6
    
    # The rank-7 Strassen decomposition uses 7 multiplications.
    # U factors (for A's elements): shape (2*2, 7) = (4, 7)
    # V factors (for B's elements): shape (2*2, 7) = (4, 7)
    # W factors (for R's elements): shape (2*2, 7) = (4, 7)
    
    # For A = [[a11, a12], [a21, a22]], indexed as a[0]=a11, a[1]=a12, a[2]=a21, a[3]=a22
    # U matrix: U[row, col] where row is index in A (0..3), col is the multiplication index (0..6)
    # 
    # m1 uses (a11 + a22) -> U[0,0]=1, U[3,0]=1
    # m2 uses (a21 + a22) -> U[2,1]=1, U[3,1]=1
    # m3 uses a11 -> U[0,2]=1
    # m4 uses a22 -> U[3,3]=1
    # m5 uses (a11 + a12) -> U[0,4]=1, U[1,4]=1
    # m6 uses (a21 - a11) -> U[2,5]=1, U[0,5]=-1
    # m7 uses (a12 - a22) -> U[1,6]=1, U[3,6]=-1
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],   # a11
        [0, 0, 0, 0, 1, 0, 1],    # a12
        [0, 1, 0, 0, 0, 1, 0],    # a21
        [1, 1, 0, 1, 0, 0, -1]    # a22
    ]
    
    # For B_j = [[b11, b12], [b21, b22]], indexed as b[0]=b11, b[1]=b12, b[2]=b21, b[3]=b22
    # V matrix: V[row, col] where row is index in B_j (0..3), col is the multiplication index (0..6)
    #
    # m1 uses (b11 + b22) -> V[0,0]=1, V[3,0]=1
    # m2 uses b11 -> V[0,1]=1
    # m3 uses (b12 - b22) -> V[1,2]=1, V[3,2]=-1
    # m4 uses (b21 - b11) -> V[2,3]=1, V[0,3]=-1
    # m5 uses b22 -> V[3,4]=1
    # m6 uses (b11 + b12) -> V[0,5]=1, V[1,5]=1
    # m7 uses (b21 + b22) -> V[2,6]=1, V[3,6]=1
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],   # b11
        [0, 0, 1, 0, 0, 1, 0],    # b12
        [0, 0, 0, 1, 0, 0, 1],    # b21
        [1, 0, -1, 0, 1, 0, 1]    # b22
    ]
    
    # For R_j = [[r11, r12], [r21, r22]], indexed as r[0]=r11, r[1]=r12, r[2]=r21, r[3]=r22
    # W matrix: W[row, col] where row is index in R_j (0..3), col is the multiplication index (0..6)
    #
    # R11 = m1 + m4 - m5 + m7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    # R12 = m3 + m5 -> W[1,2]=1, W[1,4]=1
    # R21 = m2 + m4 -> W[2,1]=1, W[2,3]=1
    # R22 = m1 - m2 + m3 + m6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],   # r11
        [0, 0, 1, 0, 1, 0, 0],    # r12
        [0, 1, 0, 1, 0, 0, 0],    # r21
        [1, -1, 1, 0, 0, 1, 0]    # r22
    ]
    
    # Now we need to construct the full U, V, W for the 2x2 * 2x10 multiplication.
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (20, rank)
    # W has shape (p*n, rank) = (20, rank)
    
    # The total rank is 5 * 7 = 35 (one Strassen decomposition per block)
    rank = 5 * 7
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(20)]
    
    # For each block j (0 to 4), we use a separate set of 7 multiplications.
    # Block j corresponds to columns 2j and 2j+1 of B, and columns 2j and 2j+1 of C.
    
    for j in range(5):
        base = j * 7
        for mult_idx in range(7):
            col = base + mult_idx
            # U: A's elements contribute to this multiplication
            for a_idx in range(4):
                U[a_idx][col] = U_strassen[a_idx][mult_idx]
            
            # V: B_j's elements contribute
            # B_j's elements are at positions 2*j (b11), 2*j+1 (b12), 2*j+2 (b21), 2*j+3 (b22)
            for b_idx in range(4):
                V[2*j + b_idx][col] = V_strassen[b_idx][mult_idx]
            
            # W: C_j's elements are at positions 2*j (r11), 2*j+1 (r12), 2*j+2 (r21), 2*j+3 (r22)
            # But wait, the output convention is W[k*n+i, r] where k is row of C, i is col of C?
            # Let me re-read: "W[k*n+i,r]" - this is the trace convention.
            # For C of shape p x n = 10 x 2, the element C[k, i] where k in [0,10), i in [0,2)
            # W index = k*n + i = k*2 + i
            # So for block j (columns 2j, 2j+1 of C), the rows are 0..9, columns are 2j, 2j+1.
            # C[0, 2j] has W index 0*2 + 2j = 2j
            # C[0, 2j+1] has W index 0*2 + (2j+1) = 2j+1
            # C[1, 2j] has W index 1*2 + 2j = 2 + 2j
            # C[1, 2j+1] has W index 1*2 + (2j+1) = 3 + 2j
            # ...
            # C[k, 2j] has W index k*2 + 2j
            # C[k, 2j+1] has W index k*2 + 2j + 1
            
            # R_j is the 2x2 result for block j, but wait - C is 10x2, not 2x10.
            # Let me reconsider. A is 2x2, B is 2x10, C = A*B is 2x10.
            # So C has shape 2x10. C[k, i] where k in [0,2), i in [0,10).
            # W index = k*n + i = k*2 + i? No, the convention says W[k*n+i, r].
            # If C is 2x10, then k is row (0..1), i is col (0..9).
            # W index = k*2 + i? That doesn't match p*n = 10*2 = 20.
            # Actually p*n = 10*2 = 20, so W has 20 rows.
            # The convention W[k*n+i] suggests k ranges over p=10 and i over n=2?
            # Or k over rows of C and i over columns?
            # C is 2x10, so rows=2, cols=10. If W index = k*n+i, with n=2:
            # If k is row index (0..1) and i is col index (0..9): W index = k*2 + i, max = 1*2+9=11. Not 20.
            # If k is col index (0..9) and i is row index (0..1): W index = k*2 + i, max = 9*2+1=19. That's 20 values (0..19).
            # So W[k*n+i] where k=col of C (0..9), i=row of C (0..1).
            # So C[i, k] maps to W[k*2 + i].
            
            # For block j (columns 2j, 2j+1 of C):
            # C[0, 2j] -> W[2j*2 + 0] = W[4j]
            # C[0, 2j+1] -> W[(2j+1)*2 + 0] = W[4j+2]
            # C[1, 2j] -> W[2j*2 + 1] = W[4j+1]
            # C[1, 2j+1] -> W[(2j+1)*2 + 1] = W[4j+3]
            
            for r_idx in range(4):
                if r_idx == 0:  # r11 = C[0, 2j]
                    W[4*j][col] = W_strassen[0][mult_idx]
                elif r_idx == 1:  # r12 = C[0, 2j+1]
                    W[4*j + 2][col] = W_strassen[1][mult_idx]
                elif r_idx == 2:  # r21 = C[1, 2j]
                    W[4*j + 1][col] = W_strassen[2][mult_idx]
                elif r_idx == 3:  # r22 = C[1, 2j+1]
                    W[4*j + 3][col] = W_strassen[3][mult_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
