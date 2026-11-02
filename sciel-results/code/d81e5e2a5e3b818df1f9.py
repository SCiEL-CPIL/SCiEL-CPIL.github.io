"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (24, rank)  
    # W: (p*n, rank) = (24, rank)
    # W[k*n+i, r] corresponds to C[i,k]
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # For each pair of columns (2p_idx, 2p_idx+1)
    for p_idx in range(4):  # 4 pairs for 8 columns
        k0 = 2 * p_idx
        k1 = 2 * p_idx + 1
        
        # A indices: A[i,j] is at U[i*m + j] = U[i*3 + j]
        # B indices: B[j,k] is at V[j*p + k] = V[j*8 + k]
        # C indices: C[i,k] is at W[k*n + i] = W[k*3 + i]
        
        # A_top = A[0:2, 0:2]: A[0,0],A[0,1],A[1,0],A[1,1] -> U[0],U[1],U[3],U[4]
        # A_mid = A[0:2, 2]: A[0,2],A[1,2] -> U[2],U[5]
        # A_bot_left = A[2, 0:2]: A[2,0],A[2,1] -> U[6],U[7]
        # A_bot_right = A[2, 2]: A[2,2] -> U[8]
        
        # B_top = B[0:2, k0:k1+1]: B[0,k0],B[0,k1],B[1,k0],B[1,k1] -> V[k0],V[k1],V[8+k0],V[8+k1]
        # B_bot = B[2, k0:k1+1]: B[2,k0],B[2,k1] -> V[16+k0],V[16+k1]
        
        # Strassen for A_top @ B_top (2x2 x 2x2)
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # M1
        U[0][col] += 1    # A[0,0]
        U[4][col] += 1    # A[1,1]
        V[k0][col] += 1      # B[0,k0]
        V[8+k1][col] += 1   # B[1,k1]
        # C[0,k0] += M1, C[1,k1] += M1
        W[k0*3+0][col] += 1
        W[k1*3+1][col] += 1
        col += 1
        
        # M2
        U[3][col] += 1    # A[1,0]
        U[4][col] += 1    # A[1,1]
        V[k0][col] += 1      # B[0,k0]
        # C[1,k0] += M2
        W[k0*3+1][col] += 1
        col += 1
        
        # M3
        U[0][col] += 1    # A[0,0]
        V[k1][col] += 1      # B[0,k1]
        V[8+k1][col] -= 1   # -B[1,k1]
        # C[0,k1] += M3, C[1,k1] += M3
        W[k1*3+0][col] += 1
        W[k1*3+1][col] += 1
        col += 1
        
        # M4
        U[4][col] += 1    # A[1,1]
        V[8+k0][col] += 1   # B[1,k0]
        V[k0][col] -= 1      # -B[0,k0]
        # C[0,k0] -= M4, C[1,k0] += M4
        W[k0*3+0][col] -= 1
        W[k0*3+1][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1    # A[0,0]
        U[1][col] += 1    # A[0,1]
        V[8+k1][col] += 1   # B[1,k1]
        # C[0,k0] += M5, C[0,k1] += M5
        W[k0*3+0][col] += 1
        W[k1*3+0][col] += 1
        col += 1
        
        # M6
        U[3][col] += 1    # A[1,0]
        U[0][col] -= 1    # -A[0,0]
        V[k0][col] += 1      # B[0,k0]
        V[k1][col] += 1      # B[0,k1]
        # C[1,k1] += M6
        W[k1*3+1][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1    # A[0,1]
        U[4][col] -= 1    # -A[1,1]
        V[8+k0][col] += 1   # B[1,k0]
        V[8+k1][col] += 1   # B[1,k1]
        # C[0,k0] -= M7
        W[k0*3+0][col] -= 1
        col += 1
        
        # A_mid @ B_bot: 2x1 * 1x2 = 2x2 (4 mults)
        # A_mid = [A[0,2], A[1,2]], B_bot = [B[2,k0], B[2,k1]]
        # Result: [[A[0,2]*B[2,k0], A[0,2]*B[2,k1]],
        #          [A[1,2]*B[2,k0], A[1,2]*B[2,k1]]]
        
        # A[0,2]*B[2,k0]
        U[2][col] += 1
        V[16+k0][col] += 1
        W[k0*3+0][col] += 1
        col += 1
        
        # A[0,2]*B[2,k1]
        U[2][col] += 1
        V[16+k1][col] += 1
        W[k1*3+0][col] += 1
        col += 1
        
        # A[1,2]*B[2,k0]
        U[5][col] += 1
        V[16+k0][col] += 1
        W[k0*3+1][col] += 1
        col += 1
        
        # A[1,2]*B[2,k1]
        U[5][col] += 1
        V[16+k1][col] += 1
        W[k1*3+1][col] += 1
        col += 1
        
        # A_bot_left @ B_top: 1x2 * 2x2 = 1x2 (4 mults)
        # A_bot_left = [A[2,0], A[2,1]], B_top = [[B[0,k0],B[0,k1]],[B[1,k0],B[1,k1]]]
        # C[2,k0] = A[2,0]*B[0,k0] + A[2,1]*B[1,k0]
        # C[2,k1] = A[2,0]*B[0,k1] + A[2,1]*B[1,k1]
        
        # A[2,0]*B[0,k0]
        U[6][col] += 1
        V[k0][col] += 1
        W[k0*3+2][col] += 1
        col += 1
        
        # A[2,1]*B[1,k0]
        U[7][col] += 1
        V[8+k0][col] += 1
        W[k0*3+2][col] += 1
        col += 1
        
        # A[2,0]*B[0,k1]
        U[6][col] += 1
        V[k1][col] += 1
        W[k1*3+2][col] += 1
        col += 1
        
        # A[2,1]*B[1,k1]
        U[7][col] += 1
        V[8+k1][col] += 1
        W[k1*3+2][col] += 1
        col += 1
        
        # A_bot_right * B_bot: 1x1 * 1x2 = 1x2 (2 mults)
        # A_bot_right = A[2,2], B_bot = [B[2,k0], B[2,k1]]
        # C[2,k0] += A[2,2]*B[2,k0]
        # C[2,k1] += A[2,2]*B[2,k1]
        
        # A[2,2]*B[2,k0]
        U[8][col] += 1
        V[16+k0][col] += 1
        W[k0*3+2][col] += 1
        col += 1
        
        # A[2,2]*B[2,k1]
        U[8][col] += 1
        V[16+k1][col] += 1
        W[k1*3+2][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
