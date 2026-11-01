"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x4x8 matrix multiplication
    # Rank 56 (vs schoolbook 64)
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # For each half of B (4 columns each), apply Strassen to 2x4x4 block
    # Strassen for 2x2x2: M = A*B where A,B are 2x2
    # M01 = (A00+A11)(B00+B11)
    # M02 = (A10+A11)B11
    # M03 = A00(B01-B11)
    # M04 = A11(B10-B00)
    # M10 = A01(B00+B10)
    # M11 = (A00-A10)(B11+B21) -- wait, that's not right for 2x2
    
    # Actually for 2x2x2 Strassen:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)h
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)e
    # p6 = (c-a)(e+g)
    # p7 = (b-d)(f+h)
    # M[0,0] = p1 + p4 - p2 + p6
    # M[0,1] = p3 + p7
    # M[1,0] = p2 + p5
    # M[1,1] = p1 - p3 + p5 - p7
    
    # For 2x4x4: A is 2x4, B is 4x4
    # Split A into A1 (2x2) and A2 (2x2), B into 2x2 blocks
    # B = [[B11, B12],[B21, B22]] where each is 2x2
    # C = A1*B11 + A2*B21 (row 0 of A times rows 0-1 of B, plus row 0 times rows 2-3... wait)
    
    # Let me be more careful. A is 2x4, B is 4x8.
    # C is 2x8.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split B into left half (cols 0-3) and right half (cols 4-7): B_L (4x4), B_R (4x4)
    # For each half, we compute A * B_half which is 2x4.
    
    # For A (2x4) * B_half (4x4):
    # Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    # Split B_half into 2x2 blocks: B11, B12 (top), B21, B22 (bottom)
    # C[i,k] for k=0,1: A_L[i,:] * B_half[:,k] = A_L[i,:] * [B11[:,k], B21[:,k]]^T
    #   = A_L[i,0]*B11[0,k] + A_L[i,1]*B11[1,k] + A_L[i,0]*B21[0,k] + A_L[i,1]*B21[1,k]
    # Wait no. A is 2x4, so A[i,j] for j=0..3.
    # B is 4x4 (one half), B[j,k] for j=0..3, k=0..3.
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    
    # Split: A = [A1 | A2] where A1 is cols 0,1 and A2 is cols 2,3
    # B = [[B11, B12],[B21, B22]] where B11 is rows 0-1, cols 0-1; B12 is rows 0-1, cols 2-3;
    #       B21 is rows 2-3, cols 0-1; B22 is rows 2-3, cols 2-3
    # C[i,k] for k=0,1: sum_{j=0,1} A[i,j]*B[j,k] + sum_{j=2,3} A[i,j]*B[j,k]
    #   = (A1 * B11[:,k]) + (A2 * B21[:,k])  -- but these are dot products
    # More precisely: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #          = (A[i,0]*B11[0,k] + A[i,1]*B11[1,k]) + (A[i,2]*B21[0,k] + A[i,3]*B21[1,k])
    #          = row_i(A1) dot col_k(B11) + row_i(A2) dot col_k(B21)
    
    # So C[:,0:2] = A1 @ B11 + A2 @ B21  (each is 2x2)
    # C[:,2:4] = A1 @ B12 + A2 @ B22  (each is 2x2)
    
    # For each 2x2x2 multiplication, we use 7 products instead of 4.
    # Total per half: 2 * 7 = 14 products
    # Two halves: 28 products total... wait that gives rank 28, not 56.
    
    # Hmm, let me reconsider. 2x4x8:
    # n=2, m=4, p=8
    # Schoolbook rank = 2*4*8 = 64
    # 
    # If I split B into two 4x4 halves, and for each half do 2x4x4:
    # 2x4x4 schoolbook = 32, two halves = 64 (no savings)
    # 2x4x4 with Strassen on 2x2 blocks: 7*2 = 14 per half? No...
    
    # Let me think differently. 2x4x4:
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into two 2x2: A1, A2
    # Split B into four 2x2: B11, B12, B21, B22
    # C[:,0:2] = A1@B11 + A2@B21
    # C[:,2:4] = A1@B12 + A2@B22
    # Each 2x2 product needs 7 Strassen products
    # 4 such products = 28 rank for 2x4x4
    # Two halves of 2x4x8: 56 rank
    # That's the reduction: 64 -> 56
    
    # Now let's implement this properly.
    # For a 2x2x2 Strassen:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)h
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)e
    # p6 = (c-a)(e+g)
    # p7 = (b-d)(f+h)
    # M[0,0] = p1 + p4 - p2 + p6
    # M[0,1] = p3 + p7
    # M[1,0] = p2 + p5
    # M[1,1] = p1 - p3 + p5 - p7
    
    # For A1@B11 (both 2x2):
    # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B11 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # Contributes to C[:,0:2]
    
    def add_strassen(A_row_offsets, A_col_offsets, B_row_offsets, B_col_offsets, 
                     C_row_offsets, C_col_offsets, base_col):
        """Add 7 Strassen products for a 2x2x2 block.
        A_row_offsets: [r0, r1] rows in A (global)
        A_col_offsets: [c0, c1] cols in A (global)  
        B_row_offsets: [r0, r1] rows in B (global)
        B_col_offsets: [c0, c1] cols in B (global)
        C_row_offsets: [r0, r1] rows in C (global)
        C_col_offsets: [c0, c1] cols in C (global)
        base_col: starting column in U,V,W
        """
        a00 = (A_row_offsets[0], A_col_offsets[0])
        a01 = (A_row_offsets[0], A_col_offsets[1])
        a10 = (A_row_offsets[1], A_col_offsets[0])
        a11 = (A_row_offsets[1], A_col_offsets[1])
        b00 = (B_row_offsets[0], B_col_offsets[0])
        b01 = (B_row_offsets[0], B_col_offsets[1])
        b10 = (B_row_offsets[1], B_col_offsets[0])
        b11 = (B_row_offsets[1], B_col_offsets[1])
        
        c00 = (C_row_offsets[0], C_col_offsets[0])
        c01 = (C_row_offsets[0], C_col_offsets[1])
        c10 = (C_row_offsets[1], C_col_offsets[0])
        c11 = (C_row_offsets[1], C_col_offsets[1])
        
        # p1 = (a00+a11)(b00+b11)
        col = base_col
        U[a00[0]*m + a00[1]][col] = 1  # Wait, A is 2x4, so index is i*m+j where i is row, j is col
        # Actually U has shape (n*m, rank) = (8, rank). Index = i*m + j where i=0..1, j=0..3
        # V has shape (m*p, rank) = (32, rank). Index = j*p + k where j=0..3, k=0..7
        # W has shape (p*n, rank) = (16, rank). Index = k*n + i where k=0..7, i=0..1
        
        # Let me redefine indices:
        # A[i,j]: i in 0..1, j in 0..3 -> U index = i*m + j = i*4 + j
        # B[j,k]: j in 0..3, k in 0..7 -> V index = j*p + k = j*8 + k
        # C[i,k]: i in 0..1, k in 0..7 -> W index = k*n + i = k*2 + i
        
        a_idx = {
            (0,0): 0*4+0, (0,1): 0*4+1, (0,2): 0*4+2, (0,3): 0*4+3,
            (1,0): 1*4+0, (1,1): 1*4+1, (1,2): 1*4+2, (1,3): 1*4+3
        }
        b_idx = {}
        for j in range(4):
            for k in range(8):
                b_idx[(j,k)] = j*8+k
        
        c_idx = {}
        for k in range(8):
            for i in range(2):
                c_idx[(i,k)] = k*2+i
        
        # p1 = (a00+a11)(b00+b11)
        col = base_col
        U[a_idx[(0,0)]][col] = 1
        U[a_idx[(1,1)]][col] = 1
        V[b_idx[(0,0)]][col] = 1
        V[b_idx[(1,1)]][col] = 1
        W[c_idx[(0,0)]][col] = 1
        W[c_idx[(0,1)]][col] = 0
        W[c_idx[(1,0)]][col] = 0
        W[c_idx[(1,1)]][col] = 1
        # M[0,0] += p1, M[1,1] += p1
        col += 1
        
        # p2 = (c+d)h = (a10+a11)*b11
        U[a_idx[(1,0)]][col] = 1
        U[a_idx[(1,1)]][col] = 1
        V[b_idx[(1,1)]][col] = 1
        W[c_idx[(1,0)]][col] = 1
        W[c_idx[(1,1)]][col] = 1
        # M[1,0] += p2, M[1,1] += p2 (wait, M[0,0] -= p2)
        # M[0,0] = p1 + p4 - p2 + p6
        # M[1,0] = p2 + p5
        # So M[0,0] gets -p2, M[1,0] gets +p2
        W[c_idx[(0,0)]][col] = -1
        W[c_idx[(1,0)]][col] = 1
        col += 1
        
        # p3 = a(f-h) = a00*(b01-b11)
        U[a_idx[(0,0)]][col] = 1
        V[b_idx[(0,1)]][col] = 1
        V[b_idx[(1,1)]][col] = -1
        W[c_idx[(0,1)]][col] = 1
        W[c_idx[(1,1)]][col] = -1
        # M[0,1] += p3, M[1,1] -= p3
        col += 1
        
        # p4 = d(g-e) = a11*(b10-b00)
        U[a_idx[(1,1)]][col] = 1
        V[b_idx[(1,0)]][col] = 1
        V[b_idx[(0,0)]][col] = -1
        W[c_idx[(0,0)]][col] = 1
        # M[0,0] += p4
        col += 1
        
        # p5 = (a+b)e = (a00+a01)*b00
        U[a_idx[(0,0)]][col] = 1
        U[a_idx[(0,1)]][col] = 1
        V[b_idx[(0,0)]][col] = 1
        W[c_idx[(1,0)]][col] = 1
        W[c_idx[(1,1)]][col] = 1
        # M[1,0] += p5, M[1,1] += p5
        col += 1
        
        # p6 = (c-a)(e+g) = (a10-a00)*(b00+b10)
        U[a_idx[(1,0)]][col] = 1
        U[a_idx[(0,0)]][col] = -1
        V[b_idx[(0,0)]][col] = 1
        V[b_idx[(1,0)]][col] = 1
        W[c_idx[(0,0)]][col] = 1
        # M[0,0] += p6
        col += 1
        
        # p7 = (b-d)(f+h) = (a01-a11)*(b01+b11)
        U[a_idx[(0,1)]][col] = 1
        U[a_idx[(1,1)]][col] = -1
        V[b_idx[(0,1)]][col] = 1
        V[b_idx[(1,1)]][col] = 1
        W[c_idx[(0,1)]][col] = 1
        # M[0,1] += p7
        col += 1
        
        return col
    
    # Now assemble for 2x4x8
    # C[:,0:2] = A1@B11 + A2@B21
    # C[:,2:4] = A1@B12 + A2@B22
    # C[:,4:6] = A1@B31 + A2@B41  (B31 = B[0:2,4:6], B41 = B[2:4,4:6])
    # C[:,6:8] = A1@B32 + A2@B42  (B32 = B[0:2,6:8], B42 = B[2:4,6:8])
    
    # A1 = A[:,0:2], A2 = A[:,2:3]
    # For each of the 4 pairs (A1@B_left, A2@B_right) where B_left is top 2x2, B_right is bottom 2x2:
    
    col = 0
    # C[:,0:2]: A1@B[0:2,0:2] + A2@B[2:4,0:2]
    col = add_strassen([0,1],[0,1],[0,1],[0,1],[0,1],[0,1], col)
    col = add_strassen([0,1],[2,3],[2,3],[0,1],[0,1],[0,1], col)
    
    # C[:,2:4]: A1@B[0:2,2:4] + A2@B[2:4,2:4]
    col = add_strassen([0,1],[0,1],[0,1],[2,3],[0,1],[2,3], col)
    col = add_strassen([0,1],[2,3],[2,3],[2,3],[0,1],[2,3], col)
    
    # C[:,4:6]: A1@B[0:2,4:6] + A2@B[2:4,4:6]
    col = add_strassen([0,1],[0,1],[0,1],[4,5],[0,1],[4,5], col)
    col = add_strassen([0,1],[2,3],[2,3],[4,5],[0,1],[4,5], col)
    
    # C[:,6:8]: A1@B[0:2,6:8] + A2@B[2:4,6:8]
    col = add_strassen([0,1],[0,1],[0,1],[6,7],[0,1],[6,7], col)
    col = add_strassen([0,1],[2,3],[2,3],[6,7],[0,1],[6,7], col)
    
    assert col == 56, f"Expected 56, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
