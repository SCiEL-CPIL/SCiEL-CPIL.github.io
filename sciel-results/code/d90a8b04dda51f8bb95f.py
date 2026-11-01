"""Program implementing a rank-28 tensor decomposition for 2x4x4 matrix multiplication using Strassen on 2x2 blocks."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28

    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    # Index conventions:
    #   U[i*m + j, c]
    #   V[j*p + k, c]
    #   W[k*n + i, c]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # We split A (2x4) into two 2x2 blocks horizontally: A = [A_L | A_R]
    #   A_L = A[:, 0:2], A_R = A[:, 2:4]
    #   A_L[i, j] corresponds to U[i*m + j] for j in {0,1}
    #   A_R[i, j] corresponds to U[i*m + (j+2)] for j in {0,1}
    #
    # We split B (4x4) into four 2x2 blocks:
    #   B = [[B_LL, B_LR], [B_RL, B_RR]]
    #   B_LL[r, c] for r in {0,1}, c in {0,1} -> V[r*4 + c]
    #   B_LR[r, c] for r in {0,1}, c in {2,3} -> V[r*4 + (c-2+2)] = V[r*4 + c]
    #   B_RL[r, c] for r in {2,3}, c in {0,1} -> V[r*4 + c]
    #   B_RR[r, c] for r in {2,3}, c in {2,3} -> V[r*4 + c]
    #
    # Wait, let me re-index carefully.
    # B is 4x4. V[j*p + k, c] where j is row index of B (0..3), k is col index of B (0..3).
    # Actually V has shape (m*p, rank) = (4*4, rank) = (16, rank).
    # The index is j*p + k where j is the row of B (0..3) and k is the column of B (0..3).
    # So V[r*4 + c, col] corresponds to B[r, c].
    #
    # A is 2x4. U[i*m + j, c] where i is row of A (0..1), j is col of A (0..3).
    # So U[i*4 + j, col] corresponds to A[i, j].
    #
    # C is 2x4. W[k*n + i, c] where k is row of C (0..3), i is col of C (0..1).
    # So W[k*2 + i, col] corresponds to C[k, i].
    #
    # Block structure:
    # A = [A11 A12] where A11 is 2x2 (cols 0,1), A12 is 2x2 (cols 2,3)
    # B = [B11 B12; B21 B22] where each is 2x2
    #   B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C = [C11 C12] where C11 = A11*B11 + A12*B21 (2x2), C12 = A11*B12 + A12*B22 (2x2)
    #
    # Strassen for 2x2 matrix multiplication:
    # For matrices X (2x2) and Y (2x2), Z = X*Y can be computed with 7 products:
    #   M1 = (x11+x22)*(y11+y22)
    #   M2 = (x21+x22)*y11
    #   M3 = x11*(y12-y22)
    #   M4 = x22*(y21-y11)
    #   M5 = (x11+x12)*y22
    #   M6 = (x21-x11)*(y11+y12)
    #   M7 = (x12-x22)*(y21+y22)
    # Then:
    #   z11 = M1 - M4 + M5 + M7
    #   z12 = M3 + M5
    #   z21 = M2 + M4
    #   z22 = M1 - M3 + M2 + M6
    #
    # We need to do this for two block products:
    #   C11 = A11*B11 + A12*B21
    #   C12 = A11*B12 + A12*B22
    #
    # But Strassen applies to a single 2x2 * 2x2 product. We have sums of two products.
    # We can't directly apply Strassen to the sum. Instead, we should think of this differently.
    #
    # Actually, let me reconsider. The standard approach for M(2,4,4) using Strassen:
    # Think of it as multiplying a 2x4 by 4x4. We can pad or restructure.
    #
    # Alternative: use the fact that M(2,4,4) <= M(4,4,4) via embedding, but that gives rank 49.
    #
    # Better: Use the decomposition of 2x4x4 into four 2x2x2 subproblems.
    # A is 2x4 = two 2x2 blocks side by side: [A1 | A2]
    # B is 4x4 = four 2x2 blocks: [[B11, B12], [B21, B22]]
    # C = [A1*B11 + A2*B21 | A1*B12 + A2*B22]
    #
    # This is two independent 2x2*2x2 + 2x2*2x2 computations, each giving a 2x2 result.
    # We can't combine them into a single Strassen call.
    #
    # However, we CAN view this as: compute P1 = A1*B11, P2 = A2*B21, then C_left = P1 + P2
    # and P3 = A1*B12, P4 = A2*B22, then C_right = P3 + P4
    # Each Pi is a 2x2*2x2 product, rank 7. Total rank 4*7 = 28.
    # The additions are just linear combinations in the output, which don't add to the bilinear rank.
    #
    # So we need 4 separate Strassen decompositions, one for each (A_block, B_block) pair:
    #   Strassen 1: A1 * B11 -> contributes to C_left
    #   Strassen 2: A2 * B21 -> contributes to C_left
    #   Strassen 3: A1 * B12 -> contributes to C_right
    #   Strassen 4: A2 * B22 -> contributes to C_right
    #
    # Let me define the Strassen formulas for each.
    
    # Helper: add a Strassen decomposition for X (2x2) * Y (2x2) -> Z (2x2)
    # X[i, j] is at U[x_base + i*2 + j, c] for i,j in {0,1}
    # Y[i, j] is at V[y_base + i*4 + j, c] for i,j in {0,1}  (since Y is embedded in 4x4, rows start at y_base)
    #   Wait, Y is a 2x2 block of the 4x4 B. If the block starts at row r0, col c0, then
    #   Y[i,j] = B[r0+i, c0+j] = V[(r0+i)*4 + (c0+j), c]
    # Z[i, j] is at W[z_base + j*2 + i, c] for i,j in {0,1}
    #   C is 2x4. If the output block is at row r0, col c0 (each 2x2), then
    #   Z[i,j] = C[r0+i, c0+j] = W[(r0+i)*2 + (c0+j), c]
    
    def strassen(x_base, y_row0, y_col0, z_row0, z_col0, start_col):
        """
        Add Strassen decomposition for X[2x2] * Y[2x2] -> Z[2x2].
        X[i,j] at U[x_base + i*2 + j]
        Y[i,j] at V[(y_row0+i)*4 + (y_col0+j)]
        Z[i,j] at W[(z_row0+i)*2 + (z_col0+j)]
        """
        # x11 at U[x_base + 0], x12 at U[x_base + 1]
        # x21 at U[x_base + 2], x22 at U[x_base + 3]
        # y11 at V[(y_row0+0)*4 + (y_col0+0)], y12 at V[(y_row0+0)*4 + (y_col0+1)]
        # y21 at V[(y_row0+1)*4 + (y_col0+0)], y22 at V[(y_row0+1)*4 + (y_col0+1)]
        # z11 at W[(z_row0+0)*2 + (z_col0+0)], z12 at W[(z_row0+0)*2 + (z_col0+1)]
        # z21 at W[(z_row0+1)*2 + (z_col0+0)], z22 at W[(z_row0+1)*2 + (z_col0+1)]
        
        col = start_col
        # M1 = (x11+x22)*(y11+y22)
        U[x_base + 0][col] += 1
        U[x_base + 3][col] += 1
        V[(y_row0+0)*4 + (y_col0+0)][col] += 1
        V[(y_row0+1)*4 + (y_col0+1)][col] += 1
        W[(z_row0+0)*2 + (z_col0+0)][col] += 1
        W[(z_row0+1)*2 + (z_col0+0)][col] -= 1
        col += 1
        
        # M2 = (x21+x22)*y11
        U[x_base + 2][col] += 1
        U[x_base + 3][col] += 1
        V[(y_row0+0)*4 + (y_col0+0)][col] += 1
        W[(z_row0+1)*2 + (z_col0+0)][col] += 1
        W[(z_row0+1)*2 + (z_col0+1)][col] += 1
        col += 1
        
        # M3 = x11*(y12-y22)
        U[x_base + 0][col] += 1
        V[(y_row0+0)*4 + (y_col0+1)][col] += 1
        V[(y_row0+1)*4 + (y_col0+1)][col] -= 1
        W[(z_row0+0)*2 + (z_col0+1)][col] += 1
        W[(z_row0+1)*2 + (z_col0+1)][col] -= 1
        col += 1
        
        # M4 = x22*(y21-y11)
        U[x_base + 3][col] += 1
        V[(y_row0+1)*4 + (y_col0+0)][col] += 1
        V[(y_row0+0)*4 + (y_col0+0)][col] -= 1
        W[(z_row0+1)*2 + (z_col0+0)][col] += 1
        W[(z_row0+0)*2 + (z_col0+0)][col] -= 1
        col += 1
        
        # M5 = (x11+x12)*y22
        U[x_base + 0][col] += 1
        U[x_base + 1][col] += 1
        V[(y_row0+1)*4 + (y_col0+1)][col] += 1
        W[(z_row0+0)*2 + (z_col0+0)][col] += 1
        W[(z_row0+0)*2 + (z_col0+1)][col] += 1
        col += 1
        
        # M6 = (x21-x11)*(y11+y12)
        U[x_base + 2][col] += 1
        U[x_base + 0][col] -= 1
        V[(y_row0+0)*4 + (y_col0+0)][col] += 1
        V[(y_row0+0)*4 + (y_col0+1)][col] += 1
        W[(z_row0+1)*2 + (z_col0+1)][col] += 1
        col += 1
        
        # M7 = (x12-x22)*(y21+y22)
        U[x_base + 1][col] += 1
        U[x_base + 3][col] -= 1
        V[(y_row0+1)*4 + (y_col0+0)][col] += 1
        V[(y_row0+1)*4 + (y_col0+1)][col] += 1
        W[(z_row0+0)*2 + (z_col0+0)][col] += 1
        W[(z_row0+1)*2 + (z_col0+1)][col] += 1
        col += 1
        
        return col
    
    # Strassen 1: A1 * B11 -> C_left (rows 0-1, cols 0-1)
    # A1 = A[:, 0:2], x_base = 0 (A[i,0:2] at U[i*4 + 0:2])
    #   A1[i,j] at U[i*4 + j] for i in {0,1}, j in {0,1}
    #   So x_base for i=0 is 0, for i=1 is 4. But our Strassen helper assumes
    #   X[i,j] at U[x_base + i*2 + j]. So x_base=0 gives:
    #   X[0,0]=U[0], X[0,1]=U[1], X[1,0]=U[2], X[1,1]=U[3]
    #   But we need X[0,0]=U[0], X[0,1]=U[1], X[1,0]=U[4], X[1,1]=U[5]
    #   This doesn't match. Let me fix the indexing.
    
    # Let me redo this more carefully.
    
    # U[i*m + j, c] = U[i*4 + j, c] for A[i,j]
    # A1[i,j] = A[i, j] for j in {0,1}
    # So A1[i,j] at U[i*4 + j]
    # A2[i,j] = A[i, 2+j] for j in {0,1}
    # So A2[i,j] at U[i*4 + 2 + j]
    
    # B11[i,j] = B[i, j] for i,j in {0,1}
    # B11[i,j] at V[i*4 + j]
    # B12[i,j] = B[i, 2+j] for i,j in {0,1}
    # B12[i,j] at V[i*4 + 2 + j]
    # B21[i,j] = B[2+i, j] for i,j in {0,1}
    # B21[i,j] at V[(2+i)*4 + j]
    # B22[i,j] = B[2+i, 2+j] for i,j in {0,1}
    # B22[i,j] at V[(2+i)*4 + 2 + j]
    
    # C11[i,j] = C[i, j] for i,j in {0,1}
    # C11[i,j] at W[i*2 + j]
    # C12[i,j] = C[i, 2+j] for i,j in {0,1}
    # C12[i,j] at W[i*2 + 2 + j]
    
    # Let me write a more flexible Strassen function.
    
    def strassen_generic(x_indices, y_indices, z_indices, start_col):
        """
        x_indices: dict mapping (i,j) -> U index
        y_indices: dict mapping (i,j) -> V index
        z_indices: dict mapping (i,j) -> W index
        """
        x11 = x_indices[(0,0)]
        x12 = x_indices[(0,1)]
        x21 = x_indices[(1,0)]
        x22 = x_indices[(1,1)]
        y11 = y_indices[(0,0)]
        y12 = y_indices[(0,1)]
        y21 = y_indices[(1,0)]
        y22 = y_indices[(1,1)]
        z11 = z_indices[(0,0)]
        z12 = z_indices[(0,1)]
        z21 = z_indices[(1,0)]
        z22 = z_indices[(1,1)]
        
        col = start_col
        # M1 = (x11+x22)*(y11+y22)
        U[x11][col] += 1
        U[x22][col] += 1
        V[y11][col] += 1
        V[y22][col] += 1
        W[z11][col] += 1
        W[z22][col] -= 1
        col += 1
        
        # M2 = (x21+x22)*y11
        U[x21][col] += 1
        U[x22][col] += 1
        V[y11][col] += 1
        W[z21][col] += 1
        W[z22][col] += 1
        col += 1
        
        # M3 = x11*(y12-y22)
        U[x11][col] += 1
        V[y12][col] += 1
        V[y22][col] -= 1
        W[z12][col] += 1
        W[z22][col] -= 1
        col += 1
        
        # M4 = x22*(y21-y11)
        U[x22][col] += 1
        V[y21][col] += 1
        V[y11][col] -= 1
        W[z21][col] += 1
        W[z11][col] -= 1
        col += 1
        
        # M5 = (x11+x12)*y22
        U[x11][col] += 1
        U[x12][col] += 1
        V[y22][col] += 1
        W[z11][col] += 1
        W[z12][col] += 1
        col += 1
        
        # M6 = (x21-x11)*(y11+y12)
        U[x21][col] += 1
        U[x11][col] -= 1
        V[y11][col] += 1
        V[y12][col] += 1
        W[z22][col] += 1
        col += 1
        
        # M7 = (x12-x22)*(y21+y22)
        U[x12][col] += 1
        U[x22][col] -= 1
        V[y21][col] += 1
        V[y22][col] += 1
        W[z11][col] += 1
        W[z22][col] += 1
        col += 1
        
        return col
    
    # Define index mappings
    # A1[i,j] at U[i*4 + j]
    A1_idx = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    # A2[i,j] at U[i*4 + 2 + j]
    A2_idx = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    
    # B11[i,j] at V[i*4 + j]
    B11_idx = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    # B12[i,j] at V[i*4 + 2 + j]
    B12_idx = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    # B21[i,j] at V[(2+i)*4 + j]
    B21_idx = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
    # B22[i,j] at V[(2+i)*4 + 2 + j]
    B22_idx = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
    
    # C11[i,j] at W[i*2 + j]
    C11_idx = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}
    # C12[i,j] at W[i*2 + 2 + j]
    C12_idx = {(0,0): 4, (0,1): 5, (1,0): 6, (1,1): 7}
    
    # Apply four Strassen decompositions
    col = 0
    col = strassen_generic(A1_idx, B11_idx, C11_idx, col)  # A1 * B11 -> C11
    col = strassen_generic(A2_idx, B21_idx, C11_idx, col)  # A2 * B21 -> C11
    col = strassen_generic(A1_idx, B12_idx, C12_idx, col)  # A1 * B12 -> C12
    col = strassen_generic(A2_idx, B22_idx, C12_idx, col)  # A2 * B22 -> C12
    
    # col should now be 28
    assert col == 28, f"Expected 28 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
