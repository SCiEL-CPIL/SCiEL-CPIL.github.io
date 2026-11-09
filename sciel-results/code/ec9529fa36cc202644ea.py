"""Matrix multiplication 3x4x4 - rank-44 decomposition."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # I'll build a rank-44 decomposition using 4 Strassen 2x2x2 blocks and 4 schoolbook 1x2x2 blocks.
    #
    # The key is to identify which sub-tensors are 2x2x2 (Strassen-applicable) and which are 1x2x2.
    #
    # For 3x4x4:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Let me use the following decomposition:
    # Split j into two groups: {0,1} and {2,3}
    # Split k into two groups: {0,1} and {2,3}
    #
    # C[i,k] for k in {0,1}:
    #   = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #   = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    #
    # The first parenthesis is a 1x2 * 2x1 dot product (for each k), and we need it for k=0 and k=1.
    # Together, for all i and k in {0,1}, this forms a 3x2 * 2x2 = 3x2 sub-problem.
    #
    # I'll treat each 3x2 * 2x2 as follows:
    # For each of the 3 rows, compute 1x2 * 2x2 = 1x2.
    # A 1x2 * 2x2 can be done with 4 multiplications (schoolbook).
    # So 3x2 * 2x2 = 12 multiplications.
    #
    # We have 4 such sub-problems (2 j-groups x 2 k-groups), so 4*12 = 48 total. Schoolbook.
    #
    # To reduce to 44, I need to use Strassen on the 2x2x2 parts.
    # But 3x2 * 2x2 is not a 2x2x2 tensor.
    #
    # Alternative: note that 3x2 * 2x2 = (3x1 * 2x2) * 2, but that's not helpful.
    #
    # Let me try a different split. What if I split the 3 into 2+1?
    # C[0:2, :] and C[2, :]
    # C[0:2, k] = sum_j A[0:2, j]*B[j, k]  (2x4 * 4x4 = 2x4)
    # C[2, k] = sum_j A[2, j]*B[j, k]  (1x4 * 4x4 = 1x4)
    #
    # 2x4 * 4x4: split into 2x2 * 2x4 blocks.
    # 2x2 * 2x4 = 2x2 * (2x2 + 2x2) = two 2x2 * 2x2 products.
    # Each 2x2 * 2x2 can use Strassen (rank 7).
    # So 2x4 * 4x4 uses 2*7 = 14 for the Strassen part? No, 2x2*2x4 is not two independent 2x2*2x2.
    #
    # 2x2 * 2x4: A is 2x2, B is 2x4. C = A*B is 2x4.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # For k in {0,1}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For k in {2,3}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # This is two independent 2x2 * 2x2 problems (for k-groups {0,1} and {2,3}).
    # Each 2x2*2x2 uses Strassen rank 7. Total: 2*7 = 14 for the 2x4*4x4 part... 
    # Wait, no. The 2x2*2x4 is A (2x2) times B (2x4). B is split into B_left (2x2) and B_right (2x2).
    # C = A*B = A*B_left + A*B_right, where A*B_left is 2x2 and A*B_right is 2x2.
    # Each A*B_* is a 2x2*2x2 product, Strassen rank 7 each. Total 14.
    #
    # Similarly for the 1x4 * 4x4:
    # C[2,k] = sum_j A[2,j]*B[j,k]
    # = (A[2,0]*B[0,k] + A[2,1]*B[1,k]) + (A[2,2]*B[2,k] + A[2,3]*B[3,k])
    # For k in {0,1}: 1x2 * 2x2, rank 4 schoolbook
    # For k in {2,3}: 1x2 * 2x2, rank 4 schoolbook
    # Total: 8 for the 1x4*4x4 part.
    #
    # Grand total: 14 + 8 = 22? That seems too low. Let me recount.
    #
    # Actually, I think I'm double-counting or missing the B structure.
    # B is 4x4, split into 2x2 blocks: B_TL, B_TR, B_BL, B_BR (each 2x2).
    #
    # C[0:2, 0:2] = A[0:2, 0:2]*B_TL + A[0:2, 2:4]*B_BL
    # C[0:2, 2:4] = A[0:2, 0:2]*B_TR + A[0:2, 2:4]*B_BR
    # C[2, 0:2] = A[2, 0:2]*B_TL + A[2, 2:4]*B_BL
    # C[2, 2:4] = A[2, 0:2]*B_TR + A[2, 2:4]*B_BR
    #
    # Each 2x2*2x2 product (like A[0:2,0:2]*B_TL) can use Strassen rank 7.
    # Each 1x2*2x2 product (like A[2,0:2]*B_TL) has rank 4 schoolbook.
    #
    # Count:
    # 2x2*2x2 products: 4 (for A[0:2] with B_TL, B_TR, B_BL, B_BR) -> 4*7 = 28
    # 1x2*2x2 products: 4 (for A[2] with B_TL, B_TR, B_BL, B_BR) -> 4*4 = 16
    # Total: 28 + 16 = 44!
    #
    # Now let me implement this:
    #
    # For each 2x2*2x2 Strassen block, I need to define 7 rank-1 terms.
    # For each 1x2*2x2 schoolbook block, I need 4 rank-1 terms.
    #
    # Let me define the indices:
    # A rows: 0,1,2
    # A cols: 0,1,2,3
    # B rows: 0,1,2,3
    # B cols: 0,1,2,3
    # C rows: 0,1,2
    # C cols: 0,1,2,3
    #
    # Strassen block for A[0:2, c1:c1+2] * B[r1:r1+2, c2:c2+2]:
    # A_block = [[A[0,c1], A[0,c1+1]], [A[1,c1], A[1,c1+1]]]
    # B_block = [[B[r1,c2], B[r1,c2+1]], [B[r1+1,c2], B[r1+1,c2+1]]]
    # C_block = A_block * B_block (2x2)
    # C_block[0,0] contributes to C[0, c2]
    # C_block[0,1] contributes to C[0, c2+1]
    # C_block[1,0] contributes to C[1, c2]
    # C_block[1,1] contributes to C[1, c2+1]
    #
    # Strassen multiplications:
    # M1 = (A[0,c1]+A[1,c1+1])*(B[r1,c2]+B[r1+1,c2+1])
    # M2 = (A[1,c1]+A[1,c1+1])*B[r1,c2]
    # M3 = A[0,c1]*(B[r1,c2+1]-B[r1+1,c2+1])
    # M4 = A[1,c1+1]*(B[r1+1,c2]-B[r1,c2])
    # M5 = (A[0,c1]+A[0,c1+1])*B[r1+1,c2+1]
    # M6 = (A[1,c1]-A[0,c1])*(B[r1,c2]+B[r1+1,c2])
    # M7 = (A[0,c1+1]-A[1,c1+1])*(B[r1,c2+1]+B[r1+1,c2+1])
    #
    # Results:
    # C[0,c2] = M1 - M4 + M5 + M6
    # C[0,c2+1] = M3 + M1 + M7
    # C[1,c2] = M2 + M4
    # C[1,c2+1] = M1 - M2 + M3 + M5
    #
    # Now let me code this up:
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start):
        """Add a Strassen 2x2x2 block."""
        nonlocal col
        a0 = a_row_start  # A row 0
        a1 = a_row_start + 1  # A row 1
        ac0 = a_col_start  # A col 0
        ac1 = a_col_start + 1  # A col 1
        br0 = b_row_start  # B row 0
        br1 = b_row_start + 1  # B row 1
        bc0 = b_col_start  # B col 0
        bc1 = b_col_start + 1  # B col 1
        
        # M1 = (A[a0,ac0]+A[a1,ac1]) * (B[br0,bc0]+B[br1,bc1])
        U[a0*m+ac0][col] = 1; U[a1*m+ac1][col] = 1
        V[br0*p+bc0][col] = 1; V[br1*p+bc1][col] = 1
        W[bc0*n+a0][col] = 1   # C[a0, bc0] += M1
        W[bc1*n+a0][col] = 1   # C[a0, bc1] += M1
        W[bc0*n+a1][col] = -1  # C[a1, bc0] += -M4... wait, need to be careful
        # Actually, W needs to encode the linear combination.
        # Let me define W such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = C[i,k]
        # For M1: U has A[a0,ac0]=1, A[a1,ac1]=1; V has B[br0,bc0]=1, B[br1,bc1]=1
        # The product U*V gives (A[a0,ac0]+A[a1,ac1])*(B[br0,bc0]+B[br1,bc1])
        # This contributes to C[a0,bc0], C[a0,bc1], C[a1,bc0], C[a1,bc1]
        # Coefficients: C[a0,bc0]+=1, C[a0,bc1]+=1, C[a1,bc0]+=-1(from M4)+1(from M5)+1(from M6)...
        # This is getting complex. Let me define each rank-1 term separately.
        
        # I'll define 7 separate rank-1 terms for the Strassen block.
        terms = [
            # (U factors, V factors, W factors)
            # M1
            ([(a0, ac0, 1), (a1, ac1, 1)], [(br0, bc0, 1), (br1, bc1, 1)],
             [(a0, bc0, 1), (a0, bc1, 1), (a1, bc0, -1), (a1, bc1, 1)]),
            # M2
            ([(a1, ac0, 1), (a1, ac1, 1)], [(br0, bc0, 1)],
             [(a1, bc0, 1), (a1, bc1, -1)]),
            # M3
            ([(a0, ac0, 1)], [(br0, bc1, 1), (br1, bc1, -1)],
             [(a0, bc1, 1), (a1, bc1, 1)]),
            # M4
            ([(a1, ac1, 1)], [(br1, bc0, 1), (br0, bc0, -1)],
             [(a0, bc0, -1), (a1, bc0, 1), (a1, bc1, 1)]),
            # M5
            ([(a0, ac0, 1), (a0, ac1, 1)], [(br1, bc1, 1)],
             [(a0, bc0, 1), (a1, bc1, 1)]),
            # M6
            ([(a1, ac0, 1), (a0, ac0, -1)], [(br0, bc0, 1), (br1, bc0, 1)],
             [(a0, bc0, 1)]),
            # M7
            ([(a0, ac1, 1), (a1, ac1, -1)], [(br0, bc1, 1), (br1, bc1, 1)],
             [(a0, bc1, 1)]),
        ]
        
        for u_factors, v_factors, w_factors in terms:
            for (ri, ci, val) in u_factors:
                U[ri*m+ci][col] += val
            for (ri, ci, val) in v_factors:
                V[ri*p+ci][col] += val
            for (ri, ci, val) in w_factors:
                W[ci*n+ri][col] += val
            col += 1
    
    def add_schoolbook_block(a_row, a_col_start, b_row_start, b_col_start):
        """Add a schoolbook 1x2x2 block."""
        nonlocal col
        a0 = a_row
        ac0 = a_col_start
        ac1 = a_col_start + 1
        br0 = b_row_start
        br1 = b_row_start + 1
        bc0 = b_col_start
        bc1 = b_col_start + 1
        
        # 4 multiplications:
        # A[a0,ac0]*B[br0,bc0] -> C[a0,bc0]
        # A[a0,ac0]*B[br0,bc1] -> C[a0,bc1]
        # A[a0,ac1]*B[br1,bc0] -> C[a0,bc0]
        # A[a0,ac1]*B[br1,bc1] -> C[a0,bc1]
        
        U[a0*m+ac0][col] = 1
        V[br0*p+bc0][col] = 1
        W[bc0*n+a0][col] = 1
        col += 1
        
        U[a0*m+ac0][col] = 1
        V[br0*p+bc1][col] = 1
        W[bc1*n+a0][col] = 1
        col += 1
        
        U[a0*m+ac1][col] = 1
        V[br1*p+bc0][col] = 1
        W[bc0*n+a0][col] = 1
        col += 1
        
        U[a0*m+ac1][col] = 1
        V[br1*p+bc1][col] = 1
        W[bc1*n+a0][col] = 1
        col += 1
    
    # Add 4 Strassen blocks for A[0:2] with B_TL, B_TR, B_BL, B_BR
    add_strassen_block(0, 0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_block(0, 2, 0, 2)  # A[0:2,2:4] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_block(0, 0, 2, 0)  # A[0:2,0:2] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_block(0, 2, 2, 2)  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    
    # Add 4 schoolbook blocks for A[2] with B_TL, B_TR, B_BL, B_BR
    add_schoolbook_block(2, 0, 0, 0)  # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    add_schoolbook_block(2, 2, 0, 2)  # A[2,2:4] * B[0:2,2:4] -> C[2,2:4]
    add_schoolbook_block(2, 0, 2, 0)  # A[2,0:2] * B[2:4,0:2] -> C[2,0:2]
    add_schoolbook_block(2, 2, 2, 2)  # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
