"""Matrix multiplication 2,3,10 with Strassen-based rank reduction."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We will use a block decomposition.
    # Split p=10 into five blocks of size 2.
    # For each block of size 2, we multiply a 2x3 (A) by a 3x2 (B_block).
    # This is T_{2,3,2}.
    # We can further decompose T_{2,3,2}.
    # Split m=3 into 1+2.
    # T_{2,3,2} = T_{2,1,2} \oplus T_{2,2,2}
    # T_{2,1,2} has rank 4 (since 2*1*2=4, and it's minimal? 
    #   Actually T_{n,m,p} rank is at least max(nm, mp, pn). 
    #   max(2, 4, 2) = 4. So rank 4 is minimal.)
    #   Let's verify: U is 2x4, V is 2x4, W is 4x4.
    #   Indices: U[i, c], V[j, c] (j in 0..1), W[k*n+i, c] (k in 0..1).
    #   Actually, for T_{2,1,2}: A is 2x1, B is 1x2. Output C is 2x2.
    #   C_{i,k} = A_{i,0} B_{0,k}.
    #   Rank is 4.
    #   
    # T_{2,2,2} has rank 7 (Strassen).
    # So T_{2,3,2} has rank 4 + 7 = 11.
    # Wait, is this minimal? 
    #   T_{2,3,2} dimensions: A 2x3, B 3x2, C 2x2.
    #   Lower bound: max(6, 6, 4) = 6.
    #   Known rank: 11? Or can we do better?
    #   Let's check if we can combine the 4 and 7.
    #   Actually, the direct sum gives 11.
    #   Is there a rank 10 decomposition?
    #   Let's assume 11 for now, as it's derived from known minimal ranks of sub-tensors.
    
    # Total rank for 5 blocks: 5 * 11 = 55.
    # This matches the known improvement from 60 to 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    # We need to construct the rank 11 decomposition of T_{2,3,2}.
    # A is 2x3, B is 3x2, C is 2x2.
    # Split A into A_left (2x1) and A_right (2x2).
    # Split B into B_left (1x2) and B_right (2x2).
    # C = A_left * B_left + A_right * B_right.
    #
    # Part 1: T_{2,1,2} (rank 4)
    #   A_left is 2x1: indices i (0..1), j=0.
    #   B_left is 1x2: indices j=0, k (0..1).
    #   C is 2x2: indices i (0..1), k (0..1).
    #   U indices: i*m + j = i*3 + 0. (0, 3)
    #   V indices: j*p + k = 0*2 + k = k. (0, 1)  <- Note: p here is 2 (block size), but global p is 10.
    #      Wait, V is indexed by j (global 0..2) and k (global 0..9).
    #      In the block, j_local (0..2) and k_local (0..1).
    #      Global j = j_local. Global k = block_offset + k_local.
    #   W indices: k*n + i. (k_global is determined by block)
    
    # Let's define the local indices for the block.
    # Block b (0..4).
    # Global k = 2*b + k_local, where k_local in 0..1.
    # Global i in 0..1.
    # Global j in 0..2.
    
    # U[i, c] is non-zero for i in 0..1. Row index in U: i*3 + j.
    # V[j, c] is non-zero for j in 0..2. Row index in V: j*2 + k_local? NO.
    # V row index is j*p + k. Here p=10. So j*10 + k.
    # W row index is k*n + i. Here n=2. So k*2 + i.
    
    # Let's hardcode the 11 terms for one block, then replicate.
    
    # Term 1-4: T_{2,1,2}
    # C_{i,k} += A_{i,0} * B_{0,k}
    # This is a rank 4 tensor.
    # We can decompose it as sum_{i,k} E_{i,0} \otimes E_{0,k} \otimes E_{k,i}
    # Where E_{i,0} is unit vector in U space (row i*3),
    # E_{0,k} is unit vector in V space (row 0*10 + k),
    # E_{k,i} is unit vector in W space (row k*2 + i).
    
    # Term 5-11: T_{2,2,2} using Strassen.
    # A_right is 2x2: rows i (0..1), cols j (1..2).
    # B_right is 2x2: rows j (1..2), cols k (0..1).
    # C contribution is 2x2: rows i, cols k.
    
    # Strassen for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4 + M6
    
    # Indices for A_right:
    # a11 corresponds to A[0,1] -> U row 0*3+1 = 1
    # a12 corresponds to A[0,2] -> U row 0*3+2 = 2
    # a21 corresponds to A[1,1] -> U row 1*3+1 = 4
    # a22 corresponds to A[1,2] -> U row 1*3+2 = 5
    
    # Indices for B_right:
    # b11 corresponds to B[1,0] (local k=0) -> V row 1*10 + k_global
    # b12 corresponds to B[1,1] (local k=1) -> V row 1*10 + k_global+1
    # b21 corresponds to B[2,0] (local k=0) -> V row 2*10 + k_global
    # b22 corresponds to B[2,1] (local k=1) -> V row 2*10 + k_global+1
    
    # Indices for C:
    # C[0,0] -> W row 0*2+0 = 0
    # C[0,1] -> W row 1*2+0 = 2
    # C[1,0] -> W row 0*2+1 = 1
    # C[1,1] -> W row 1*2+1 = 3
    
    # Note: W index is k*n+i.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    
    # Let's define the contributions for each Strassen term.
    # We need to be careful with signs.
    
    # For a specific block b, k_base = 2*b.
    
    # We will build the U, V, W for one block (11 columns), then shift for other blocks.
    
    U_block = [[0]*11 for _ in range(6)]
    V_block = [[0]*11 for _ in range(30)]
    W_block = [[0]*11 for _ in range(20)]
    
    # Terms 0-3: T_{2,1,2}
    # c=0: i=0, k=0. U[0], V[k_base], W[0]
    U_block[0][0] = 1
    V_block[0*10 + 0][0] = 1  # k_local=0
    W_block[0*2 + 0][0] = 1
    
    # c=1: i=0, k=1. U[0], V[k_base+1], W[2]
    U_block[0][1] = 1
    V_block[0*10 + 1][1] = 1  # k_local=1
    W_block[1*2 + 0][1] = 1
    
    # c=2: i=1, k=0. U[3], V[k_base], W[1]
    U_block[3][2] = 1
    V_block[0*10 + 0][2] = 1
    W_block[0*2 + 1][2] = 1
    
    # c=3: i=1, k=1. U[3], V[k_base+1], W[3]
    U_block[3][3] = 1
    V_block[0*10 + 1][3] = 1
    W_block[1*2 + 1][3] = 1
    
    # Terms 4-10: Strassen
    # M1 = (a11+a22)(b11+b22)
    # a11: U[1], a22: U[5]
    # b11: V[1*10+k], b22: V[2*10+k+1]
    # C: C11(W[0]) += 1, C22(W[3]) += 1
    
    # c=4: M1
    U_block[1][4] = 1
    U_block[5][4] = 1
    V_block[1*10 + 0][4] = 1
    V_block[2*10 + 1][4] = 1
    W_block[0][4] = 1
    W_block[3][4] = 1
    
    # M2 = (a21+a22)b11
    # a21: U[4], a22: U[5]
    # b11: V[1*10+k]
    # C: C21(W[1]) += 1
    
    # c=5: M2
    U_block[4][5] = 1
    U_block[5][5] = 1
    V_block[1*10 + 0][5] = 1
    W_block[1][5] = 1
    
    # M3 = a11(b12-b22)
    # a11: U[1]
    # b12: V[1*10+k+1], b22: V[2*10+k+1]
    # C: C12(W[2]) += 1, C22(W[3]) -= 1
    
    # c=6: M3
    U_block[1][6] = 1
    V_block[1*10 + 1][6] = 1
    V_block[2*10 + 1][6] = -1
    W_block[2][6] = 1
    W_block[3][6] = -1
    
    # M4 = a22(b21-b11)
    # a22: U[5]
    # b21: V[2*10+k], b11: V[1*10+k]
    # C: C11(W[0]) -= 1, C22(W[3]) += 1
    
    # c=7: M4
    U_block[5][7] = 1
    V_block[2*10 + 0][7] = 1
    V_block[1*10 + 0][7] = -1
    W_block[0][7] = -1
    W_block[3][7] = 1
    
    # M5 = (a11+a12)b22
    # a11: U[1], a12: U[2]
    # b22: V[2*10+k+1]
    # C: C11(W[0]) += 1
    
    # c=8: M5
    U_block[1][8] = 1
    U_block[2][8] = 1
    V_block[2*10 + 1][8] = 1
    W_block[0][8] = 1
    
    # M6 = (a21-a11)(b11+b12)
    # a21: U[4], a11: U[1]
    # b11: V[1*10+k], b12: V[1*10+k+1]
    # C: C21(W[1]) += 1, C22(W[3]) += 1
    
    # c=9: M6
    U_block[4][9] = 1
    U_block[1][9] = -1
    V_block[1*10 + 0][9] = 1
    V_block[1*10 + 1][9] = 1
    W_block[1][9] = 1
    W_block[3][9] = 1
    
    # M7 = (a12-a22)(b21+b22)
    # a12: U[2], a22: U[5]
    # b21: V[2*10+k], b22: V[2*10+k+1]
    # C: C12(W[2]) += 1, C11(W[0]) -= 1
    
    # c=10: M7
    U_block[2][10] = 1
    U_block[5][10] = -1
    V_block[2*10 + 0][10] = 1
    V_block[2*10 + 1][10] = 1
    W_block[2][10] = 1
    W_block[0][10] = -1
    
    # Now, replicate this block for all 5 blocks.
    # For block b, k_base = 2*b.
    # We need to shift the V indices by k_base.
    # U and W indices are unchanged.
    
    for b in range(5):
        offset = b * 11
        k_base = 2 * b
        
        for c in range(11):
            # Copy U
            for i in range(6):
                if U_block[i][c] != 0:
                    U[i][offset + c] = U_block[i][c]
            
            # Copy V (with shift)
            for j in range(30):
                if V_block[j][c] != 0:
                    # j is j_global*10 + k_local
                    # We need to map k_local to k_global = k_base + k_local
                    # But V_block was constructed with k_local in 0..1.
                    # Actually, V_block indices were j*10 + k_local.
                    # We need to find which V_block row corresponds to which.
                    # It's easier to just re-iterate over the non-zeros of V_block.
                    pass
            
            # Let's do it more carefully.
            # V_block has non-zeros at specific rows.
            # For each non-zero in V_block at row j_row, col c:
            # j_row = j*10 + k_local.
            # The global index is j*10 + (k_base + k_local).
            # So we can just add k_base to the index if we store it as j*10+k?
            # No, because j is the first part.
            # Let's just iterate through V_block rows.
            
            for i_v in range(30):
                if V_block[i_v][c] != 0:
                    # i_v = j*10 + k_local
                    j = i_v // 10
                    k_local = i_v % 10
                    k_global = k_base + k_local
                    i_v_global = j*10 + k_global
                    V[i_v_global][offset + c] = V_block[i_v][c]
            
            # Copy W
            for i_w in range(20):
                if W_block[i_w][c] != 0:
                    W[i_w][offset + c] = W_block[i_w][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
