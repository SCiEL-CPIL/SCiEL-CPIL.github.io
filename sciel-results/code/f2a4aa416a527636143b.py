import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will build a rank 98 decomposition
    # Based on rank 49 for 4x4x4, duplicated for 4x4x8
    
    # First, define the rank 7 decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    # Indices: i*2+j for U, j*2+k for V, k*2+i for W
    
    U_base = [[0]*7 for _ in range(4)]
    V_base = [[0]*7 for _ in range(4)]
    W_base = [[0]*7 for _ in range(4)]
    
    # M1
    U_base[0][0] = 1 # A00
    U_base[3][0] = 1 # A11
    V_base[0][0] = 1 # B00
    V_base[3][0] = 1 # B11
    W_base[0][0] = 1 # C00 (k=0, i=0)
    W_base[3][0] = 1 # C11 (k=1, i=1)
    
    # M2
    U_base[2][1] = 1 # A10
    U_base[3][1] = 1 # A11
    V_base[0][1] = 1 # B00
    W_base[2][1] = 1 # C10 (k=0, i=1)
    W_base[3][1] = 1 # C11 (k=1, i=1)
    
    # M3
    U_base[0][2] = 1 # A00
    V_base[2][2] = 1 # B10
    V_base[3][2] = -1 # B11
    W_base[1][2] = 1 # C01 (k=1, i=0)
    W_base[3][2] = 1 # C11 (k=1, i=1)
    
    # M4
    U_base[3][3] = 1 # A11
    V_base[1][3] = 1 # B01
    V_base[0][3] = -1 # B00
    W_base[0][3] = 1 # C00
    W_base[2][3] = 1 # C10
    
    # M5
    U_base[0][4] = 1 # A00
    U_base[2][4] = 1 # A10
    V_base[3][4] = 1 # B11
    W_base[0][4] = -1 # C00
    W_base[1][4] = 1 # C01
    
    # M6
    U_base[3][5] = 1 # A11
    U_base[0][5] = -1 # A00
    V_base[0][5] = 1 # B00
    V_base[2][5] = 1 # B10
    W_base[3][5] = 1 # C11
    
    # M7
    U_base[1][6] = 1 # A01
    U_base[3][6] = -1 # A11
    V_base[3][6] = 1 # B11
    V_base[2][6] = 1 # B10
    W_base[0][6] = 1 # C00

    # Now build the rank 49 decomposition for 4x4x4
    # We have 7 block-level Strassen terms.
    # Each term s has coefficients for the 4 blocks of A and 4 blocks of B.
    # We need to map these to the 16 entries of A and 16 entries of B.
    
    # Block indices:
    # A blocks: 00, 01, 10, 11
    # A00 corresponds to A[0:2, 0:2] -> indices 0,1,2,3 in U (4x4)
    # A01 corresponds to A[0:2, 2:4] -> indices 4,5,6,7
    # A10 corresponds to A[2:4, 0:2] -> indices 8,9,10,11
    # A11 corresponds to A[2:4, 2:4] -> indices 12,13,14,15
    
    # Similarly for B.
    
    # We will create U_44, V_44, W_44 of size 16x49, 16x49, 16x49
    
    U_44 = [[0]*49 for _ in range(16)]
    V_44 = [[0]*49 for _ in range(16)]
    W_44 = [[0]*49 for _ in range(16)]
    
    # The 7 block-level Strassen terms are the same as M1..M7 above.
    # For each term s (0..6), we have u_base, v_base, w_base vectors.
    # These vectors are over the 4 block entries.
    # We need to expand each block entry into 4 scalar entries.
    
    # For a given term s, and for each base rank-1 term r (0..6):
    # The contribution is a product of (sum of A blocks) and (sum of B blocks).
    # This is a 2x2x2 multiplication of the blocks.
    # So we use the base decomposition U_base, V_base, W_base.
    
    # Let's iterate over the 7 block-level terms (s)
    for s in range(7):
        # For each s, we have a "meta" rank-1 term.
        # But we need to decompose the block multiplication into 7 scalar multiplications.
        # So we iterate over the 7 base terms (r)
        for r in range(7):
            col = s * 7 + r
            
            # The U part for this column is the expansion of the A-block combination for term s.
            # The A-block combination is given by U_base[:, s].
            # U_base[0, s] is coeff for A00 block.
            # U_base[1, s] is coeff for A01 block.
            # U_base[2, s] is coeff for A10 block.
            # U_base[3, s] is coeff for A11 block.
            
            # Wait, U_base is defined for the 2x2x2 tensor.
            # In the 2x2x2 tensor, the indices are i,j.
            # U_base[i*2+j, s].
            # So U_base[0, s] is A00.
            # U_base[1, s] is A01.
            # U_base[2, s] is A10.
            # U_base[3, s] is A11.
            
            # For the 4x4x4 tensor, A00 block corresponds to 4 entries.
            # The coefficient for each entry in the block is the same.
            
            # So for each block index b (0..3) in U_base:
            # coeff = U_base[b, s]
            # If coeff != 0:
            #   For each of the 4 entries in that block:
            #     U_44[entry, col] += coeff * (something from base r?)
            
            # No, this is not right.
            # The term is (A_comb) * (B_comb).
            # A_comb is a 2x2 matrix of blocks.
            # B_comb is a 2x2 matrix of blocks.
            # Their product is a 2x2 matrix of blocks.
            # This product is decomposed into 7 rank-1 terms using the base decomposition.
            
            # So for a fixed s, we have a 2x2x2 multiplication of the blocks.
            # We use the base decomposition (U_base, V_base, W_base) for this.
            # The base decomposition has 7 terms.
            # For each base term r:
            #   u_r is a vector of 4 coeffs for the A blocks.
            #   v_r is a vector of 4 coeffs for the B blocks.
            #   w_r is a vector of 4 coeffs for the C blocks.
            
            # The u_r comes from U_base[:, s]? No.
            # U_base is the decomposition of the *identity* multiplication.
            # Here, the inputs are the *combinations* from term s.
            
            # Let's re-evaluate.
            # Term s is $M_s = P_s Q_s$.
            # $P_s$ is a linear combination of A blocks.
            # $Q_s$ is a linear combination of B blocks.
            # $P_s$ is a 2x2 matrix. $Q_s$ is a 2x2 matrix.
            # $M_s = P_s Q_s$.
            # We want to decompose $P_s Q_s$ into rank-1 terms.
            # We can use the fact that $P_s Q_s = \sum_{r=1}^7 u_r(P_s) v_r(Q_s) w_r$?
            # No, $P_s Q_s$ is a specific matrix.
            # We can write $P_s Q_s = \sum_{r=1}^7 (P_s u_r) (v_r^T Q_s) w_r$?
            # This is getting complicated.
            
            # Simpler approach:
            # Just generate the 49 terms by "expanding" the block Strassen.
            # Each block Strassen term $M_s$ is a product of two 2x2 block matrices.
            # Let $P_s$ be the 2x2 matrix of A-block coefficients.
            # Let $Q_s$ be the 2x2 matrix of B-block coefficients.
            # Then $M_s = P_s Q_s$.
            # We can decompose $P_s Q_s$ using the rank-7 decomposition of $2 \times 2 \times 2$.
            # The rank-7 decomposition says:
            # $X Y = \sum_{r=1}^7 (X u_r) (v_r Y) w_r$?
            # No.
            # The decomposition is $T_{i,j,k} = \sum_r u_r(i,j) v_r(j,k) w_r(k,i)$.
            # Here $X, Y$ are the inputs.
            # So $M_s = \sum_{r=1}^7 ( \sum_{i,j} X_{ij} U_{base}[i*2+j, r] ) ( \sum_{j,k} Y_{jk} V_{base}[j*2+k, r] ) W_{base}[k*2+i, r]$?
            # This is not a rank-1 term.
            
            # Actually, we can just use the linearity.
            # The tensor for $4 \times 4 \times 4$ is the sum of 7 tensors, each corresponding to a Strassen term.
            # Each Strassen term is a $2 \times 2 \times 2$ multiplication of the blocks.
            # So each Strassen term can be decomposed into 7 rank-1 terms of the $2 \times 2 \times 2$ tensor.
            # The total is 49 terms.
            
            # To construct the 49 terms:
            # For each s in 0..6:
            #   For each r in 0..6:
            #     The term is defined by the coefficients in U_base, V_base, W_base for index r.
            #     BUT, the inputs to this $2 \times 2 \times 2$ multiplication are the *combinations* from term s.
            
            # This means we need to "pre-multiply" the U_base and V_base by the combination matrices.
            
            # Let's define the combination matrices for the 7 Strassen terms.
            # A_comb[s] is a 2x2 matrix of coefficients for A blocks.
            # B_comb[s] is a 2x2 matrix of coefficients for B blocks.
            
            A_comb = [
                [[1, 0], [0, 1]], # M1: A00+A11
                [[0, 0], [1, 1]], # M2: A10+A11
                [[1, 0], [0, 0]], # M3: A00
                [[0, 0], [0, 1]], # M4: A11
                [[1, 0], [1, 0]], # M5: A00+A10
                [[-1, 0], [0, 1]], # M6: A11-A00
                [[0, 1], [0, -1]]  # M7: A01-A11
            ]
            
            B_comb = [
                [[1, 0], [0, 1]], # M1: B00+B11
                [[1, 0], [0, 0]], # M2: B00
                [[0, 1], [0, -1]], # M3: B10-B11
                [[-1, 1], [0, 0]], # M4: B01-B00
                [[0, 0], [0, 1]], # M5: B11
                [[1, 0], [1, 0]], # M6: B00+B10
                [[0, 0], [1, 1]]  # M7: B11+B10
            ]
            
            # For each s, the term is $P_s Q_s$.
            # $P_s$ is a 2x2 matrix. $Q_s$ is a 2x2 matrix.
            # We use the base decomposition to decompose $P_s Q_s$.
            # The base decomposition is $T = \sum_r u_r \otimes v_r \otimes w_r$.
            # So $P_s Q_s = \sum_r (P_s u_r) (v_r Q_s) w_r$?
            # No.
            # The term is a bilinear form.
            # Let's just use the explicit expansion.
            
            # For a fixed s, we have $P_s$ and $Q_s$.
            # We want to express $P_s Q_s$ as a sum of 7 rank-1 terms.
            # We can use the fact that any $2 \times 2$ matrix product can be decomposed.
            # But we want to use the *same* 7 terms for all s?
            # No, the terms depend on s.
            
            # Actually, we can just generate the 49 terms by iterating s and r.
            # For each s, we have a "virtual" $2 \times 2 \times 2$ multiplication.
            # The inputs are the blocks.
            # The decomposition of this virtual multiplication is given by the base U_base, V_base, W_base.
            # BUT, the "blocks" are linear combinations of the actual A, B entries.
            
            # So, for term (s, r):
            # U part: The coefficient for each A entry.
            # The A entry is in a block.
            # The block has a coefficient from A_comb[s].
            # AND the block entry has a coefficient from U_base[r].
            
            # So U_44[entry, col] = A_comb[s][block_i][block_j] * U_base[local_i*2+local_j, r]
            
            # Similarly for V.
            
            # And W part:
            # The output is a block.
            # The block has a coefficient from W_base[r].
            # AND the block is placed at a specific location in the 4x4 output.
            
            # This seems correct.
            
            # Let's implement this.
            
            for s in range(7):
                for r in range(7):
                    col = s * 7 + r
                    
                    # U part
                    for bi in range(2): # block row
                        for bj in range(2): # block col
                            coeff_a = A_comb[s][bi][bj]
                            if coeff_a == 0: continue
                            
                            # Map block (bi, bj) to 4 entries in U_44
                            # Block (bi, bj) corresponds to A[bi*2:bi*2+2, bj*2:bj*2+2]
                            # U_44 index: i*4+j
                            # i is global row, j is global col.
                            for li in range(2):
                                for lj in range(2):
                                    i = bi*2 + li
                                    j = bj*2 + lj
                                    idx_u = i*4 + j
                                    
                                    # U_base index for local (li, lj)
                                    idx_base = li*2 + lj
                                    coeff_base = U_base[idx_base][r]
                                    
                                    if coeff_base != 0:
                                        U_44[idx_u][col] += coeff_a * coeff_base
                                        
                    # V part
                    for bi in range(2): # block row (j)
                        for bj in range(2): # block col (k)
                            coeff_b = B_comb[s][bi][bj]
                            if coeff_b == 0: continue
                            
                            for li in range(2):
                                for lj in range(2):
                                    j = bi*2 + li
                                    k = bj*2 + lj
                                    idx_v = j*4 + k
                                    
                                    idx_base = li*2 + lj
                                    coeff_base = V_base[idx_base][r]
                                    
                                    if coeff_base != 0:
                                        V_44[idx_v][col] += coeff_b * coeff_base
                                        
                    # W part
                    # W_base is for 2x2x2.
                    # W_base[k*2+i, r]
                    # The output is a 2x2 block.
                    # The block location is determined by the Strassen term s?
                    # No, the Strassen term s produces a 2x2 matrix of blocks.
                    # So the output is the whole 4x4 matrix?
                    # No, the Strassen algorithm computes the 4 block outputs.
                    # The 7 terms M1..M7 are combined to get the 4 blocks.
                    # So the W part must map the 2x2 block output to the 4x4 output.
                    
                    # Wait, the base decomposition W_base gives the contribution to the 2x2 output.
                    # So for each (k,i) in the 2x2 output, we have a coefficient.
                    # This (k,i) corresponds to a block in the 4x4 output.
                    
                    for bk in range(2): # block row (k)
                        for bi in range(2): # block col (i)
                            # W_base index: bk*2 + bi
                            idx_base = bk*2 + bi
                            coeff_base = W_base[idx_base][r]
                            
                            if coeff_base == 0: continue
                            
                            # Map block (bk, bi) to 4 entries in W_44
                            # W_44 index: k*4 + i
                            for lk in range(2):
                                for li in range(2):
                                    k = bk*2 + lk
                                    i = bi*2 + li
                                    idx_w = k*4 + i
                                    
                                    W_44[idx_w][col] += coeff_base

    # Now we have U_44, V_44, W_44 for 4x4x4.
    # We need 4x4x8.
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # C is 4x8. Split into C1 (4x4) and C2 (4x4).
    # C1 = A B1. C2 = A B2.
    # We use the same decomposition for both.
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    for r in range(49):
        # Copy to first half
        for i in range(16):
            U[i][r] = U_44[i][r]
            U[i][r+49] = U_44[i][r]
            
        # V first half (B1)
        for j in range(16):
            V[j][r] = V_44[j][r]
        # V second half (B2)
        for j in range(16):
            V[j+16][r+49] = V_44[j][r]
            
        # W first half (C1)
        for k in range(16):
            W[k][r] = W_44[k][r]
        # W second half (C2)
        for k in range(16):
            W[k+16][r+49] = W_44[k][r]
            
    return {"U": U, "V": V, "W": W}
