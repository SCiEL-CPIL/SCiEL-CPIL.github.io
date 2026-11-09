"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for 4x4x4, which has rank 49.
    # For 4x4x8, we use two such decompositions, rank 98.
    
    # Step 1: Construct rank-7 decomposition for 2x2x2 (Strassen)
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen definitions for 2x2
    # M1 = (a+d)(e+h)
    # a:0,0 d:1,1 e:0,0 h:1,1
    # M2 = (c+d)e
    # c:1,0 d:1,1 e:0,0
    # M3 = a(f-h)
    # a:0,0 f:0,1 h:1,1
    # M4 = d(g-e)
    # d:1,1 g:1,0 e:0,0
    # M5 = (a+b)h
    # a:0,0 b:0,1 h:1,1
    # M6 = (c-a)(e+f)
    # c:1,0 a:0,0 e:0,0 f:0,1
    # M7 = (b-d)(f+h)
    # b:0,1 d:1,1 f:0,1 h:1,1
    
    # We need to map these to U, V, W indices.
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    # M1: (A00+A11)(B00+B11)
    # U: A00 (0), A11 (3) -> U[0]=1, U[3]=1
    # V: B00 (0), B11 (3) -> V[0]=1, V[3]=1
    # W: C00 (0), C11 (3) -> W[0]=1, W[3]=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00
    # U: A10 (2), A11 (3) -> U[2]=1, U[3]=1
    # V: B00 (0) -> V[0]=1
    # W: C10 (2), C11 (3) -> W[2]=1, W[3]=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3: A00(B01-B11)
    # U: A00 (0) -> U[0]=1
    # V: B01 (1), B11 (3) -> V[1]=1, V[3]=-1
    # W: C01 (1), C11 (3) -> W[1]=1, W[3]=-1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4: A11(B10-B00)
    # U: A11 (3) -> U[3]=1
    # V: B10 (2), B00 (0) -> V[2]=1, V[0]=-1
    # W: C10 (2), C00 (0) -> W[2]=1, W[0]=-1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[0, 3] = -1
    
    # M5: (A00+A01)B11
    # U: A00 (0), A01 (1) -> U[0]=1, U[1]=1
    # V: B11 (3) -> V[3]=1
    # W: C01 (1), C11 (3) -> W[1]=1, W[3]=1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: A10 (2), A00 (0) -> U[2]=1, U[0]=-1
    # V: B00 (0), B01 (1) -> V[0]=1, V[1]=1
    # W: C10 (2), C00 (0) -> W[2]=1, W[0]=-1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[0, 5] = -1
    
    # M7: (A01-A11)(B10+B11)
    # U: A01 (1), A11 (3) -> U[1]=1, U[3]=-1
    # V: B10 (2), B11 (3) -> V[2]=1, V[3]=1
    # W: C01 (1), C11 (3) -> W[1]=1, W[3]=-1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1; W2[3, 6] = -1

    # Step 2: Construct rank-49 decomposition for 4x4x4
    # U4 (16x49), V4 (16x49), W4 (16x49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Strassen 4x4 definitions
    # A blocks: A00(0-3), A01(4-7), A10(8-11), A11(12-15)
    # B blocks: B00(0-3), B01(4-7), B10(8-11), B11(12-15)
    # C blocks: C00(0-3), C01(4-7), C10(8-11), C11(12-15)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map the 2x2 U/V/W patterns to the 4x4 blocks.
    # For a given Strassen product g (0-6), it involves specific A and B blocks.
    # The "base" 2x2 multiplication inside that product uses the U2/V2/W2 patterns.
    
    # Let's define the block indices for each Strassen product.
    # A_block_indices[g] = list of (block_index, coeff)
    # B_block_indices[g] = list of (block_index, coeff)
    # C_block_indices[g] = list of (block_index, coeff)
    
    # Block indices in 4x4 matrix (flattened 16x16? No, U is 16xR)
    # U rows: i*4+j.
    # Block 0 (A00): rows 0,1,4,5 (i=0,1; j=0,1) -> 0,1,4,5
    # Block 1 (A01): rows 0,1,6,7 (i=0,1; j=2,3) -> 0,1,6,7
    # Block 2 (A10): rows 2,3,4,5 (i=2,3; j=0,1) -> 2,3,4,5
    # Block 3 (A11): rows 2,3,6,7 (i=2,3; j=2,3) -> 2,3,6,7
    
    # V rows: j*4+k.
    # Block 0 (B00): cols 0,1,4,5 (j=0,1; k=0,1) -> 0,1,4,5
    # Block 1 (B01): cols 0,1,6,7 (j=0,1; k=2,3) -> 0,1,6,7
    # Block 2 (B10): cols 2,3,4,5 (j=2,3; k=0,1) -> 2,3,4,5
    # Block 3 (B11): cols 2,3,6,7 (j=2,3; k=2,3) -> 2,3,6,7
    
    # W rows: k*4+i.
    # Block 0 (C00): rows 0,1,4,5 (k=0,1; i=0,1) -> 0,1,4,5
    # Block 1 (C01): rows 0,1,6,7 (k=0,1; i=2,3) -> 0,1,6,7
    # Block 2 (C10): rows 2,3,4,5 (k=2,3; i=0,1) -> 2,3,4,5
    # Block 3 (C11): rows 2,3,6,7 (k=2,3; i=2,3) -> 2,3,6,7

    # Helper to get row indices for a block
    def get_block_rows(block_idx, matrix_type):
        # matrix_type: 'U', 'V', 'W'
        # U: i*4+j. Block (bi, bj) where bi, bj in 0,1.
        # Block 0: bi=0, bj=0. Rows: i in 0,1; j in 0,1.
        # Block 1: bi=0, bj=1. Rows: i in 0,1; j in 2,3.
        # Block 2: bi=1, bj=0. Rows: i in 2,3; j in 0,1.
        # Block 3: bi=1, bj=1. Rows: i in 2,3; j in 2,3.
        
        bi = block_idx // 2
        bj = block_idx % 2
        
        if matrix_type == 'U':
            rows = []
            for i in range(2*bi, 2*bi+2):
                for j in range(2*bj, 2*bj+2):
                    rows.append(i*4+j)
            return rows
        elif matrix_type == 'V':
            # V: j*4+k. Block (bj, bk) where bj, bk in 0,1.
            # Block 0: bj=0, bk=0. Rows: j in 0,1; k in 0,1.
            # Block 1: bj=0, bk=1. Rows: j in 0,1; k in 2,3.
            # Block 2: bj=1, bk=0. Rows: j in 2,3; k in 0,1.
            # Block 3: bj=1, bk=1. Rows: j in 2,3; k in 2,3.
            rows = []
            for j in range(2*bi, 2*bi+2): # bi is actually bj here
                for k in range(2*bj, 2*bj+2): # bj is actually bk here
                    rows.append(j*4+k)
            return rows
        elif matrix_type == 'W':
            # W: k*4+i. Block (bk, bi) where bk, bi in 0,1.
            # Block 0: bk=0, bi=0. Rows: k in 0,1; i in 0,1.
            # Block 1: bk=0, bi=1. Rows: k in 0,1; i in 2,3.
            # Block 2: bk=1, bi=0. Rows: k in 2,3; i in 0,1.
            # Block 3: bk=1, bi=1. Rows: k in 2,3; i in 2,3.
            rows = []
            for k in range(2*bi, 2*bi+2): # bi is bk
                for i in range(2*bj, 2*bj+2): # bj is bi
                    rows.append(k*4+i)
            return rows

    # Strassen product definitions:
    # g: (A_blocks, B_blocks, C_blocks)
    # Each is list of (block_idx, coeff)
    strassen_defs = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, -1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (0, -1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (0, -1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(1, 1), (3, -1)])
    ]

    for g in range(7):
        a_blocks, b_blocks, c_blocks = strassen_defs[g]
        
        for b_idx in range(7):
            col = g * 7 + b_idx
            
            # U
            for a_block, coeff in a_blocks:
                rows = get_block_rows(a_block, 'U')
                for r in rows:
                    # The base U2 pattern for column b_idx
                    # U2[r_2x2, b_idx]
                    # We need to map the 2x2 index to the 4x4 block row.
                    # The block row r corresponds to a specific i,j in the 4x4 matrix.
                    # The 2x2 sub-matrix within that block has local indices i_loc, j_loc.
                    # The U2 index is i_loc*2 + j_loc.
                    # We need to extract i_loc, j_loc from r and the block definition.
                    pass
            
            # This mapping is tricky. Let's simplify.
            # Instead of complex mapping, we can just build the full 49x49? No.
            # Let's use the property that Strassen is linear.
            
    # Given the complexity and risk of error in manual mapping, 
    # and the fact that rank 98 is a solid improvement over 128,
    # but rank 128 is the seed, let's ensure the code is correct.
    
    # Actually, a simpler and safer bet for "rank minimal" in a coding contest context
    # where explicit matrices are required is often to use known small decompositions.
    # But Strassen is the most reliable "algorithmic" reduction.
    
    # Let's try to just return the schoolbook if the Strassen construction is too risky.
    # But 128 is the seed. We MUST beat it.
    # Rank 98 is a valid target.
    
    # Let's refine the Strassen construction to be correct.
    
    # Re-implementing U4, V4, W4 construction carefully.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # U2, V2, W2 are defined above.
    
    for g in range(7):
        a_blocks, b_blocks, c_blocks = strassen_defs[g]
        for b_idx in range(7):
            col = g * 7 + b_idx
            
            # U part
            for a_block, coeff in a_blocks:
                # Get the 4 rows of this block in U
                rows = get_block_rows(a_block, 'U')
                # For each row in the block, we need to add coeff * U2[local_idx, b_idx]
                # local_idx corresponds to the position within the 2x2 block.
                # U2 index is i_loc*2 + j_loc.
                # In the 4x4 block, the local i_loc, j_loc are 0 or 1.
                # The global row r in U4 is i*4+j.
                # For block (bi, bj), i = 2*bi + i_loc, j = 2*bj + j_loc.
                # So r = (2*bi + i_loc)*4 + (2*bj + j_loc).
                # We can iterate i_loc, j_loc from 0 to 1.
                bi = a_block // 2
                bj = a_block % 2
                for i_loc in range(2):
                    for j_loc in range(2):
                        r = (2*bi + i_loc)*4 + (2*bj + j_loc)
                        local_idx = i_loc*2 + j_loc
                        U4[r, col] += coeff * U2[local_idx, b_idx]
            
            # V part
            for b_block, coeff in b_blocks:
                bi = b_block // 2
                bj = b_block % 2
                for j_loc in range(2):
                    for k_loc in range(2):
                        r = (2*bi + j_loc)*4 + (2*bj + k_loc)
                        local_idx = j_loc*2 + k_loc
                        V4[r, col] += coeff * V2[local_idx, b_idx]
            
            # W part
            for c_block, coeff in c_blocks:
                bi = c_block // 2
                bj = c_block % 2
                for k_loc in range(2):
                    for i_loc in range(2):
                        r = (2*bi + k_loc)*4 + (2*bj + i_loc)
                        local_idx = k_loc*2 + i_loc
                        W4[r, col] += coeff * W2[local_idx, b_idx]

    # Now construct U, V, W for 4x4x8
    # U (32x98), V (32x98), W (32x98)
    U = np.zeros((16, 98), dtype=int) # n*m = 4*4 = 16
    V = np.zeros((32, 98), dtype=int) # m*p = 4*8 = 32
    W = np.zeros((32, 98), dtype=int) # p*n = 8*4 = 32
    
    # First half (B0-3, C0-3)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4 # B0-3 rows 0-15
    W[0:16, 0:49] = W4 # C0-3 rows 0-15
    
    # Second half (B4-7, C4-7)
    U[:, 49:98] = U4
    # B4-7 corresponds to rows 16-31 in V
    # V4 rows are 0-15. We need to map them to 16-31.
    V[16:32, 49:98] = V4
    # C4-7 corresponds to rows 16-31 in W
    # W4 rows are 0-15. We need to map them to 16-31.
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
