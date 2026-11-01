"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strassen's 7 products for 2x2 matrices
    # For A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    
    # Each 2x2 product has rank 7. We'll represent each as 7 terms.
    # For a 2x2 product of matrices X (2x2) and Y (2x2) giving Z (2x2):
    # X indices: (0,0)=x00, (0,1)=x01, (1,0)=x10, (1,1)=x11
    # Y indices: (0,0)=y00, (0,1)=y01, (1,0)=y10, (1,1)=y11
    # Z indices: (0,0)=z00, (0,1)=z01, (1,0)=z10, (1,1)=z11
    
    # The 7 Strassen terms for one 2x2 product:
    # term 0: (x00+x11)*(y00+y11) -> contributes to z00 and z11
    # term 1: (x10+x11)*y00 -> contributes to z10 and z11
    # term 2: x00*(y01-y11) -> contributes to z01 and z11
    # term 3: x11*(y10-y00) -> contributes to z00 and z10
    # term 4: (x00+x01)*y11 -> contributes to z00 and z01
    # term 5: (x10-x00)*(y00+y01) -> contributes to z11
    # term 6: (x01-x11)*(y10+y11) -> contributes to z00 and z10
    
    # Let me be more careful with the standard Strassen decomposition.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For a single 2x2 product, the 7 terms contribute to the 4 output elements.
    # I'll represent each term as (u_entries, v_entries, w_entries) where each is a dict from index to coefficient.
    
    def strassen_2x2_terms():
        """Return list of 7 terms, each with U, V, W contributions for a 2x2 product."""
        # X indices: 0=x00, 1=x01, 2=x10, 3=x11
        # Y indices: 0=y00, 1=y01, 2=y10, 3=y11
        # Z indices: 0=z00, 1=z01, 2=z10, 3=z11
        
        terms = []
        # M1 = (x00+x11)(y00+y11)
        # U: x00->1, x11->1; V: y00->1, y11->1; W: z00->1, z11->1
        terms.append({
            'u': {0: 1, 3: 1},
            'v': {0: 1, 3: 1},
            'w': {0: 1, 3: 1}
        })
        # M2 = (x10+x11)y00
        # U: x10->1, x11->1; V: y00->1; W: z10->1, z11->(-1)
        terms.append({
            'u': {2: 1, 3: 1},
            'v': {0: 1},
            'w': {2: 1, 3: -1}
        })
        # M3 = x00(y01-y11)
        # U: x00->1; V: y01->1, y11->(-1); W: z01->1, z11->1
        terms.append({
            'u': {0: 1},
            'v': {1: 1, 3: -1},
            'w': {1: 1, 3: 1}
        })
        # M4 = x11(y10-y00)
        # U: x11->1; V: y10->1, y00->(-1); W: z00->1, z10->1
        terms.append({
            'u': {3: 1},
            'v': {2: 1, 0: -1},
            'w': {0: 1, 2: 1}
        })
        # M5 = (x00+x01)y11
        # U: x00->1, x01->1; V: y11->1; W: z00->(-1), z01->1
        terms.append({
            'u': {0: 1, 1: 1},
            'v': {3: 1},
            'w': {0: -1, 1: 1}
        })
        # M6 = (x10-x00)(y00+y01)
        # U: x10->1, x00->(-1); V: y00->1, y01->1; W: z11->1
        terms.append({
            'u': {2: 1, 0: -1},
            'v': {0: 1, 1: 1},
            'w': {3: 1}
        })
        # M7 = (x01-x11)(y10+y11)
        # U: x01->1, x11->(-1); V: y10->1, y11->1; W: z00->1, z10->1
        terms.append({
            'u': {1: 1, 3: -1},
            'v': {2: 1, 3: 1},
            'w': {0: 1, 2: 1}
        })
        return terms
    
    # Now for 4x4, we use Strassen at the block level.
    # A blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C blocks: C11=C[0:2,0:2], C12=C[0:2,2:4], C21=C[2:4,0:2], C22=C[2:4,2:4]
    
    # Block-level Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_i is a 2x2 product. We decompose each into 7 rank-1 terms using Strassen.
    # Total rank = 7 * 7 = 49.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Helper to get index for a 2x2 submatrix element
    # For a 2x2 block at offset (bi, bj) in the 4x4 matrix:
    # element (di, dj) maps to global index (bi*2 + di, bj*2 + dj)
    
    # For U (A matrix): A[i,j] -> index i*4 + j
    # For V (B matrix): B[j,k] -> index j*4 + k
    # For W (C matrix): C[i,k] -> index k*4 + i (per the convention W[k*n+i])
    
    # Let me define the 7 block-level Strassen products and for each, the 7 sub-products.
    
    # Block-level term definitions:
    # Each block-level term t has:
    #   u_blocks: dict from block_name to coefficient (for A blocks)
    #   v_blocks: dict from block_name to coefficient (for B blocks)
    #   w_blocks: dict from block_name to coefficient (for C blocks)
    # Block names: '11', '12', '21', '22'
    
    block_terms = [
        # M1 = (A11+A22)(B11+B22)
        {'u': {'11': 1, '22': 1}, 'v': {'11': 1, '22': 1}, 'w': {'11': 1, '22': 1}},
        # M2 = (A21+A22)B11
        {'u': {'21': 1, '22': 1}, 'v': {'11': 1}, 'w': {'21': 1, '22': -1}},
        # M3 = A11(B12-B22)
        {'u': {'11': 1}, 'v': {'12': 1, '22': -1}, 'w': {'12': 1, '22': 1}},
        # M4 = A22(B21-B11)
        {'u': {'22': 1}, 'v': {'21': 1, '11': -1}, 'w': {'11': 1, '21': 1}},
        # M5 = (A11+A12)B22
        {'u': {'11': 1, '12': 1}, 'v': {'22': 1}, 'w': {'11': -1, '12': 1}},
        # M6 = (A21-A11)(B11+B12)
        {'u': {'21': 1, '11': -1}, 'v': {'11': 1, '12': 1}, 'w': {'22': 1}},
        # M7 = (A12-A22)(B21+B22)
        {'u': {'12': 1, '22': -1}, 'v': {'21': 1, '22': 1}, 'w': {'11': 1, '21': 1}},
    ]
    
    # For each block-level term, we need to decompose the 2x2 product into 7 Strassen terms.
    # The 2x2 product involves specific A-block and B-block with coefficients.
    
    # For a block-level term with u_blocks and v_blocks, the actual 2x2 matrices are:
    # X = sum_{(bi,bj) in u_blocks} u_blocks[(bi,bj)] * A_block(bi,bj)
    # Y = sum_{(bj,bk) in v_blocks} v_blocks[(bj,bk)] * B_block(bj,bk)
    
    # Wait, I need to be more careful. The block names correspond to positions.
    # A block '11' = A[0:2, 0:2], '12' = A[0:2, 2:4], '21' = A[2:4, 0:2], '22' = A[2:4, 2:4]
    # B block '11' = B[0:2, 0:2], '12' = B[0:2, 2:4], '21' = B[2:4, 0:2], '22' = B[2:4, 2:4]
    
    # For a 2x2 product X*Y where X is a 2x2 matrix and Y is a 2x2 matrix:
    # X has elements x00, x01, x10, x11 corresponding to positions (0,0),(0,1),(1,0),(1,1)
    # Y has elements y00, y01, y10, y11
    # Z = X*Y has elements z00, z01, z10, z11
    
    # In the context of a block-level term, X is a linear combination of A-blocks and Y is a linear combination of B-blocks.
    # But actually, each block-level term defines X and Y as specific linear combinations.
    
    # Let me think differently. For each block-level term t, we have:
    # X = linear combination of A-blocks (each a 2x2 matrix)
    # Y = linear combination of B-blocks (each a 2x2 matrix)
    # Z = linear combination of C-blocks (each a 2x2 matrix)
    # And Z = X * Y
    
    # To decompose this into rank-1 terms, we use Strassen on the 2x2 matrices.
    # The 7 Strassen terms for a 2x2 product give us 7 rank-1 decompositions.
    
    # For each of the 7 Strassen sub-terms, we need to express:
    # - The U contribution: which A elements (global indices) get what coefficients
    # - The V contribution: which B elements (global indices) get what coefficients
    # - The W contribution: which C elements (global indices) get what coefficients
    
    # Let me define a function to expand a 2x2 product decomposition given the block coefficients.
    
    def expand_2x2_product(u_coeff, v_coeff, w_coeff, base_idx):
        """
        u_coeff: dict from local 2x2 index (0,1,2,3) to coefficient, for the X matrix
                 where X is a linear combination of A-blocks
        v_coeff: dict from local 2x2 index to coefficient, for the Y matrix
        w_coeff: dict from local 2x2 index to coefficient, for the Z matrix
        base_idx: starting index in the global rank dimension
        
        Returns: list of 7 tuples of (u_dict, v_dict, w_dict) for global indices
        """
        # This is getting complex. Let me take a different approach.
        pass
    
    # Let me take a cleaner approach. I'll directly construct the 49 terms.
    
    # For each block-level term (7 of them) and each Strassen sub-term (7 of them),
    # I'll compute the global U, V, W entries.
    
    # Block offset mapping:
    # Block '11': row offset 0, col offset 0
    # Block '12': row offset 0, col offset 2
    # Block '21': row offset 2, col offset 0
    # Block '22': row offset 2, col offset 2
    
    block_offsets = {
        '11': (0, 0),
        '12': (0, 2),
        '21': (2, 0),
        '22': (2, 2)
    }
    
    # For A-block (bi,bj): element (di,dj) -> global A index (bi*2+di, bj*2+dj) -> U index (bi*2+di)*4 + (bj*2+dj)
    # For B-block (bj,bk): element (di,dj) -> global B index (bj*2+di, bk*2+dj) -> V index (bj*2+di)*4 + (bk*2+dj)
    # For C-block (bi,bk): element (di,dj) -> global C index (bi*2+di, bk*2+dj) -> W index (bk*2+dj)*4 + (bi*2+di)
    
    # Strassen 2x2 terms (local indices 0..3 for 2x2 matrix):
    # Local index 0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1)
    
    strassen_terms = [
        # M1: (x0+x3)(y0+y3) -> z0 += 1, z3 += 1
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        # M2: (x2+x3)y0 -> z2 += 1, z3 += -1
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        # M3: x0(y1-y3) -> z1 += 1, z3 += 1
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        # M4: x3(y2-y0) -> z0 += 1, z2 += 1
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        # M5: (x0+x1)y3 -> z0 += -1, z1 += 1
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        # M6: (x2-x0)(y0+y1) -> z3 += 1
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        # M7: (x1-x3)(y2+y3) -> z0 += 1, z2 += 1
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1, 2: 1}},
    ]
    
    # Now, for each block-level term, the X matrix is a linear combination of A-blocks.
    # u_blocks tells us which A-blocks are involved with what coefficients.
    # Similarly for v_blocks and w_blocks.
    
    # For a given block-level term and a Strassen sub-term, the global contributions are:
    # For each (block, coeff) in u_blocks:
    #   For each (local_idx, local_coeff) in strassen_term['u']:
    #     global_A_index = (block_row_offset + local_row) * 4 + (block_col_offset + local_col)
    #     U[global_A_index][rank_idx] += coeff * local_coeff
    # Similarly for V and W.
    
    # Local index to (row, col): 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    local_to_rc = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
    
    rank_idx = 0
    for block_term in block_terms:
        u_blocks = block_term['u']
        v_blocks = block_term['v']
        w_blocks = block_term['w']
        
        for strassen_term in strassen_terms:
            # Process U contributions
            for block_name, block_coeff in u_blocks.items():
                bi, bj = block_offsets[block_name]
                for local_idx, local_coeff in strassen_term['u'].items():
                    li, lj = local_to_rc[local_idx]
                    gi = bi + li
                    gj = bj + lj
                    u_idx = gi * 4 + gj
                    U[u_idx][rank_idx] += block_coeff * local_coeff
            
            # Process V contributions
            for block_name, block_coeff in v_blocks.items():
                bj, bk = block_offsets[block_name]
                for local_idx, local_coeff in strassen_term['v'].items():
                    li, lj = local_to_rc[local_idx]
                    gi = bj + li
                    gj = bk + lj
                    v_idx = gi * 4 + gj
                    V[v_idx][rank_idx] += block_coeff * local_coeff
            
            # Process W contributions
            for block_name, block_coeff in w_blocks.items():
                bi, bk = block_offsets[block_name]
                for local_idx, local_coeff in strassen_term['w'].items():
                    li, lj = local_to_rc[local_idx]
                    gi = bi + li
                    gj = bk + lj
                    # W index: k*n + i = gj*4 + gi
                    w_idx = gj * 4 + gi
                    W[w_idx][rank_idx] += block_coeff * local_coeff
            
            rank_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
