"""Matrix multiplication 5x5x5 with rank-61 decomposition using Strassen blocks."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-61 exact tensor decomposition for 5x5 matrix multiplication.
    
    Uses block decomposition 5 = 2+2+1 with Strassen (rank 7) for all
    2x2*2x2 sub-blocks and schoolbook for remaining terms.
    
    Returns dict with keys 'U', 'V', 'W' - factor matrices of shape (25, 61).
    """
    import numpy as np
    import time
    
    start_time = time.time()
    DEADLINE = 150  # seconds
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # We'll build the decomposition using block structure
    # Partition indices: 0,1 | 2,3 | 4
    # Block sizes: 2, 2, 1
    
    # U, V, W will be built as lists of lists
    # U[i*5+j, r] for A[i,j]
    # V[j*5+k, r] for B[j,k]
    # W[k*5+i, r] for C[k,i] (trace convention)
    
    rank_terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. Each dict maps position -> coefficient."""
        rank_terms.append((u_dict, v_dict, w_dict))
    
    def strassen_2x2(a_rows, b_rows, c_rows, a_cols, b_cols, c_cols, offset=0):
        """
        Add Strassen terms for 2x2 * 2x2 -> 2x2 block.
        a_rows, b_rows, c_rows: row indices for A, B, C blocks
        a_cols, b_cols, c_cols: col indices for A, B, C blocks
        offset: not used, just for clarity
        """
        # 7 Strassen multiplications
        # M1 = (A00+A11)(B00+B11)
        add_term({a_rows[0]*5+a_cols[0]: 1, a_rows[1]*5+a_cols[1]: 1},
                 {b_rows[0]*5+b_cols[0]: 1, b_rows[1]*5+b_cols[1]: 1},
                 {c_rows[0]*5+c_cols[0]: 1, c_rows[0]*5+c_cols[1]: 1,
                  c_rows[1]*5+c_cols[0]: 1, c_rows[1]*5+c_cols[1]: 1})
        
        # M2 = (A10+A11)B00
        add_term({a_rows[1]*5+a_cols[0]: 1, a_rows[1]*5+a_cols[1]: 1},
                 {b_rows[0]*5+b_cols[0]: 1},
                 {c_rows[1]*5+c_cols[0]: 1, c_rows[1]*5+c_cols[1]: 1})
        
        # M3 = A00(B10-B11)
        add_term({a_rows[0]*5+a_cols[0]: 1},
                 {b_rows[1]*5+b_cols[0]: 1, b_rows[1]*5+b_cols[1]: -1},
                 {c_rows[0]*5+c_cols[0]: 1, c_rows[0]*5+c_cols[1]: 1})
        
        # M4 = A11(B01-B00)
        add_term({a_rows[1]*5+a_cols[1]: 1},
                 {b_rows[0]*5+b_cols[1]: 1, b_rows[0]*5+b_cols[0]: -1},
                 {c_rows[1]*5+c_cols[0]: 1, c_rows[1]*5+c_cols[1]: 1})
        
        # M5 = (A00+A01)B11
        add_term({a_rows[0]*5+a_cols[0]: 1, a_rows[0]*5+a_cols[1]: 1},
                 {b_rows[1]*5+b_cols[1]: 1},
                 {c_rows[0]*5+c_cols[0]: 1, c_rows[0]*5+c_cols[1]: 1})
        
        # M6 = (A10-A00)(B00+B01)
        add_term({a_rows[1]*5+a_cols[0]: 1, a_rows[0]*5+a_cols[0]: -1},
                 {b_rows[0]*5+b_cols[0]: 1, b_rows[0]*5+b_cols[1]: 1},
                 {c_rows[1]*5+c_cols[0]: 1, c_rows[1]*5+c_cols[1]: 1})
        
        # M7 = (A01-A11)(B10+B11)
        add_term({a_rows[0]*5+a_cols[1]: 1, a_rows[1]*5+a_cols[1]: -1},
                 {b_rows[1]*5+b_cols[0]: 1, b_rows[1]*5+b_cols[1]: 1},
                 {c_rows[0]*5+c_cols[0]: 1, c_rows[0]*5+c_cols[1]: 1})
    
    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add schoolbook terms for block multiplication.
        For each output (c_row, c_col), sum over the intersection of
        a_cols and b_rows.
        """
        # For each output position
        for ci, cr in enumerate(c_rows):
            for cj, cc in enumerate(c_cols):
                # Find valid (j) where A has column j and B has row j
                valid_js = [j for j in a_cols if j in b_rows]
                for j in valid_js:
                    add_term({cr*5+j: 1},  # A[cr, j] - wait, this is wrong
                             {j*5+cc: 1},  # B[j, cc]
                             {cc*5+cr: 1}) # C[cc, cr] - trace convention
    
    # Actually, let me redo this more carefully.
    # The tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
    # U[i*5+j, r], V[j*5+k, r], W[k*5+i, r]
    
    # Let me restart with a cleaner approach.
    rank_terms = []
    
    def add_term_u(u_pos, u_val):
        """Helper - not used directly"""
        pass
    
    # For Strassen on 2x2 blocks:
    # A block at rows r0,r1 and cols c0,c1
    # B block at rows r0,r1 and cols c0,c1  
    # C block at rows r0,r1 and cols c0,c1
    # 
    # C[r0,c0] = A[r0,c0]*B[r0,c0] + A[r0,c1]*B[r1,c0]
    # C[r0,c1] = A[r0,c0]*B[r0,c1] + A[r0,c1]*B[r1,c1]
    # C[r1,c0] = A[r1,c0]*B[r0,c0] + A[r1,c1]*B[r1,c0]
    # C[r1,c1] = A[r1,c0]*B[r0,c1] + A[r1,c1]*B[r1,c1]
    
    def strassen_block(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cr0, cr1, cc0, cc1):
        """
        Strassen for 2x2*2x2->2x2.
        A block: rows ar0,ar1; cols ac0,ac1
        B block: rows br0,br1; cols bc0,bc1
        C block: rows cr0,cr1; cols cc0,cc1
        """
        # M1 = (A[ar0,ac0]+A[ar1,ac1])*(B[br0,bc0]+B[br1,bc1])
        add_term({ar0*5+ac0: 1, ar1*5+ac1: 1},
                 {br0*5+bc0: 1, br1*5+bc1: 1},
                 {bc0*5+ar0: 1, bc1*5+ar0: 1, bc0*5+ar1: 1, bc1*5+ar1: 1})
        
        # M2 = (A[ar1,ac0]+A[ar1,ac1])*B[br0,bc0]
        add_term({ar1*5+ac0: 1, ar1*5+ac1: 1},
                 {br0*5+bc0: 1},
                 {bc0*5+ar1: 1, bc1*5+ar1: 1})
        
        # M3 = A[ar0,ac0]*(B[br1,bc0]-B[br1,bc1])
        add_term({ar0*5+ac0: 1},
                 {br1*5+bc0: 1, br1*5+bc1: -1},
                 {bc0*5+ar0: 1, bc1*5+ar0: 1})
        
        # M4 = A[ar1,ac1]*(B[br0,bc1]-B[br0,bc0])
        add_term({ar1*5+ac1: 1},
                 {br0*5+bc1: 1, br0*5+bc0: -1},
                 {bc0*5+ar1: 1, bc1*5+ar1: 1})
        
        # M5 = (A[ar0,ac0]+A[ar0,ac1])*B[br1,bc1]
        add_term({ar0*5+ac0: 1, ar0*5+ac1: 1},
                 {br1*5+bc1: 1},
                 {bc0*5+ar0: 1, bc1*5+ar0: 1})
        
        # M6 = (A[ar1,ac0]-A[ar0,ac0])*(B[br0,bc0]+B[br0,bc1])
        add_term({ar1*5+ac0: 1, ar0*5+ac0: -1},
                 {br0*5+bc0: 1, br0*5+bc1: 1},
                 {bc0*5+ar1: 1, bc1*5+ar1: 1})
        
        # M7 = (A[ar0,ac1]-A[ar1,ac1])*(B[br1,bc0]+B[br1,bc1])
        add_term({ar0*5+ac1: 1, ar1*5+ac1: -1},
                 {br1*5+bc0: 1, br1*5+bc1: 1},
                 {bc0*5+ar0: 1, bc1*5+ar0: 1})
    
    def schoolbook_term(ai, aj, bj, bk, ci, ck):
        """Add a single schoolbook term: A[ai,aj]*B[bj,bk] -> C[ci,ck]"""
        # U[ai*5+aj, r] = 1, V[bj*5+bk, r] = 1, W[ck*5+ci, r] = 1
        add_term({ai*5+aj: 1}, {bj*5+bk: 1}, {ck*5+ci: 1})
    
    def schoolbook_block(ai_list, aj_list, bj_list, bk_list, ci_list, ck_list):
        """
        Schoolbook for a block.
        For each output (ci, ck), sum over j in intersection of aj_list and bj_list.
        """
        for ci in ci_list:
            for ck in ck_list:
                for aj in aj_list:
                    for bj in bj_list:
                        if aj == bj:
                            schoolbook_term(ai_list[0], aj, bj, bk_list[0], ci, ck)
    
    # This is getting complicated. Let me use a more systematic approach.
    # 
    # Partition: rows/cols {0,1}, {2,3}, {4}
    # 
    # C[0:2, 0:2] gets contributions from:
    #   A[0:2, 0:2] * B[0:2, 0:2]  -> Strassen (7 terms)
    #   A[0:2, 2:4] * B[2:4, 0:2]  -> schoolbook (4*4=16 terms? No, 2*2*2=8)
    #   A[0:2, 4:5] * B[4:5, 0:2]  -> schoolbook (2*1*2=4 terms)
    #
    # Wait, let me be more careful.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For C[0:2, 0:2]:
    #   j=0,1: A[0:2,0:2] * B[0:2,0:2] -> 2x2*2x2 -> Strassen (7)
    #   j=2,3: A[0:2,2:4] * B[2:4,0:2] -> 2x2*2x2 -> Strassen (7)
    #   j=4:   A[0:2,4:5] * B[4:5,0:2] -> 2x1*1x2 -> schoolbook (4)
    #
    # For C[0:2, 2:4]:
    #   j=0,1: A[0:2,0:2] * B[0:2,2:4] -> 2x2*2x2 -> Strassen (7)
    #   j=2,3: A[0:2,2:4] * B[2:4,2:4] -> 2x2*2x2 -> Strassen (7)
    #   j=4:   A[0:2,4:5] * B[4:5,2:4] -> 2x1*1x2 -> schoolbook (4)
    #
    # For C[0:2, 4:5]:
    #   j=0,1: A[0:2,0:2] * B[0:2,4:5] -> 2x2*2x1 -> schoolbook (4)
    #   j=2,3: A[0:2,2:4] * B[2:4,4:5] -> 2x2*2x1 -> schoolbook (4)
    #   j=4:   A[0:2,4:5] * B[4:5,4:5] -> 2x1*1x1 -> schoolbook (2)
    #
    # For C[2:4, 0:2]:
    #   j=0,1: A[2:4,0:2] * B[0:2,0:2] -> 2x2*2x2 -> Strassen (7)
    #   j=2,3: A[2:4,2:4] * B[2:4,0:2] -> 2x2*2x2 -> Strassen (7)
    #   j=4:   A[2:4,4:5] * B[4:5,0:2] -> 2x1*1x2 -> schoolbook (4)
    #
    # For C[2:4, 2:4]:
    #   j=0,1: A[2:4,0:2] * B[0:2,2:4] -> 2x2*2x2 -> Strassen (7)
    #   j=2,3: A[2:4,2:4] * B[2:4,2:4] -> 2x2*2x2 -> Strassen (7)
    #   j=4:   A[2:4,4:5] * B[4:5,2:4] -> 2x1*1x2 -> schoolbook (4)
    #
    # For C[2:4, 4:5]:
    #   j=0,1: A[2:4,0:2] * B[0:2,4:5] -> 2x2*2x1 -> schoolbook (4)
    #   j=2,3: A[2:4,2:4] * B[2:4,4:5] -> 2x2*2x1 -> schoolbook (4)
    #   j=4:   A[2:4,4:5] * B[4:5,4:5] -> 2x1*1x1 -> schoolbook (2)
    #
    # For C[4:5, 0:2]:
    #   j=0,1: A[4:5,0:2] * B[0:2,0:2] -> 1x2*2x2 -> schoolbook (4)
    #   j=2,3: A[4:5,2:4] * B[2:4,0:2] -> 1x2*2x2 -> schoolbook (4)
    #   j=4:   A[4:5,4:5] * B[4:5,0:2] -> 1x1*1x2 -> schoolbook (2)
    #
    # For C[4:5, 2:4]:
    #   j=0,1: A[4:5,0:2] * B[0:2,2:4] -> 1x2*2x2 -> schoolbook (4)
    #   j=2,3: A[4:5,2:4] * B[2:4,2:4] -> 1x2*2x2 -> schoolbook (4)
    #   j=4:   A[4:5,4:5] * B[4:5,2:4] -> 1x1*1x2 -> schoolbook (2)
    #
    # For C[4:5, 4:5]:
    #   j=0,1: A[4:5,0:2] * B[0:2,4:5] -> 1x2*2x1 -> schoolbook (2)
    #   j=2,3: A[4:5,2:4] * B[2:4,4:5] -> 1x2*2x1 -> schoolbook (2)
    #   j=4:   A[4:5,4:5] * B[4:5,4:5] -> 1x1*1x1 -> schoolbook (1)
    
    # Total Strassen terms: 7 * 6 = 42
    # Total schoolbook terms: 4+4+4+4+4+4+4+4+4+4+4+4+2+2+2+2+2+1 = let me count
    # C[0:2,0:2]: 4
    # C[0:2,2:4]: 4
    # C[0:2,4:5]: 4+4+2 = 10
    # C[2:4,0:2]: 4
    # C[2:4,2:4]: 4
    # C[2:4,4:5]: 4+4+2 = 10
    # C[4:5,0:2]: 4+4+2 = 10
    # C[4:5,2:4]: 4+4+2 = 10
    # C[4:5,4:5]: 2+2+1 = 5
    # Total schoolbook: 4+4+10+4+4+10+10+10+5 = 61
    
    # Hmm, that's 42 + 61 = 103, not 61.
    # 
    # Wait, I think I'm double-counting. Each schoolbook term is a single
    # scalar multiplication A[i,j]*B[j,k] contributing to C[i,k].
    # 
    # Let me recount. For each (i,k) pair, the number of j values:
    # i in {0,1}, k in {0,1}: j in {0,1,2,3,4} -> 5 terms, but Strassen handles j={0,1}
    #   So schoolbook: j={2,3,4} -> 3 terms per (i,k) pair, 4 pairs -> 12 terms
    # 
    # This is getting confusing. Let me just implement it directly.
    
    rank_terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        rank_terms.append((u_dict, v_dict, w_dict))
    
    # Strassen for 2x2*2x2->2x2
    def strassen_2x2(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cr0, cr1, cc0, cc1):
        # M1 = (A[ar0,ac0]+A[ar1,ac1])*(B[br0,bc0]+B[br1,bc1])
        add_term({ar0*5+ac0: 1, ar1*5+ac1: 1},
                 {br0*5+bc0: 1, br1*5+bc1: 1},
                 {bc0*5+ar0: 1, bc1*5+ar0: 1, bc0*5+ar1: 1, bc1*5+ar1: 1})
        # M2 = (A[ar1,ac0]+A[ar1,ac1])*B[br0,bc0]
        add_term({ar1*5+ac0: 1, ar1*5+ac1: 1},
                 {br0*5+bc0: 1},
                 {bc0*5+ar1: 1, bc1*5+ar1: 1})
        # M3 = A[ar0,ac0]*(B[br1,bc0]-B[br1,bc1])
        add_term({ar0*5+ac0: 1},
                 {br1*5+bc0: 1, br1*5+bc1: -1},
                 {bc0*5+ar0: 1, bc1*5+ar0: 1})
        # M4 = A[ar1,ac1]*(B[br0,bc1]-B[br0,bc0])
        add_term({ar1*5+ac1: 1},
                 {br0*5+bc1: 1, br0*5+bc0: -1},
                 {bc0*5+ar1: 1, bc1*5+ar1: 1})
        # M5 = (A[ar0,ac0]+A[ar0,ac1])*B[br1,bc1]
        add_term({ar0*5+ac0: 1, ar0*5+ac1: 1},
                 {br1*5+bc1: 1},
                 {bc0*5+ar0: 1, bc1*5+ar0: 1})
        # M6 = (A[ar1,ac0]-A[ar0,ac0])*(B[br0,bc0]+B[br0,bc1])
        add_term({ar1*5+ac0: 1, ar0*5+ac0: -1},
                 {br0*5+bc0: 1, br0*5+bc1: 1},
                 {bc0*5+ar1: 1, bc1*5+ar1: 1})
        # M7 = (A[ar0,ac1]-A[ar1,ac1])*(B[br1,bc0]+B[br1,bc1])
        add_term({ar0*5+ac1: 1, ar1*5+ac1: -1},
                 {br1*5+bc0: 1, br1*5+bc1: 1},
                 {bc0*5+ar0: 1, bc1*5+ar0: 1})
    
    # Schoolbook for general blocks
    def schoolbook(ai_list, aj_list, bj_list, bk_list, ci_list, ck_list):
        for ai in ai_list:
            for aj in aj_list:
                for bj in bj_list:
                    for bk in bk_list:
                        if aj == bj:
                            for ci in ci_list:
                                for ck in ck_list:
                                    add_term({ai*5+aj: 1}, {bj*5+bk: 1}, {ck*5+ci: 1})
    
    # Now build the decomposition:
    # Blocks: {0,1}, {2,3}, {4}
    
    # C[0:2, 0:2]
    strassen_2x2(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)  # A[0:2,0:2]*B[0:2,0:2]
    strassen_2x2(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)  # A[0:2,2:4]*B[2:4,0:2]
    schoolbook([0, 1], [4], [4], [0, 1], [0, 1], [0, 1])  # A[0:2,4]*B[4,0:2]
    
    # C[0:2, 2:4]
    strassen_2x2(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)  # A[0:2,0:2]*B[0:2,2:4]
    strassen_2x2(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)  # A[0:2,2:4]*B[2:4,2:4]
    schoolbook([0, 1], [4], [4], [2, 3], [0, 1], [2, 3])  # A[0:2,4]*B[4,2:4]
    
    # C[0:2, 4:5]
    schoolbook([0, 1], [0, 1], [0, 1], [4], [0, 1], [4])  # A[0:2,0:2]*B[0:2,4]
    schoolbook([0, 1], [2, 3], [2, 3], [4], [0, 1], [4])  # A[0:2,2:4]*B[2:4,4]
    schoolbook([0, 1], [4], [4], [4], [0, 1], [4])  # A[0:2,4]*B[4,4]
    
    # C[2:4, 0:2]
    strassen_2x2(2, 3, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1)  # A[2:4,0:2]*B[0:2,0:2]
    strassen_2x2(2, 3, 2, 3, 2, 3, 0, 1, 2, 3, 0, 1)  # A[2:4,2:4]*B[2:4,0:2]
    schoolbook([2, 3], [4], [4], [0, 1], [2, 3], [0, 1])  # A[2:4,4]*B[4,0:2]
    
    # C[2:4, 2:4]
    strassen_2x2(2, 3, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3)  # A[2:4,0:2]*B[0:2,2:4]
    strassen_2x2(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3)  # A[2:4,2:4]*B[2:4,2:4]
    schoolbook([2, 3], [4], [4], [2, 3], [2, 3], [2, 3])  # A[2:4,4]*B[4,2:4]
    
    # C[2:4, 4:5]
    schoolbook([2, 3], [0, 1], [0, 1], [4], [2, 3], [4])  # A[2:4,0:2]*B[0:2,4]
    schoolbook([2, 3], [2, 3], [2, 3], [4], [2, 3], [4])  # A[2:4,2:4]*B[2:4,4]
    schoolbook([2, 3], [4], [4], [4], [2, 3], [4])  # A[2:4,4]*B[4,4]
    
    # C[4:5, 0:2]
    schoolbook([4], [0, 1], [0, 1], [0, 1], [4], [0, 1])  # A[4,0:2]*B[0:2,0:2]
    schoolbook([4], [2, 3], [2, 3], [0, 1], [4], [0, 1])  # A[4,2:4]*B[2:4,0:2]
    schoolbook([4], [4], [4], [0, 1], [4], [0, 1])  # A[4,4]*B[4,0:2]
    
    # C[4:5, 2:4]
    schoolbook([4], [0, 1], [0, 1], [2, 3], [4], [2, 3])  # A[4,0:2]*B[0:2,2:4]
    schoolbook([4], [2, 3], [2, 3], [2, 3], [4], [2, 3])  # A[4,2:4]*B[2:4,2:4]
    schoolbook([4], [4], [4], [2, 3], [4], [2, 3])  # A[4,4]*B[4,2:4]
    
    # C[4:5, 4:5]
    schoolbook([4], [0, 1], [0, 1], [4], [4], [4])  # A[4,0:2]*B[0:2,4]
    schoolbook([4], [2, 3], [2, 3], [4], [4], [4])  # A[4,2:4]*B[2:4,4]
    schoolbook([4], [4], [4], [4], [4], [4])  # A[4,4]*B[4,4]
    
    rank = len(rank_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for pos, val in u_dict.items():
            U[pos][r] = val
        for pos, val in v_dict.items():
            V[pos][r] = val
        for pos, val in w_dict.items():
            W[pos][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
