"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    # Uses Strassen-style decomposition on 2x2 blocks to reduce rank from 56 to 50.
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # We'll build up the decomposition by splitting the problem.
    # Split m=4 into two blocks of size 2: j1={0,1}, j2={2,3}
    # Split p=7 into three blocks: k1={0,1,2,3} (size 4), k2={4,5,6} (size 3)
    
    # For each (j_block, k_block) pair, we need to compute a (2 x 2) * (2 x k_size) multiplication.
    # (2x2)*(2x4) = two (2x2)*(2x2) Strassen blocks = 2*7 = 14 terms
    # (2x2)*(2x3) = one (2x2)*(2x2) + one (2x2)*(2x1) = 7 + 4 = 11 terms
    
    # So for j1: (2x2)*(2x4) = 14, (2x2)*(2x3) = 11, total 25
    # For j2: (2x2)*(2x4) = 14, (2x2)*(2x3) = 11, total 25
    # Grand total: 50
    
    terms = []  # List of (U_row_indices, V_row_indices, W_row_indices) with coefficients
    
    def strassen_2x2_2x2():
        """Return the 7 Strassen terms for 2x2 * 2x2.
        Each term is (u_list, v_list, w_list) where:
        - u_list: list of (row_idx_in_A, coeff) for the 2x2 A block
        - v_list: list of (row_idx_in_B, coeff) for the 2x2 B block  
        - w_list: list of (row_idx_in_C, coeff) for the 2x2 C block
        """
        # A = [[a,b],[c,d]] indexed as (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        # B = [[e,f],[g,h]] indexed as (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        # C = [[C00,C01],[C10,C11]] indexed as (0,0), (0,1), (1,0), (1,1)
        
        s = []
        # M1 = (a+d)(e+h)
        s.append(([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1),((0,1),1),((1,0),1)]))
        # M2 = (c+d)(e)
        s.append(([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1)]))
        # M3 = a(f-h)
        s.append(([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1)]))
        # M4 = d(g-e)
        s.append(([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]))
        # M5 = (a+b)h
        s.append(([((0,0),1),((0,1),1)], [((1,1),1)], [((1,1),1),((0,1),-1)]))
        # M6 = (c-a)(e+f)
        s.append(([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]))
        # M7 = (b-d)(g+h)
        s.append(([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((1,0),1),((1,1),-1)]))
        return s
    
    def schoolbook_2x2_2x1():
        """Return the 4 schoolbook terms for 2x2 * 2x1.
        A is 2x2, B is 2x1 (single column), C is 2x1 (single column).
        """
        s = []
        # C[0] = a*e + b*g
        # C[1] = c*e + d*g
        # Term 1: a*e
        s.append([[(0,0),1], [(0,0),1], [(0,0),1]])
        # Term 2: b*g
        s.append([[(0,1),1], [(1,0),1], [(0,0),1]])
        # Term 3: c*e
        s.append([[(1,0),1], [(0,0),1], [(1,0),1]])
        # Term 4: d*g
        s.append([[(1,1),1], [(1,0),1], [(1,0),1]])
        return s
    
    # Now build the full decomposition
    # A is 2x4: rows i=0,1; columns j=0,1,2,3
    # B is 4x7: rows j=0,1,2,3; columns k=0,1,2,3,4,5,6
    # C is 2x7: rows i=0,1; columns k=0,1,2,3,4,5,6
    # W[k*n+i, r] convention: W index is k*2+i
    
    U_rows = []  # Each entry: list of (row_index, coeff)
    V_rows = []
    W_rows = []
    
    # Helper to add a term
    def add_term(u_entries, v_entries, w_entries):
        """u_entries, v_entries, w_entries are lists of (local_idx, coeff)
        where local_idx maps to the global row via offsets."""
        U_rows.append(u_entries)
        V_rows.append(v_entries)
        W_rows.append(w_entries)
    
    # For each j_block (2 blocks of size 2) and k_block:
    # j_block 0: j_local = 0,1 -> j_global = 0,1
    # j_block 1: j_local = 0,1 -> j_global = 2,3
    
    # k_block 0: k_global = 0,1,2,3 (size 4, split into two 2x2 Strassen)
    # k_block 1: k_global = 4,5,6 (size 3, split into one 2x2 Strassen + one 2x1 schoolbook)
    
    for j_blk in range(2):
        j_off = j_blk * 2  # global offset for j
        for k_blk in range(2):
            if k_blk == 0:
                # k_global = 0,1,2,3 -> two Strassen 2x2x2 blocks
                for k_sub in range(2):
                    k_off = k_sub * 2  # global offset for k
                    for (u_loc, v_loc, w_loc) in strassen_2x2_2x2():
                        # Map local indices to global
                        # u: A[i, j_off + j_loc] -> row = i*4 + (j_off+j_loc)
                        # v: B[j_off + j_loc, k_off + k_loc] -> row = (j_off+j_loc)*7 + (k_off+k_loc)
                        # w: C[k_off + k_loc, i] -> row = (k_off+k_loc)*2 + i
                        u_g = []
                        for (loc_idx, coeff) in u_loc:
                            i_loc, j_loc_idx = loc_idx
                            u_g.append((i_loc*4 + (j_off+j_loc_idx), coeff))
                        v_g = []
                        for (loc_idx, coeff) in v_loc:
                            j_loc_idx, k_loc_idx = loc_idx
                            v_g.append(((j_off+j_loc_idx)*7 + (k_off+k_loc_idx), coeff))
                        w_g = []
                        for (loc_idx, coeff) in w_loc:
                            k_loc_idx, i_loc = loc_idx
                            w_g.append(((k_off+k_loc_idx)*2 + i_loc, coeff))
                        add_term(u_g, v_g, w_g)
            else:
                # k_global = 4,5,6 -> one Strassen 2x2x2 (k=4,5) + one schoolbook 2x2x1 (k=6)
                # Strassen part: k_off = 4
                for (u_loc, v_loc, w_loc) in strassen_2x2_2x2():
                    u_g = []
                    for (loc_idx, coeff) in u_loc:
                        i_loc, j_loc_idx = loc_idx
                        u_g.append((i_loc*4 + (j_off+j_loc_idx), coeff))
                    v_g = []
                    for (loc_idx, coeff) in v_loc:
                        j_loc_idx, k_loc_idx = loc_idx
                        # k_loc_idx is 0 or 1, maps to k_global = 4+k_loc_idx
                        v_g.append(((j_off+j_loc_idx)*7 + (4+k_loc_idx), coeff))
                    w_g = []
                    for (loc_idx, coeff) in w_loc:
                        k_loc_idx, i_loc = loc_idx
                        w_g.append(((4+k_loc_idx)*2 + i_loc, coeff))
                    add_term(u_g, v_g, w_g)
                
                # Schoolbook part for k=6: 2x2 * 2x1
                for (u_loc, v_loc, w_loc) in schoolbook_2x2_2x1():
                    u_g = []
                    for (loc_idx, coeff) in u_loc:
                        i_loc, j_loc_idx = loc_idx
                        u_g.append((i_loc*4 + (j_off+j_loc_idx), coeff))
                    v_g = []
                    for (loc_idx, coeff) in v_loc:
                        j_loc_idx, k_loc_idx = loc_idx
                        # k_loc_idx=0 means the single column, k_global=6
                        v_g.append(((j_off+j_loc_idx)*7 + 6, coeff))
                    w_g = []
                    for (loc_idx, coeff) in w_loc:
                        k_loc_idx, i_loc = loc_idx
                        # k_loc_idx=0, k_global=6
                        w_g.append((6*2 + i_loc, coeff))
                    add_term(u_g, v_g, w_g)
    
    rank = len(U_rows)
    
    # Build the matrices
    U = [[0]*rank for _ in range(n*m)]  # n*m = 8 rows
    V = [[0]*rank for _ in range(m*p)]  # m*p = 28 rows
    W = [[0]*rank for _ in range(p*n)]  # p*n = 14 rows
    
    for col in range(rank):
        for (row, coeff) in U_rows[col]:
            U[row][col] = coeff
        for (row, coeff) in V_rows[col]:
            V[row][col] = coeff
        for (row, coeff) in W_rows[col]:
            W[row][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

