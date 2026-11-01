"""Rewritten program for 2x5x6 matrix multiplication tensor decomposition.

Strategy:
- Split the 5 columns of A (and 5 rows of B) into blocks of sizes [2, 2, 1].
- For each block, we multiply a 2xb block of A with a bx6 block of B.
- For the 2x2x6 subproblems, we further split the 6 columns into three 2-column blocks
  and apply the Strassen decomposition (rank 7) for each 2x2x2 subproblem.
- For the 2x1x6 subproblem, we use the naive schoolbook decomposition (rank 4 per 2x1x2 block,
  so 3 blocks * 4 = 12).

Total rank = 2 * (3 * 7) + 3 * 4 = 42 + 12 = 54.
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We'll build U, V, W as lists of lists, accumulating columns.
    U_cols = []  # Each column is a list of length n*m = 10
    V_cols = []  # Each column is a list of length m*p = 30
    W_cols = []  # Each column is a list of length p*n = 12
    
    def add_column(u_row, v_row, w_row, u_val, v_val, w_val):
        """Add a single tensor term: U[u_row] += u_val, V[v_row] += v_val, W[w_row] += w_val."""
        for col in range(len(U_cols)):
            U_cols[col][u_row] += u_val
            V_cols[col][v_row] += v_val
            W_cols[col][w_row] += w_val
    
    def add_term(i, j, k, u_val=1, v_val=1, w_val=1):
        """Add a schoolbook term for indices i, j, k with given values."""
        U_cols.append([0] * (n * m))
        V_cols.append([0] * (m * p))
        W_cols.append([0] * (p * n))
        U_cols[-1][i * m + j] = u_val
        V_cols[-1][j * p + k] = v_val
        W_cols[-1][k * n + i] = w_val
    
    def strassen_2x2x2(i_off, j_off, k_off):
        """Add Strassen decomposition for a 2x2x2 subproblem.
        
        i_off: offset for i indices (0 or 1)
        j_off: offset for j indices (start of the 2 j's)
        k_off: offset for k indices (start of the 2 k's)
        
        Strassen formulas for C = AB where A, B, C are 2x2:
        M1 = (A11 + A22) * (B11 + B22)
        M2 = (A21 + A22) * B11
        M3 = A11 * (B12 - B22)
        M4 = A22 * (B21 - B11)
        M5 = (A11 + A12) * B22
        M6 = (A21 - A11) * (B11 + B12)
        M7 = (A12 - A22) * (B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        Each Mi corresponds to a rank-1 tensor term with appropriate U, V, W values.
        """
        # Strassen terms: each term is (u_coeffs, v_coeffs, w_coeff)
        # For term t, U row i*m+j gets u_val_t[i,j], V row j*p+k gets v_val_t[j,k], W row k*n+i gets w_val_t
        
        # Let's define each Strassen factor as a set of (i, j, val) for U, (j, k, val) for V, and (k, i, val) for W
        # Actually, for a rank-1 term in Strassen, we have:
        # U contribution: for each (i,j) in the 2x2 A block, U[i,j] += alpha_{ij}
        # V contribution: for each (j,k) in the 2x2 B block, V[j,k] += beta_{jk}
        # W contribution: for each (k,i) in the 2x2 C block, W[k,i] += gamma_{ki}
        
        # Strassen factors (each is a rank-1 term):
        # Term 1: M1 = (A11 + A22)(B11 + B22)
        #   U: A11=1, A22=1 => [(0,0,1), (1,1,1)]
        #   V: B11=1, B22=1 => [(0,0,1), (1,1,1)]
        #   W: C11 += M1, C22 += M1 => (0,0,1), (1,1,1)
        
        # Let me define each term as (U_list, V_list, W_list) where each list is [(idx1, idx2, val), ...]
        
        terms = [
            # T1: M1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # T2: M2 = (A21+A22)*B11
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,0,1)]),  # W: C21 += M2 => (k=0, i=1)
            # T3: M3 = A11*(B12-B22)
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,0,1),(0,1,1)]),  # W: C11 += M3? No. C12 += M3
            # Correction: C12 = M3 + M5, so M3 contributes to C12: (k=0, i=0)? No, C12 is row 1, col 2 of C.
            # C indices: C[i,j]. C12 means i=0, j=1 (0-indexed: i=0, j=1).
            # W[k*n + i] for output C[i,j]... wait, W is indexed by [k*n+i] but that doesn't capture j.
            # 
            # Wait, I need to re-examine the W convention. The task says W[k*n+i, r].
            # But C is n x p, so C[i,k]. The index k*n+i means for each output element C[i,k],
            # it's at row k*n+i in W. So W row = k*n + i, where k is the column of C and i is the row of C.
            # 
            # So for C[i,k], W row is k*n + i.
            # C11: i=0, k=0 => W row = 0*2+0 = 0
            # C12: i=0, k=1 => W row = 1*2+0 = 2
            # C21: i=1, k=0 => W row = 0*2+1 = 1
            # C22: i=1, k=1 => W row = 1*2+1 = 3
        ]
        
        # Let me redo this more carefully.
        # For a 2x2x2 subproblem with offsets i_off, j_off, k_off:
        # A submatrix rows: i_off, i_off+1; cols: j_off, j_off+1
        # B submatrix rows: j_off, j_off+1; cols: k_off, k_off+1
        # C submatrix rows: i_off, i_off+1; cols: k_off, k_off+1
        
        # Strassen terms (standard):
        # Each term t has U contributions to A[i,j], V contributions to B[j,k], W contributions to C[i,k]
        # U row index: i*m + j, V row index: j*p + k, W row index: k*n + i
        
        # Let me define terms as (u_list, v_list, w_list)
        # u_list: [(i_local, j_local, val), ...] where i_local in {0,1}, j_local in {0,1}
        # v_list: [(j_local, k_local, val), ...]
        # w_list: [(k_local, i_local, val), ...]
        
        strassen_terms = [
            # T1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # T2: M2 = (A10+A11)B00 -> C10 += M2
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)]),
            # T3: M3 = A00(B01-B11) -> C01 += M3
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1)]),
            # T4: M4 = A11(B10-B00) -> C00 += M4, C10 += M4
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
            # T5: M5 = (A00+A01)B11 -> C00 += M5, C01 += M5
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(1,0,1)]),
            # T6: M6 = (A10-A00)(B00+B01) -> C11 += M6
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            # T7: M7 = (A01-A11)(B10+B11) -> C00 += M7
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for u_list, v_list, w_list in strassen_terms:
            U_cols.append([0] * (n * m))
            V_cols.append([0] * (m * p))
            W_cols.append([0] * (p * n))
            for i_loc, j_loc, val in u_list:
                i = i_off + i_loc
                j = j_off + j_loc
                U_cols[-1][i * m + j] = val
            for j_loc, k_loc, val in v_list:
                j = j_off + j_loc
                k = k_off + k_loc
                V_cols[-1][j * p + k] = val
            for k_loc, i_loc, val in w_list:
                k = k_off + k_loc
                i = i_off + i_loc
                W_cols[-1][k * n + i] = val
    
    # Split m=5 into [2, 2, 1]
    m_splits = [2, 2, 1]
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    
    # For each m-block, process the 2xb x bx6 subproblem
    for block_idx, (b, j_off) in enumerate(zip(m_splits, m_offsets)):
        # Split p=6 into three 2-blocks: [2, 2, 2]
        p_splits = [2, 2, 2]
        p_offsets = [0]
        for s in p_splits[:-1]:
            p_offsets.append(p_offsets[-1] + s)
        
        for k_off in p_offsets:
            if b == 2:
                # Use Strassen for 2x2x2
                strassen_2x2x2(0, j_off, k_off)
            else:
                # b == 1, use schoolbook for 2x1x2
                for i in range(n):
                    for k in range(2):
                        add_term(i, j_off, k_off + k)
    
    # Convert lists of columns to lists of rows (transpose)
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
