"""Matrix multiplication 2x3x5 with Strassen optimization."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_row, v_row, w_row, coeff=1):
        """Add a single rank-1 term to the decomposition."""
        U_cols.append([0] * (n * m))
        V_cols.append([0] * (m * p))
        W_cols.append([0] * (p * n))
        U_cols[-1][u_row] = coeff
        V_cols[-1][v_row] = coeff
        W_cols[-1][w_row] = coeff
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, scale_a=1, scale_b=1):
        """
        Add Strassen decomposition for 2x2 * 2x2 -> 2x2.
        a_offset: (i0, j0) top-left of A submatrix
        b_offset: (j0, k0) top-left of B submatrix
        c_offset: (i0, k0) top-left of C submatrix
        """
        ai0, aj0 = a_offset
        bj0, bk0 = b_offset
        ci0, ck0 = c_offset
        
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1 = (A11+A22)(B11+B22)
        # A part: A[ai0,aj0] + A[ai0+1,aj0+1]
        # B part: B[bj0,bk0] + B[bj0+1,bk0+1]
        # This is not a single rank-1 term in the tensor sense.
        # We need to expand: (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # But that's 4 terms, which defeats the purpose.
        
        # Actually, in the tensor decomposition framework, each "multiplication" in Strassen
        # corresponds to a rank-1 term where the U, V, W vectors have multiple non-zero entries.
        # Let me reconsider.
        
        # For M1 = (A11+A22)(B11+B22):
        # U vector has 1 at positions corresponding to A11 and A22
        # V vector has 1 at positions corresponding to B11 and B22
        # W vector has 1 at positions corresponding to C11, C22 (with appropriate signs)
        
        # This is getting complex. Let me use a different approach.
        # I'll directly enumerate the rank-1 terms for each Strassen multiplication.
        
        pass
    
    # Let me take a cleaner approach: directly construct the rank-1 terms.
    # For the full 2x3x5 decomposition with Strassen on 2x2x2 blocks.
    
    # Reset
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_entries, v_entries, w_entries):
        """
        u_entries: list of (row_idx, value) for U matrix
        v_entries: list of (row_idx, value) for V matrix  
        w_entries: list of (row_idx, value) for W matrix
        """
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # A is 2x3: A[i,j] where i in {0,1}, j in {0,1,2}
    # B is 3x5: B[j,k] where j in {0,1,2}, k in {0,1,2,3,4}
    # C is 2x5: C[i,k] = sum_j A[i,j]*B[j,k]
    # U row index: i*m + j = i*3 + j
    # V row index: j*p + k = j*5 + k
    # W row index: k*n + i = k*2 + i
    
    # Part 1: A[:,0:2] * B[0:2,:] (2x2 * 2x5)
    # Split B[0:2,:] into B[0:2,0:3] (2x3) and B[0:2,3:5] (2x2)
    
    # Part 1a: A[:,0:2] * B[0:2,0:3] (2x2 * 2x3)
    # Split B[0:2,0:3] into B[0:2,0:2] (2x2) and B[0:2,2:3] (2x1)
    
    # Part 1a-i: A[:,0:2] * B[0:2,0:2] (2x2 * 2x2) -> Strassen, rank 7
    # A submatrix: A[0:2, 0:2], so A[i,j] for i in {0,1}, j in {0,1}
    # B submatrix: B[0:2, 0:2], so B[j,k] for j in {0,1}, k in {0,1}
    # C submatrix: C[0:2, 0:2], so C[i,k] for i in {0,1}, k in {0,1}
    
    # Strassen for this block:
    # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
    # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
    # C11 = C[0,0], C12 = C[0,1], C21 = C[1,0], C22 = C[1,1]
    
    # U indices: A[i,j] -> i*3+j
    #   A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # V indices: B[j,k] -> j*5+k
    #   B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    # W indices: C[i,k] -> k*2+i
    #   C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A11+A22)(B11+B22)
    # U: A[0,0]=0 and A[1,1]=4
    # V: B[0,0]=0 and B[1,1]=6
    # W: C[0,0]=0 (+1), C[1,1]=3 (+1)
    add_rank1([(0,1),(4,1)], [(0,1),(6,1)], [(0,1),(3,1)])
    
    # M2 = (A21+A22)B11
    # U: A[1,0]=3 and A[1,1]=4
    # V: B[0,0]=0
    # W: C[1,0]=2 (+1), C[1,1]=3 (-1)
    add_rank1([(3,1),(4,1)], [(0,1)], [(2,1),(3,-1)])
    
    # M3 = A11(B12-B22)
    # U: A[0,0]=0
    # V: B[0,1]=1 (+1), B[1,1]=6 (-1)
    # W: C[0,1]=1 (+1), C[1,1]=3 (+1)
    add_rank1([(0,1)], [(1,1),(6,-1)], [(1,1),(3,1)])
    
    # M4 = A22(B21-B11)
    # U: A[1,1]=4
    # V: B[1,0]=5 (+1), B[0,0]=0 (-1)
    # W: C[0,0]=0 (+1), C[1,0]=2 (+1)
    add_rank1([(4,1)], [(5,1),(0,-1)], [(0,1),(2,1)])
    
    # M5 = (A11+A12)B22
    # U: A[0,0]=0 and A[0,1]=1
    # V: B[1,1]=6
    # W: C[0,0]=0 (-1), C[0,1]=1 (+1)
    add_rank1([(0,1),(1,1)], [(6,1)], [(0,-1),(1,1)])
    
    # M6 = (A21-A11)(B11+B12)
    # U: A[1,0]=3 (+1), A[0,0]=0 (-1)
    # V: B[0,0]=0 and B[0,1]=1
    # W: C[1,1]=3 (+1)
    add_rank1([(3,1),(0,-1)], [(0,1),(1,1)], [(3,1)])
    
    # M7 = (A12-A22)(B21+B22)
    # U: A[0,1]=1 (+1), A[1,1]=4 (-1)
    # V: B[1,0]=5 and B[1,1]=6
    # W: C[0,0]=0 (+1)
    add_rank1([(1,1),(4,-1)], [(5,1),(6,1)], [(0,1)])
    
    # Part 1a-ii: A[:,0:2] * B[0:2,2:3] (2x2 * 2x1) -> Schoolbook, rank 4
    # A[i,j] for i in {0,1}, j in {0,1}
    # B[j,2] for j in {0,1}
    # C[i,2] for i in {0,1}
    
    # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # V indices: B[0,2]=2, B[1,2]=7
    # W indices: C[0,2]=4, C[1,2]=5
    
    add_rank1([(0,1)], [(2,1)], [(4,1)])  # A[0,0]*B[0,2] -> C[0,2]
    add_rank1([(1,1)], [(2,1)], [(4,1)])  # A[0,1]*B[0,2] -> C[0,2]
    add_rank1([(3,1)], [(7,1)], [(5,1)])  # A[1,0]*B[1,2] -> C[1,2]
    add_rank1([(4,1)], [(7,1)], [(5,1)])  # A[1,1]*B[1,2] -> C[1,2]
    
    # Part 1b: A[:,0:2] * B[0:2,3:5] (2x2 * 2x2) -> Strassen, rank 7
    # A[i,j] for i in {0,1}, j in {0,1}
    # B[j,k] for j in {0,1}, k in {3,4}
    # C[i,k] for i in {0,1}, k in {3,4}
    
    # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # V indices: B[0,3]=3, B[0,4]=4, B[1,3]=8, B[1,4]=9
    # W indices: C[0,3]=6, C[0,4]=7, C[1,3]=8, C[1,4]=9
    
    # M1 = (A11+A22)(B11+B22)
    add_rank1([(0,1),(4,1)], [(3,1),(9,1)], [(6,1),(9,1)])
    
    # M2 = (A21+A22)B11
    add_rank1([(3,1),(4,1)], [(3,1)], [(8,1),(9,-1)])
    
    # M3 = A11(B12-B22)
    add_rank1([(0,1)], [(4,1),(9,-1)], [(7,1),(9,1)])
    
    # M4 = A22(B21-B11)
    add_rank1([(4,1)], [(8,1),(3,-1)], [(6,1),(8,1)])
    
    # M5 = (A11+A12)B22
    add_rank1([(0,1),(1,1)], [(9,1)], [(6,-1),(7,1)])
    
    # M6 = (A21-A11)(B11+B12)
    add_rank1([(3,1),(0,-1)], [(3,1),(4,1)], [(9,1)])
    
    # M7 = (A12-A22)(B21+B22)
    add_rank1([(1,1),(4,-1)], [(8,1),(9,1)], [(6,1)])
    
    # Part 2: A[:,2:3] * B[2:3,:] (2x1 * 1x5) -> Schoolbook, rank 10
    # A[i,2] for i in {0,1}
    # B[2,k] for k in {0,1,2,3,4}
    # C[i,k] for i in {0,1}, k in {0,1,2,3,4}
    
    # U indices: A[0,2]=2, A[1,2]=5
    # V indices: B[2,0]=10, B[2,1]=11, B[2,2]=12, B[2,3]=13, B[2,4]=14
    # W indices: C[0,0]=0, C[0,1]=1, C[0,2]=4, C[0,3]=6, C[0,4]=7
    #            C[1,0]=2, C[1,1]=3, C[1,2]=5, C[1,3]=8, C[1,4]=9
    
    for i in range(2):
        for k in range(5):
            u_idx = i * 3 + 2  # A[i,2]
            v_idx = 2 * 5 + k  # B[2,k]
            w_idx = k * 2 + i  # C[i,k]
            add_rank1([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    # Build final matrices
    rank = len(U_cols)
    U = [U_cols[c] for c in range(rank)]
    V = [V_cols[c] for c in range(rank)]
    W = [W_cols[c] for c in range(rank)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
