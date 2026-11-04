"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5
    
    # We'll build the decomposition by partitioning:
    # n=3: [2, 1]
    # m=4: [2, 2]  
    # p=5: [2, 2, 1]
    
    # For each sub-block, we use either Strassen (2x2x2, rank 7) or schoolbook
    # and map indices accordingly.
    
    # Strassen 2x2x2 decomposition:
    # C = A*B where A,B are 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U entries for Strassen (A is 2x2, indexed as A[0,0],A[0,1],A[1,0],A[1,1]):
    # V entries for Strassen (B is 2x2):
    # W entries for Strassen (C is 2x2):
    
    def strassen_entries():
        """Return (U_coeffs, V_coeffs, W_coeffs) for 2x2x2 Strassen.
        U maps A entries to 7 intermediates, V maps B entries to 7 intermediates,
        W maps 7 intermediates to C entries."""
        # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        U = [[0]*7 for _ in range(4)]  # 4 A-entries, 7 intermediates
        V = [[0]*7 for _ in range(4)]  # 4 B-entries, 7 intermediates
        W = [[0]*7 for _ in range(4)]  # 4 C-entries, 7 intermediates
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1  # A[0,0]
        U[3][0] = 1  # A[1,1]
        V[0][0] = 1  # B[0,0]
        V[3][0] = 1  # B[1,1]
        
        # M2 = (A21+A22)B11
        U[2][1] = 1  # A[1,0]
        U[3][1] = 1  # A[1,1]
        V[0][1] = 1  # B[0,0]
        
        # M3 = A11(B21-B22)
        U[0][2] = 1  # A[0,0]
        V[2][2] = 1  # B[1,0]
        V[3][2] = -1 # B[1,1]
        
        # M4 = A22(B12-B11)
        U[3][3] = 1  # A[1,1]
        V[1][3] = 1  # B[0,1]
        V[0][3] = -1 # B[0,0]
        
        # M5 = (A11+A12)B22
        U[0][4] = 1  # A[0,0]
        U[1][4] = 1  # A[0,1]
        V[3][4] = 1  # B[1,1]
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1  # A[1,0]
        U[0][5] = -1 # A[0,0]
        V[0][5] = 1  # B[0,0]
        V[1][5] = 1  # B[0,1]
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1  # A[0,1]
        U[3][6] = -1 # A[1,1]
        V[2][6] = 1  # B[1,0]
        V[3][6] = 1  # B[1,1]
        
        # C = M1+M4-M5+M7, M3+M5, M2+M4, M1-M2+M3+M6
        W[0][0] = 1  # C[0,0] from M1
        W[0][3] = 1  # C[0,0] from M4
        W[0][4] = -1 # C[0,0] from M5
        W[0][6] = 1  # C[0,0] from M7
        
        W[1][2] = 1  # C[0,1] from M3
        W[1][4] = 1  # C[0,1] from M5
        
        W[2][1] = 1  # C[1,0] from M2
        W[2][3] = 1  # C[1,0] from M4
        
        W[3][0] = 1  # C[1,1] from M1
        W[3][1] = -1 # C[1,1] from M2
        W[3][2] = 1  # C[1,1] from M3
        W[3][5] = 1  # C[1,1] from M6
        
        return U, V, W
    
    # Build the full decomposition
    # Partition: n=[2,1], m=[2,2], p=[2,2,1]
    n_parts = [2, 1]
    m_parts = [2, 2]
    p_parts = [2, 2, 1]
    
    # Compute offsets
    n_offsets = [0]
    for i in range(len(n_parts)-1):
        n_offsets.append(n_offsets[-1] + n_parts[i])
    m_offsets = [0]
    for i in range(len(m_parts)-1):
        m_offsets.append(m_offsets[-1] + m_parts[i])
    p_offsets = [0]
    for i in range(len(p_parts)-1):
        p_offsets.append(p_offsets[-1] + p_parts[i])
    
    # Collect all sub-problems
    subproblems = []
    for ni in range(len(n_parts)):
        for mi in range(len(m_parts)):
            for pi in range(len(p_parts)):
                sn = n_parts[ni]
                sm = m_parts[mi]
                sp = p_parts[pi]
                subproblems.append((ni, mi, pi, sn, sm, sp))
    
    # For each subproblem, determine rank and generate entries
    # 2x2x2: rank 7 (Strassen)
    # 2x2x1: rank 4 (schoolbook)
    # 1x2x2: rank 4 (schoolbook)
    # 1x2x1: rank 2 (schoolbook)
    # 2x1x2: rank 4 (schoolbook) - not present here
    # 1x1x2: rank 2 (schoolbook) - not present here
    # 1x1x1: rank 1
    
    def get_subproblem_decomp(sn, sm, sp):
        """Return (rank, U_sub, V_sub, W_sub) for sn x sm x sp.
        U_sub: (sn*sm) x rank, V_sub: (sm*sp) x rank, W_sub: (sp*sn) x rank
        """
        if sn == 2 and sm == 2 and sp == 2:
            U_s, V_s, W_s = strassen_entries()
            return 7, U_s, V_s, W_s
        else:
            # Schoolbook
            rank = sn * sm * sp
            U_sub = [[0]*rank for _ in range(sn*sm)]
            V_sub = [[0]*rank for _ in range(sm*sp)]
            W_sub = [[0]*rank for _ in range(sp*sn)]
            col = 0
            for i in range(sn):
                for j in range(sm):
                    for k in range(sp):
                        U_sub[i*sm+j][col] = 1
                        V_sub[j*sp+k][col] = 1
                        W_sub[k*sn+i][col] = 1
                        col += 1
            return rank, U_sub, V_sub, W_sub
    
    # First pass: compute total rank
    total_rank = 0
    sub_ranks = []
    for (ni, mi, pi, sn, sm, sp) in subproblems:
        r, _, _, _ = get_subproblem_decomp(sn, sm, sp)
        sub_ranks.append(r)
        total_rank += r
    
    # Second pass: build full matrices
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for idx, (ni, mi, pi, sn, sm, sp) in enumerate(subproblems):
        r, U_sub, V_sub, W_sub = get_subproblem_decomp(sn, sm, sp)
        
        no = n_offsets[ni]
        mo = m_offsets[mi]
        po = p_offsets[pi]
        
        for i in range(sn):
            for j in range(sm):
                for c in range(r):
                    U[(no+i)*m + (mo+j)][col_offset + c] = U_sub[i*sm+j][c]
        
        for j in range(sm):
            for k in range(sp):
                for c in range(r):
                    V[(mo+j)*p + (po+k)][col_offset + c] = V_sub[j*sp+k][c]
        
        for k in range(sp):
            for i in range(sn):
                for c in range(r):
                    W[(po+k)*n + (no+i)][col_offset + c] = W_sub[k*sn+i][c]
        
        col_offset += r
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
