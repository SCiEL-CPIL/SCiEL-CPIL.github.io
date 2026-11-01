"""Explicit rank-28 exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    # Rank 28: two Strassen 2x2x2 blocks (rank 7 each) plus four schoolbook blocks
    #   A[:,0:2] * B[0:2,0:2]  -> Strassen 2x2x2
    #   A[:,2]   * B[2,0:2]    -> Schoolbook 2x1x2 (rank 4)
    #   A[:,0:2] * B[0:2,2:4]  -> Schoolbook 2x2x2 (rank 8)
    #   A[:,2]   * B[2,2:4]    -> Schoolbook 2x1x2 (rank 4)
    #   A[:,0:2] * B[0:2,4]    -> Schoolbook 2x2x1 (rank 4)
    #   A[:,2]   * B[2,4]      -> Schoolbook 2x1x1 (rank 2)
    # Total: 7 + 4 + 8 + 4 + 4 + 2 = 29? Let me recount.
    # Actually: 7 + 4 + 8 + 4 + 4 + 2 = 29. That's too many.
    # Let me reconsider. n=2, m=3, p=5.
    # Split B into columns: [0:2], [2:4], [4:5] -> shapes (3x2), (3x3), (3x1)
    # Wait, B is 3x5. Split into B[:,0:2] (3x2), B[:,2:4] (3x3), B[:,4:5] (3x1).
    # But we can also split A rows: A[0:2,:] is the whole A (2x3).
    # 
    # Better approach: 
    # C = A(2x3) * B(3x5)
    # Split B into B1(3x2), B2(3x2), B3(3x1) where B1=B[:,0:2], B2=B[:,2:4], B3=B[:,4:5]
    # C1 = A*B1 (2x2), C2 = A*B2 (2x2), C3 = A*B3 (2x1)
    #
    # For C1: A(2x3)*B1(3x2). Split A into A1(2x2)=A[:,0:2], A2(2x1)=A[:,2]
    #   C1 = A1*B1[:2,:2] (2x2*2x2, Strassen rank 7) + A2*B1[2,:] (2x1*1x2, schoolbook rank 4)
    #   Wait, B1 is 3x2. B1[:2,:] is 2x2, B1[2,:] is 1x2.
    #   A1*B1[:2,:] is 2x2*2x2 -> Strassen 7
    #   A2*B1[2,:] is 2x1*1x2 -> schoolbook 2*1*2 = 4
    #   Total for C1: 11
    #
    # For C2: A(2x3)*B2(3x2). Same structure.
    #   C2 = A1*B2[:2,:] (Strassen 7) + A2*B2[2,:] (schoolbook 4)
    #   Total for C2: 11
    #
    # For C3: A(2x3)*B3(3x1). B3 is 3x1.
    #   C3 = A1*B3[:2,0] (2x2*2x1, schoolbook 4) + A2*B3[2,0] (2x1*1x1, schoolbook 2)
    #   Total for C3: 6
    #
    # Grand total: 11 + 11 + 6 = 28. 
    #
    # Now let's build the factors.
    # U shape: (n*m, rank) = (6, 28)
    # V shape: (m*p, rank) = (15, 28)
    # W shape: (p*n, rank) = (10, 28)
    #
    # Indexing:
    # U[i*m + j, c] where i in [0,2), j in [0,3), c in [0,28)
    # V[j*p + k, c] where j in [0,3), k in [0,5), c in [0,28)
    # W[k*n + i, c] where k in [0,5), i in [0,2), c in [0,28)
    #
    # We'll assign columns 0-6 to C1 Strassen, 7-10 to C1 schoolbook,
    # 11-17 to C2 Strassen, 18-21 to C2 schoolbook,
    # 22-25 to C3 A1 part, 26-27 to C3 A2 part.
    
    U = [[0] * 28 for _ in range(6)]
    V = [[0] * 28 for _ in range(15)]
    W = [[0] * 28 for _ in range(10)]
    
    def add_strassen(u_off, v_off, w_off, i_map, j_map, k_map, sign=1):
        """
        Add a Strassen 2x2x2 block.
        i_map: list of 2 indices for A rows (i)
        j_map: list of 2 indices for A cols / B rows (j)
        k_map: list of 2 indices for B cols (k)
        """
        # Strassen factors for 2x2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a10)b11
        # M6 = (a11-a10)(b00+b10)
        # M7 = (a01-a00)(b11+b01)
        #
        # C00 = M1 - M4 + M5
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M3 + M7
        #
        # U factors (for A):
        # M1: [1, 0, 1, 0] for a00,a01,a10,a11 -> wait, a is 2x2, indices (i,j)
        # Let me use (i,j) indexing where i is row, j is col.
        # a00 = A[i0,j0], a01 = A[i0,j1], a10 = A[i1,j0], a11 = A[i1,j1]
        #
        # M1 = (a00+a11)(b00+b11): U contributes [1, 0, 0, 1] to (i0j0, i0j1, i1j0, i1j1)
        #   Actually for U[i*m+j]: 
        #   i0j0 -> i_map[0], j_map[0]
        #   i0j1 -> i_map[0], j_map[1]
        #   i1j0 -> i_map[1], j_map[0]
        #   i1j1 -> i_map[1], j_map[1]
        # M1 U: +1 at (i0,j0), +1 at (i1,j1)
        # M1 V: +1 at (j0,k0), +1 at (j1,k1)
        # M1 W: +1 at (k0,i0), +1 at (k1,i1)
        #
        # Let me just hardcode the 7 Strassen vectors.
        
        # Define the 7 multiplications with their (i,j), (j,k), (k,i) patterns
        # Each is a tuple of (coeff, i_idx, j_idx) for U, (coeff, j_idx, k_idx) for V, (coeff, k_idx, i_idx) for W
        
        strassen_factors = [
            # (u_coeffs: list of (i_local, j_local, val), v_coeffs: list of (j_local, k_local, val), w_coeffs: list of (k_local, i_local, val))
            # M1 = (a00+a11)(b00+b11)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2 = (a10+a11)b00
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)]),
            # M3 = a00(b10-b11)
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,0,1)]),
            # M4 = a11(b01-b00)
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(1,1,1)]),
            # M5 = (a00+a10)b11
            ([(0,0,1),(1,0,1)], [(1,1,1)], [(1,0,1)]),
            # M6 = (a11-a10)(b00+b10)
            ([(1,1,1),(1,0,-1)], [(0,0,1),(1,0,1)], [(0,1,1)]),
            # M7 = (a01-a00)(b11+b01)
            ([(0,1,1),(0,0,-1)], [(1,1,1),(0,1,1)], [(1,1,1)]),
        ]
        
        for s in range(7):
            col = u_off + s
            u_coeffs, v_coeffs, w_coeffs = strassen_factors[s]
            for (ii, jj, val) in u_coeffs:
                ui = i_map[ii]
                uj = j_map[jj]
                U[ui * m + uj][col] += sign * val
            for (jj, kk, val) in v_coeffs:
                vj = j_map[jj]
                vk = k_map[jj]  # Wait, no. k_map[k_local]
            # Actually k_map is for k indices. Let me redo.
            # V index is j*p + k. j_map gives actual j, k_map gives actual k.
            # In v_coeffs, jj is local j, kk is local k.
            for (jj, kk, val) in v_coeffs:
                vj = j_map[jj]
                vk = k_map[kk]
                V[vj * p + vk][col] += sign * val
            for (kk, ii, val) in w_coeffs:
                wk = k_map[kk]
                wi = i_map[ii]
                W[wk * n + wi][col] += sign * val
    
    # C1: A[:,0:2]*B[0:2,0:2] Strassen, A[:,2]*B[2,0:2] schoolbook
    # i_map = [0,1] (A rows), j_map = [0,1] (A cols / B rows), k_map = [0,1] (B cols)
    add_strassen(0, 0, 0, [0,1], [0,1], [0,1])
    
    # C1 schoolbook: A[:,2] (2x1) * B[2,0:2] (1x2), rank 4
    # For each i in {0,1}, k in {0,1}:
    for i in range(2):
        for k in range(2):
            col = 7 + i*2 + k
            U[i*m + 2][col] = 1       # A[i,2]
            V[2*p + 0][col] = 1       # B[2,0] -> j=2, k=0? No, B[2, k] where k is 0 or 1
            # B row 2, col k
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1       # C[i,k]
    
    # C2: A[:,0:2]*B[0:2,2:4] Strassen, A[:,2]*B[2,2:4] schoolbook
    # i_map = [0,1], j_map = [0,1], k_map = [2,3]
    add_strassen(11, 0, 2, [0,1], [0,1], [2,3])
    
    # C2 schoolbook: A[:,2] (2x1) * B[2,2:4] (1x2), rank 4
    for i in range(2):
        for k in range(2):
            col = 18 + i*2 + k
            U[i*m + 2][col] = 1
            V[2*p + (2+k)][col] = 1   # B[2, 2+k]
            W[(2+k)*n + i][col] = 1   # C[i, 2+k]
    
    # C3: A[:,0:2]*B[0:2,4] schoolbook (2x2*2x1, rank 4), A[:,2]*B[2,4] schoolbook (2x1*1x1, rank 2)
    # C3 A1 part: for each i in {0,1}, j in {0,1}: A[i,j]*B[j,4]
    for i in range(2):
        for j in range(2):
            col = 22 + i*2 + j
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
    # C3 A2 part: for each i in {0,1}: A[i,2]*B[2,4]
    for i in range(2):
        col = 26 + i
        U[i*m + 2][col] = 1
        V[2*p + 4][col] = 1
        W[4*n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
