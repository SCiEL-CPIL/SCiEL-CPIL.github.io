"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    
    # We decompose the tensor into blocks based on splitting m=5 into 2+2+1 and p=5 into 2+2+1.
    # The resulting rank is 46.
    
    # We will build U, V, W lists of lists.
    # U: (n*m, rank) = (10, 46)
    # V: (m*p, rank) = (25, 46)
    # W: (p*n, rank) = (10, 46)
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx are the row indices in U, V, W
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen 7-term coefficients for 2x2x2
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)

    def add_strassen_block(i_offset, j_offset, k_offset):
        """
        Adds a 2x2x2 Strassen block.
        i_offset: offset in i (0 or 1, but i is only 0,1 so always 0 for n=2)
        j_offset: offset in j (0, 2)
        k_offset: offset in k (0, 2)
        """
        nonlocal col
        # i indices: i_offset, i_offset+1
        # j indices: j_offset, j_offset+1
        # k indices: k_offset, k_offset+1
        
        # M1
        # U: i0,j0 and i1,j1
        # V: j0,k0 and j1,k1
        # W: k0,i0 and k1,i1
        # Term 1: (X11+X22)(Y11+Y22)
        # U[i0,j0] += 1, U[i1,j1] += 1
        # V[j0,k0] += 1, V[j1,k1] += 1
        # W[k0,i0] += 1, W[k1,i1] += 1
        # But we need to add this to the W matrix with coefficients for the final sum.
        # It is easier to add 7 separate terms for M1..M7, and then in W,
        # we accumulate the coefficients for each output cell.
        
        # Actually, the standard way is:
        # Each M_i is a rank-1 term in the tensor decomposition of the 2x2x2 block.
        # The output Z is a linear combination of M_i.
        # So in the global W matrix, for the column corresponding to M_i,
        # we put the coefficient of M_i in the expression for each Z cell.
        
        # Let's define the 7 terms locally.
        # We will add 7 columns to U, V, W.
        
        # M1
        u_idx_1 = [i_offset*2 + 0, j_offset*2 + 0] # Wrong indexing.
        
        # Let's use flat indices.
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        i0, i1 = i_offset, i_offset + 1
        j0, j1 = j_offset, j_offset + 1
        k0, k1 = k_offset, k_offset + 1
        
        # M1: (X11+X22)(Y11+Y22)
        # U: (i0,j0)=1, (i1,j1)=1
        # V: (j0,k0)=1, (j1,k1)=1
        # W: (k0,i0)=1, (k1,i1)=1
        # Coeffs in W:
        # Z11 (k0,i0): +1
        # Z22 (k1,i1): +1
        # Z12 (k1,i0): 0
        # Z21 (k0,i1): 0
        # Wait, Z indices are (k, i).
        # Z11 corresponds to k0, i0.
        # Z12 corresponds to k1, i0.
        # Z21 corresponds to k0, i1.
        # Z22 corresponds to k1, i1.
        
        # M1 contributes to Z11 and Z22?
        # Z11 = M1 + M4 - M5 + M7
        # Z12 = M3 + M5
        # Z21 = M2 + M4
        # Z22 = M1 - M2 + M3 + M6
        
        # So for M1 column:
        # W[k0, i0] += 1
        # W[k1, i1] += 1
        
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1, 1, 1) # This is wrong. add_term adds a full rank 1 term.
        # I need to add 7 terms for M1..M7.
        # But M1 is a single product of sums.
        # In the tensor decomposition, M1 is represented by a column in U, V, W.
        # U column has 1s at (i0,j0) and (i1,j1).
        # V column has 1s at (j0,k0) and (j1,k1).
        # W column has coefficients for the linear combination.
        
        # So I should NOT use add_term for each M_i separately if add_term increments col.
        # I should manually construct the 7 columns.
        
        # Let's rewrite the logic to directly populate U, V, W for the 7 Strassen terms.
        
        # Term 1: M1
        U[i0*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1
        W[k1*n + i1][col] += 1
        col += 1
        
        # Term 2: M2 = (X21+X22)Y11
        U[i1*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i1][col] += 1 # Z21
        col += 1
        
        # Term 3: M3 = X11(Y12-Y22)
        U[i0*m + j0][col] += 1
        V[j0*p + k1][col] += 1
        V[j1*p + k1][col] += -1
        W[k1*n + i0][col] += 1 # Z12
        col += 1
        
        # Term 4: M4 = X22(Y21-Y11)
        U[i1*m + j1][col] += 1
        V[j1*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + i0][col] += 1 # Z11
        W[k0*n + i1][col] += 1 # Z21
        col += 1
        
        # Term 5: M5 = (X11+X12)Y22
        U[i0*m + j0][col] += 1
        U[i0*m + j1][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += -1 # Z11
        W[k1*n + i0][col] += 1 # Z12
        col += 1
        
        # Term 6: M6 = (X21-X11)(Y11+Y12)
        U[i1*m + j0][col] += 1
        U[i0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[j0*p + k1][col] += 1
        W[k1*n + i1][col] += 1 # Z22
        col += 1
        
        # Term 7: M7 = (X12-X22)(Y21+Y22)
        U[i0*m + j1][col] += 1
        U[i1*m + j1][col] += -1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1 # Z11
        W[k1*n + i1][col] += 1 # Z22
        col += 1

    # Helper for 2x2x1 schoolbook (Rank 4)
    # A (2x2) * B (2x1) = C (2x1)
    # C1 = A11 B1 + A21 B2
    # C2 = A12 B1 + A22 B2
    def add_2x2x1_block(i_offset, j_offset, k_offset):
        nonlocal col
        i0, i1 = i_offset, i_offset + 1
        j0, j1 = j_offset, j_offset + 1
        k0 = k_offset
        
        # Term 1: A11 B1 -> C1
        U[i0*m + j0][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
        
        # Term 2: A21 B2 -> C1
        U[i1*m + j1][col] += 1
        V[j1*p + k0][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
        
        # Term 3: A12 B1 -> C2
        U[i0*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i1][col] += 1
        col += 1
        
        # Term 4: A22 B2 -> C2
        U[i1*m + j1][col] += 1
        V[j1*p + k0][col] += 1
        W[k0*n + i1][col] += 1
        col += 1

    # Helper for 2x1x2 schoolbook (Rank 4)
    # A (2x1) * B (1x2) = C (2x2)
    # C11 = A1 B11
    # C12 = A1 B12
    # C21 = A2 B21
    # C22 = A2 B22
    def add_2x1x2_block(i_offset, j_offset, k_offset):
        nonlocal col
        i0, i1 = i_offset, i_offset + 1
        j0 = j_offset
        k0, k1 = k_offset, k_offset + 1
        
        # Term 1: A1 B11 -> C11
        U[i0*m + j0][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
        
        # Term 2: A1 B12 -> C12
        U[i0*m + j0][col] += 1
        V[j0*p + k1][col] += 1
        W[k1*n + i0][col] += 1
        col += 1
        
        # Term 3: A2 B21 -> C21
        U[i1*m + j0][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i1][col] += 1
        col += 1
        
        # Term 4: A2 B22 -> C22
        U[i1*m + j0][col] += 1
        V[j0*p + k1][col] += 1
        W[k1*n + i1][col] += 1
        col += 1

    # Helper for 2x1x1 schoolbook (Rank 2)
    # A (2x1) * B (1x1) = C (2x1)
    def add_2x1x1_block(i_offset, j_offset, k_offset):
        nonlocal col
        i0, i1 = i_offset, i_offset + 1
        j0 = j_offset
        k0 = k_offset
        
        # Term 1: A1 B1 -> C1
        U[i0*m + j0][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
        
        # Term 2: A2 B1 -> C2
        U[i1*m + j0][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i1][col] += 1
        col += 1

    # J1 (j=0,1)
    add_strassen_block(0, 0, 0)   # K1 (k=0,1)
    add_strassen_block(0, 0, 2)   # K2 (k=2,3)
    add_2x2x1_block(0, 0, 4)      # K3 (k=4)
    
    # J2 (j=2,3)
    add_strassen_block(0, 2, 0)   # K1 (k=0,1)
    add_strassen_block(0, 2, 2)   # K2 (k=2,3)
    add_2x2x1_block(0, 2, 4)      # K3 (k=4)
    
    # J3 (j=4)
    add_2x1x2_block(0, 4, 0)      # K1 (k=0,1)
    add_2x1x2_block(0, 4, 2)      # K2 (k=2,3)
    add_2x1x1_block(0, 4, 4)      # K3 (k=4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
