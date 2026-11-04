import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors
    # U, V, W are 4x7
    # Row order for U, V: A00, A01, A10, A11 (row-major)
    # Row order for W: C00, C10, C01, C11 (col-major: k*n+i)
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ])
    
    S_V = np.array([
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 0, 1],   # B01
        [0, 0, 1, -1, 0, 1, 0],  # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ])
    
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00 (i=0, k=0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (i=1, k=0)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (i=0, k=1)
        [1, -1, 1, 0, 0, 1, 0]   # C11 (i=1, k=1)
    ])
    
    # Generate 4x4 Strassen factors via Kronecker Product
    # U44 = S_U (x) S_U
    # Row index r = r1 * 4 + r2
    # r1 is row in S_U (0..3), r2 is row in S_U (0..3)
    # This corresponds to A[i, j] where i = i1*2+i2, j = j1*2+j2
    # r1 = i1*2+j1, r2 = i2*2+j2
    # We need to map local row index (i*4+j) to Kronecker row index (r1*4+r2)
    
    def get_kronecker_row(i, j):
        i1 = i // 2
        i2 = i % 2
        j1 = j // 2
        j2 = j % 2
        r1 = i1 * 2 + j1
        r2 = i2 * 2 + j2
        return r1 * 4 + r2

    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for i in range(4):
        for j in range(4):
            r_local = i * 4 + j
            r_kron = get_kronecker_row(i, j)
            
            # Extract rows from S_U, S_V
            u_row = S_U[r_kron // 4, :]
            u_sub = S_U[r_kron % 4, :]
            
            v_row = S_V[r_kron // 4, :]
            v_sub = S_V[r_kron % 4, :]
            
            w_row = S_W[r_kron // 4, :]
            w_sub = S_W[r_kron % 4, :]
            
            # Kronecker product of vectors
            U44[r_local, :] = np.kron(u_row, u_sub)
            V44[r_local, :] = np.kron(v_row, v_sub)
            W44[r_local, :] = np.kron(w_row, w_sub)
            
    # Initialize Global Matrices
    total_rank = 49 + 32 + 32 + 16 + 32 + 16 + 16 + 7
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Helper to get global row index for U (A)
    def u_idx(i, j):
        return i * 6 + j
        
    # Helper to get global row index for V (B)
    def v_idx(i, j):
        return i * 6 + j
        
    # Helper to get global row index for W (C)
    # Convention: W[k*n + i]
    def w_idx(i, k):
        return k * 6 + i

    # Block 1: A11 B11 -> C11 (Rank 49)
    offset = 0
    for r in range(16):
        i = r // 4
        j = r % 4
        U[u_idx(i, j), offset:offset+49] = U44[r, :]
        V[v_idx(i, j), offset:offset+49] = V44[r, :]
        # W44 row r corresponds to C[i, j] ?
        # In 4x4, W44 row r_local = i*4+j corresponds to C[i, j]
        # But W44 was constructed from S_W which is col-major.
        # Let's check W44 construction.
        # W44[r_local, :] = kron(w_row, w_sub).
        # w_row is S_W[r1], w_sub is S_W[r2].
        # S_W rows are col-major: 0:C00, 1:C10, 2:C01, 3:C11.
        # So W44 row r_local corresponds to C[i, j] where i, j are derived from r_local?
        # r_local = i*4+j.
        # r_kron = r1*4+r2.
        # r1 = i1*2+j1, r2 = i2*2+j2.
        # S_W[r1] corresponds to C_block[i1, j1] in col-major?
        # S_W row index r1 = k1*2 + i1.
        # So r1 = j1*2 + i1? No, S_W row 0 is C00 (i=0,k=0). Row 1 is C10 (i=1,k=0).
        # So S_W row index = k*2 + i.
        # So r1 = j1*2 + i1.
        # r2 = j2*2 + i2.
        # r_kron = r1*4 + r2 = (j1*2+i1)*4 + (j2*2+i2).
        # This is NOT i*4+j.
        # So W44 rows are permuted relative to U44 rows.
        # We need to map W44 row index to global W index correctly.
        
        # Let's re-evaluate W44 row mapping.
        # W44 row r_kron corresponds to C[i, j] where:
        # i = i1*2 + i2, j = j1*2 + j2.
        # r1 = j1*2 + i1, r2 = j2*2 + i2.
        # So for a given global C[i, j], we need to find r_kron.
        # i1 = i//2, i2 = i%2. j1 = j//2, j2 = j%2.
        # r1 = j1*2 + i1. r2 = j2*2 + i2.
        # r_kron = r1*4 + r2.
        
        # In the loop above, we iterated r_local (i*4+j) and set U44, V44, W44.
        # For U and V, the row index in S_U/S_V is i*2+j (row-major).
        # So r1 = i1*2+j1, r2 = i2*2+j2.
        # This matches get_kronecker_row(i, j).
        # So U44[r_local] and V44[r_local] are correct for A[i, j] and B[i, j].
        
        # For W, S_W row index is k*2+i (col-major).
        # So r1 = j1*2 + i1, r2 = j2*2 + i2.
        # The row in W44 corresponding to C[i, j] is at index r_w_kron = (j1*2+i1)*4 + (j2*2+i2).
        # But we stored W44 in the loop using r_kron = (i1*2+j1)*4 + (i2*2+j2).
        # So W44[r_local] is actually W for C[i', j'] where i', j' are swapped in some way?
        # No, W44[r_local] contains the factors for the term corresponding to r_kron.
        # The term r_kron corresponds to C[i_w, j_w] where:
        # i_w = i1*2+i2, j_w = j1*2+j2.
        # But r_kron was calculated from i, j using row-major logic for U/V.
        # So W44[r_local] is the factor for C[i, j] ONLY IF the indexing of S_W matches S_U.
        # It does not.
        
        # Correction: We must map the W44 row to the correct global W index.
        # The value W44[r_local, c] is the coefficient for C[i, j] ?
        # No. W44 was built by kron(w_row, w_sub) where w_row=S_W[r1], w_sub=S_W[r2].
        # r1, r2 were calculated from i, j using row-major logic (i1*2+j1).
        # But S_W[r1] is the row for C[k1, i1] where r1 = k1*2 + i1.
        # So S_W[i1*2+j1] is the row for C[j1, i1].
        # So W44[r_local] is actually the factor for C[j, i] (transposed)?
        # Let's check.
        # We want factor for C[i, j].
        # We need S_W row for C[i1, j1] which is index j1*2 + i1.
        # But we used S_W row index i1*2 + j1.
        # So we used the wrong rows from S_W.
        
        # We need to regenerate W44 or permute it.
        # Let's create a separate W44_correct.
        
    # Regenerate W44 correctly
    W44_corr = np.zeros((16, 49))
    for i in range(4):
        for j in range(4):
            r_local = i * 4 + j
            i1 = i // 2
            i2 = i % 2
            j1 = j // 2
            j2 = j % 2
            
            # S_W row index for C[i1, j1] is j1*2 + i1
            r1_w = j1 * 2 + i1
            r2_w = j2 * 2 + i2
            
            w_row = S_W[r1_w, :]
            w_sub = S_W[r2_w, :]
            
            W44_corr[r_local, :] = np.kron(w_row, w_sub)
            
    # Now fill Block 1
    for r in range(16):
        i = r // 4
        j = r % 4
        U[u_idx(i, j), offset:offset+49] = U44[r, :]
        V[v_idx(i, j), offset:offset+49] = V44[r, :]
        # W44_corr[r] is factor for C[i, j]
        # Global W index for C[i, j] is w_idx(i, j)
        W[w_idx(i, j), offset:offset+49] = W44_corr[r, :]
        
    offset += 49
    
    # Block 2: A12 B21 -> C11 (Rank 32)
    # A12: i in 0..3, k in 0..1. Global U: i*6 + 4+k
    # B21: k in 0..1, l in 0..3. Global V: 4+k*6 + l
    # C11: i in 0..3, l in 0..3. Global W: l*6 + i
    for i in range(4):
        for k in range(2):
            for l in range(4):
                u_r = u_idx(i, 4+k)
                v_r = v_idx(4+k, l)
                w_r = w_idx(i, l)
                U[u_r, offset] = 1
                V[v_r, offset] = 1
                W[w_r, offset] = 1
                offset += 1
                
    # Block 3: A11 B12 -> C12 (Rank 32)
    # A11: i in 0..3, k in 0..3. Global U: i*6 + k
    # B12: k in 0..3, l in 0..1. Global V: k*6 + 4+l
    # C12: i in 0..3, l in 0..1. Global W: 4+l*6 + i
    for i in range(4):
        for k in range(4):
            for l in range(2):
                u_r = u_idx(i, k)
                v_r = v_idx(k, 4+l)
                w_r = w_idx(i, 4+l)
                U[u_r, offset] = 1
                V[v_r, offset] = 1
                W[w_r, offset] = 1
                offset += 1

    # Block 4: A12 B22 -> C12 (Rank 16)
    # A12: i in 0..3, k in 0..1. Global U: i*6 + 4+k
    # B22: k in 0..1, l in 0..1. Global V: 4+k*6 + 4+l
    # C12: i in 0..3, l in 0..1. Global W: 4+l*6 + i
    for i in range(4):
        for k in range(2):
            for l in range(2):
                u_r = u_idx(i, 4+k)
                v_r = v_idx(4+k, 4+l)
                w_r = w_idx(i, 4+l)
                U[u_r, offset] = 1
                V[v_r, offset] = 1
                W[w_r, offset] = 1
                offset += 1

    # Block 5: A21 B11 -> C21 (Rank 32)
    # A21: i in 0..1, k in 0..3. Global U: 4+i*6 + k
    # B11: k in 0..3, l in 0..3. Global V: k*6 + l
    # C21: i in 0..1, l in 0..3. Global W: l*6 + 4+i
    for i in range(2):
        for k in range(4):
            for l in range(4):
                u_r = u_idx(4+i, k)
                v_r = v_idx(k, l)
                w_r = w_idx(4+i, l)
                U[u_r, offset] = 1
                V[v_r, offset] = 1
                W[w_r, offset] = 1
                offset += 1

    # Block 6: A22 B21 -> C21 (Rank 16)
    # A22: i in 0..1, k in 0..1. Global U: 4+i*6 + 4+k
    # B21: k in 0..1, l in 0..3. Global V: 4+k*6 + l
    # C21: i in 0..1, l in 0..3. Global W: l*6 + 4+i
    for i in range(2):
        for k in range(2):
            for l in range(4):
                u_r = u_idx(4+i, 4+k)
                v_r = v_idx(4+k, l)
                w_r = w_idx(4+i, l)
                U[u_r, offset] = 1
                V[v_r, offset] = 1
                W[w_r, offset] = 1
                offset += 1

    # Block 7: A21 B12 -> C22 (Rank 16)
    # A21: i in 0..1, k in 0..3. Global U: 4+i*6 + k
    # B12: k in 0..3, l in 0..1. Global V: k*6 + 4+l
    # C22: i in 0..1, l in 0..1. Global W: 4+l*6 + 4+i
    for i in range(2):
        for k in range(4):
            for l in range(2):
                u_r = u_idx(4+i, k)
                v_r = v_idx(k, 4+l)
                w_r = w_idx(4+i, 4+l)
                U[u_r, offset] = 1
                V[v_r, offset] = 1
                W[w_r, offset] = 1
                offset += 1

    # Block 8: A22 B22 -> C22 (Rank 7)
    # Use S_U, S_V, S_W
    # S_U row r (0..3) corresponds to A22[i, k] where r = i*2+k
    # S_V row r (0..3) corresponds to B22[i, k] where r = i*2+k
    # S_W row r (0..3) corresponds to C22[i, k] where r = k*2+i
    
    for c in range(7):
        for r in range(4):
            val_u = S_U[r, c]
            if val_u != 0:
                i = r // 2
                k = r % 2
                u_r = u_idx(4+i, 4+k)
                U[u_r, offset+c] = val_u
                
            val_v = S_V[r, c]
            if val_v != 0:
                i = r // 2
                k = r % 2
                v_r = v_idx(4+i, 4+k)
                V[v_r, offset+c] = val_v
                
            # S_W row r corresponds to C[i, k] where r = k*2+i
            # We need to find i, k from r
            k = r // 2
            i = r % 2
            val_w = S_W[r, c]
            if val_w != 0:
                w_r = w_idx(4+i, 4+k)
                W[w_r, offset+c] = val_w

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
