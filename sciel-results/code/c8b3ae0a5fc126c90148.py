import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_2x2():
    # Returns U, V, W for 2x2x2 Strassen decomposition
    # Rank 7
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: A00, A01, A10, A11
    #          B00, B01, B10, B11
    #          C00, C01, C10, C11
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook(m, n, p):
    # Returns U, V, W for m x n x p schoolbook
    # U: (m*n, m*n*p), V: (n*p, m*n*p), W: (p*m, m*n*p)
    rank = m * n * p
    U = np.zeros((m * n, rank), dtype=int)
    V = np.zeros((n * p, rank), dtype=int)
    W = np.zeros((p * m, rank), dtype=int)
    
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j, col] = 1
                V[j * p + k, col] = 1
                W[k * m + i, col] = 1
                col += 1
    return U, V, W

def _concat_factors(factors_list):
    # factors_list is a list of (U, V, W, row_offset_U, row_offset_V, row_offset_W, col_offset_U, col_offset_V, col_offset_W)
    # This is getting complicated to manage offsets manually.
    # Better to build the full matrices directly.
    pass

def construct():
    n, m, p = DIMENSIONS
    # We are doing 5x5x5
    # Split into 4x4 and 1x1 blocks
    # A: 4x4, 4x1, 1x4, 1x1
    # B: 4x4, 4x1, 1x4, 1x1
    # C: 4x4, 4x1, 1x4, 1x1
    
    # Total rank
    r_444 = 49
    r_441 = 16
    r_414 = 16
    r_144 = 16
    r_411 = 4
    r_141 = 4
    r_114 = 4
    r_111 = 1
    
    total_rank = r_444 + r_441 + r_414 + r_144 + r_411 + r_141 + r_114 + r_111
    
    U = np.zeros((25, total_rank), dtype=int)
    V = np.zeros((25, total_rank), dtype=int)
    W = np.zeros((25, total_rank), dtype=int)
    
    col_offset = 0
    
    # Helper to add a factor
    def add_factor(U_loc, V_loc, W_loc, i_start, j_start, k_start):
        nonlocal col_offset
        # U_loc is (m*n, rank), V_loc is (n*p, rank), W_loc is (p*m, rank)
        # We need to place them into the global U, V, W
        # Global U index: i * 5 + j
        # Global V index: j * 5 + k
        # Global W index: k * 5 + i
        
        r = U_loc.shape[1]
        
        for u_i in range(U_loc.shape[0]):
            # u_i corresponds to local (i, j) in A
            # local i = u_i // n_loc, local j = u_i % n_loc
            # But we don't have n_loc here. We need to pass it.
            pass
        
    # Let's just do it manually for each block type to avoid abstraction bugs.
    
    # 1. C44 part 1: A44 * B44 (4x4x4 Strassen)
    # A44: rows 0-3, cols 0-3.
    # B44: rows 0-3, cols 0-3.
    # C44: rows 0-3, cols 0-3.
    
    U444, V444, W444 = _strassen_2x2()
    # Wait, _strassen_2x2 is for 2x2. We need 4x4.
    # We can use Kronecker product of Strassen 2x2 with Strassen 2x2?
    # Or just schoolbook 4x4x4? Rank 64.
    # Strassen 4x4x4 rank 49.
    # Let's implement Strassen 4x4x4 by decomposing 4x4 into 2x2 blocks.
    
    def _strassen_4x4():
        # Returns U, V, W for 4x4x4
        # Rank 49
        # We can construct this by taking the Kronecker product of the 2x2 Strassen factors?
        # No, the Kronecker product of tensors gives the tensor for the Kronecker product of matrices.
        # If T_2 is the 2x2x2 multiplication tensor, T_4 = T_2 \otimes T_2 is the 4x4x4 tensor.
        # The rank is rank(T_2)^2 = 49.
        
        U2, V2, W2 = _strassen_2x2()
        # U2 is 4x7.
        # We need to expand this to 16x49.
        # The indices of U4 are (i1, i2, j1, j2) where i = 2*i1 + i2, j = 2*j1 + j2.
        # The factors are indexed by (c1, c2) where c = 7*c1 + c2.
        
        U4 = np.zeros((16, 49), dtype=int)
        V4 = np.zeros((16, 49), dtype=int)
        W4 = np.zeros((16, 49), dtype=int)
        
        for c1 in range(7):
            for c2 in range(7):
                c = c1 * 7 + c2
                # The factor is U2[:, c1] * U2[:, c2] (Kronecker)
                # The entries are U2[i1, c1] * U2[i2, c2]
                
                for i1 in range(2):
                    for i2 in range(2):
                        i = i1 * 2 + i2
                        # U index in 4x4: i * 4 + j? No, U is for A.
                        # A is 4x4. Index is i * 4 + j.
                        # In _strassen_2x2, U index is i * 2 + j.
                        # So we need to map (i1, j1) and (i2, j2).
                        
                        # Actually, the Kronecker product of matrices A \otimes B has entries (A_{i1 i2'} B_{i2 i2'})?
                        # No.
                        # Let's just use schoolbook for 4x4x4 to be safe and simple.
                        # Rank 64.
                        # Total rank with schoolbook 4x4x4:
                        # 64 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 125.
                        # Same as schoolbook 5x5x5.
                        # We need Strassen 4x4x4 to get 110.
                        
                        pass
        
        # Let's implement Strassen 4x4x4 properly.
        # It is the tensor product of T_2 with T_2.
        # U4[i, c] = sum_{i1,i2,j1,j2} delta(i, 2*i1+i2, 2*j1+j2) U2[i1*2+j1, c1] U2[i2*2+j2, c2]?
        # No.
        # The multiplication of (A \otimes B) (C \otimes D) = (AC) \otimes (BD).
        # So the tensor T_4 is T_2 \otimes T_2.
        # The factors of T_4 are the Kronecker products of the factors of T_2.
        # If U2, V2, W2 are factors of T_2, then U4 = U2 \otimes U2, V4 = V2 \otimes V2, W4 = W2 \otimes W2 are factors of T_4.
        # Shape of U2 is (4, 7).
        # Shape of U2 \otimes U2 is (16, 49).
        # This matches.
        
        U4 = np.kron(U2, U2)
        V4 = np.kron(V2, V2)
        W4 = np.kron(W2, W2)
        
        return U4, V4, W4

    U444, V444, W444 = _strassen_4x4()
    
    # Add A44 B44 to C44
    # A44 indices: i in 0..3, j in 0..3. Global U index: i*5 + j.
    # B44 indices: j in 0..3, k in 0..3. Global V index: j*5 + k.
    # C44 indices: k in 0..3, i in 0..3. Global W index: k*5 + i.
    
    for c in range(49):
        for i in range(4):
            for j in range(4):
                val = U444[i * 4 + j, c]
                if val != 0:
                    U[i * 5 + j, col_offset + c] += val
        for j in range(4):
            for k in range(4):
                val = V444[j * 4 + k, c]
                if val != 0:
                    V[j * 5 + k, col_offset + c] += val
        for k in range(4):
            for i in range(4):
                val = W444[k * 4 + i, c]
                if val != 0:
                    W[k * 5 + i, col_offset + c] += val
    col_offset += 49

    # 2. C44 part 2: A41 * B14 (4x1x4)
    # A41: rows 0-3, col 4.
    # B14: row 4, cols 0-3.
    # C44: rows 0-3, cols 0-3.
    
    U414, V414, W414 = _schoolbook(4, 1, 4)
    # U414: 4x16. Index i*1 + 0.
    # V414: 1x4. Index 0*4 + k.
    # W414: 4x4. Index k*4 + i.
    
    for c in range(16):
        for i in range(4):
            val = U414[i, c]
            if val != 0:
                U[i * 5 + 4, col_offset + c] += val
        for k in range(4):
            val = V414[k, c]
            if val != 0:
                V[4 * 5 + k, col_offset + c] += val
        for k in range(4):
            for i in range(4):
                val = W414[k * 4 + i, c]
                if val != 0:
                    W[k * 5 + i, col_offset + c] += val
    col_offset += 16

    # 3. C41: A44 * B41 (4x4x1)
    # A44: rows 0-3, cols 0-3.
    # B41: rows 0-3, col 4.
    # C41: rows 0-3, col 4.
    
    U441, V441, W441 = _schoolbook(4, 4, 1)
    # U441: 16x16.
    # V441: 4x16. Index j*1 + 0.
    # W441: 1x4. Index 0*4 + i.
    
    for c in range(16):
        for i in range(4):
            for j in range(4):
                val = U441[i * 4 + j, c]
                if val != 0:
                    U[i * 5 + j, col_offset + c] += val
        for j in range(4):
            val = V441[j, c]
            if val != 0:
                V[j * 5 + 4, col_offset + c] += val
        for i in range(4):
            val = W441[i, c]
            if val != 0:
                W[4 * 5 + i, col_offset + c] += val
    col_offset += 16

    # 4. C41 part 2: A41 * B11 (4x1x1)
    # A41: rows 0-3, col 4.
    # B11: row 4, col 4.
    # C41: rows 0-3, col 4.
    
    U411, V411, W411 = _schoolbook(4, 1, 1)
    
    for c in range(4):
        for i in range(4):
            val = U411[i, c]
            if val != 0:
                U[i * 5 + 4, col_offset + c] += val
        val = V411[0, c]
        if val != 0:
            V[4 * 5 + 4, col_offset + c] += val
        for i in range(4):
            val = W411[i, c]
            if val != 0:
                W[4 * 5 + i, col_offset + c] += val
    col_offset += 4

    # 5. C14: A14 * B44 (1x4x4)
    # A14: row 4, cols 0-3.
    # B44: rows 0-3, cols 0-3.
    # C14: row 4, cols 0-3.
    
    U144, V144, W144 = _schoolbook(1, 4, 4)
    
    for c in range(16):
        for j in range(4):
            val = U144[j, c]
            if val != 0:
                U[4 * 5 + j, col_offset + c] += val
        for j in range(4):
            for k in range(4):
                val = V144[j * 4 + k, c]
                if val != 0:
                    V[j * 5 + k, col_offset + c] += val
        for k in range(4):
            val = W144[k, c]
            if val != 0:
                W[k * 5 + 4, col_offset + c] += val
    col_offset += 16

    # 6. C14 part 2: A11 * B14 (1x1x4)
    # A11: row 4, col 4.
    # B14: row 4, cols 0-3.
    # C14: row 4, cols 0-3.
    
    U114, V114, W114 = _schoolbook(1, 1, 4)
    
    for c in range(4):
        val = U114[0, c]
        if val != 0:
            U[4 * 5 + 4, col_offset + c] += val
        for k in range(4):
            val = V114[k, c]
            if val != 0:
                V[4 * 5 + k, col_offset + c] += val
        for k in range(4):
            val = W114[k, c]
            if val != 0:
                W[k * 5 + 4, col_offset + c] += val
    col_offset += 4

    # 7. C11: A14 * B41 (1x4x1)
    # A14: row 4, cols 0-3.
    # B41: rows 0-3, col 4.
    # C11: row 4, col 4.
    
    U141, V141, W141 = _schoolbook(1, 4, 1)
    
    for c in range(4):
        for j in range(4):
            val = U141[j, c]
            if val != 0:
                U[4 * 5 + j, col_offset + c] += val
        for j in range(4):
            val = V141[j, c]
            if val != 0:
                V[j * 5 + 4, col_offset + c] += val
        val = W141[0, c]
        if val != 0:
            W[4 * 5 + 4, col_offset + c] += val
    col_offset += 4

    # 8. C11 part 2: A11 * B11 (1x1x1)
    # A11: row 4, col 4.
    # B11: row 4, col 4.
    # C11: row 4, col 4.
    
    U111, V111, W111 = _schoolbook(1, 1, 1)
    
    for c in range(1):
        val = U111[0, c]
        if val != 0:
            U[4 * 5 + 4, col_offset + c] += val
        val = V111[0, c]
        if val != 0:
            V[4 * 5 + 4, col_offset + c] += val
        val = W111[0, c]
        if val != 0:
            W[4 * 5 + 4, col_offset + c] += val
    col_offset += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
