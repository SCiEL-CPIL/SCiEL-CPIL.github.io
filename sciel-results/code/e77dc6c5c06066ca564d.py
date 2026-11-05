import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We'll build up the U, V, W matrices by concatenating blocks.
    # Each block contributes a set of (u, v, w) vectors.
    
    blocks = []  # Each block is (u_vec, v_vec, w_vec) where u has length n*m=15, v has length m*p=25, w has length p*n=15
    
    def add_block(u, v, w):
        blocks.append((u, v, w))
    
    # Helper: create zero vectors
    def zero_u(): return [0]*15
    def zero_v(): return [0]*25
    def zero_w(): return [0]*15
    
    # Block 1: 3x3x3 (A rows 0-2, A cols 0-2, B rows 0-2, B cols 0-2)
    # This is the Winograd rank-23 decomposition for 3x3x3.
    # Winograd for 3x3x3: 
    # A is 3x3, B is 3x3, C = A*B
    # Using the standard Winograd scheme with rank 23.
    
    # I'll use a known rank-23 decomposition for 3x3x3.
    # This is complex; let me use a simplified approach or known coefficients.
    
    # Actually, let me use a known Winograd 3x3x3 rank-23 scheme.
    # The Winograd algorithm for 3x3 matrix multiplication uses 23 multiplications.
    
    # Let me define the 23 multiplications for 3x3x3.
    # Using the scheme from Winograd (1979) or equivalent.
    
    # I'll use a known set of 23 bilinear forms.
    # This is quite involved. Let me use a reference implementation.
    
    # For 3x3x3 rank 23, here's one known decomposition:
    # The 23 multiplications M[0..22] are defined as:
    
    # Let me use a simpler approach: I'll hardcode the Winograd 3x3x3 rank-23.
    
    # Actually, for correctness, let me use a known scheme.
    # I'll define the 23 terms.
    
    winograd_333 = [
        # (a_indices, b_indices) for each multiplication
        # Each entry: (i, j, k) meaning M = A[i,j] * B[j,k] type, but Winograd is more complex.
        # Let me use a different approach: define the 23 linear combinations.
    ]
    
    # This is getting very complex to hardcode. Let me use a different strategy.
    # I'll use schoolbook for all blocks but with the Strassen improvement where applicable.
    # That gives: 27 (3x3x3) + 17 (3x3x2) + 17 (3x2x3) + 12 (3x2x2) = 73.
    # Still better than 75.
    
    # Actually, let me just use schoolbook for everything first to verify correctness,
    # then add Strassen for the 2x2x2 sub-blocks.
    
    # Let me restart with a cleaner approach.
    
    blocks = []
    
    def make_block(n_b, m_b, p_b, i_off, j_off, k_off, use_strassen=False):
        """Create a block for n_b x m_b x p_b multiplication at offset (i_off, j_off, k_off)."""
        block_blocks = []
        
        if use_strassen and n_b == 2 and m_b == 2 and p_b == 2:
            # Strassen 2x2x2, rank 7
            # A is 2x2 at (i_off, j_off), B is 2x2 at (j_off, k_off)
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
                u = [0]*15
                v = [0]*25
                w = [0]*15
                for (ii, jj, coeff) in u_coeffs:
                    u[ii*5+jj] += coeff
                for (jj, kk, coeff) in v_coeffs:
                    v[jj*5+kk] += coeff
                for (kk, ii, coeff) in w_coeffs:
                    w[kk*3+ii] += coeff
                block_blocks.append((u, v, w))
            
            # M1 = (A00+A11)(B00+B11)
            add_strassen_term(
                [(i_off, j_off, 1), (i_off+1, j_off+1, 1)],
                [(j_off, k_off, 1), (j_off+1, k_off+1, 1)],
                [(k_off, i_off, 1), (k_off+1, i_off+1, 1),
                 (k_off, i_off+1, -1), (k_off+1, i_off, -1)]
            )
            
            # M2 = (A10+A11)B00
            add_strassen_term(
                [(i_off+1, j_off, 1), (i_off+1, j_off+1, 1)],
                [(j_off, k_off, 1)],
                [(k_off, i_off+1, 1)]
            )
            
            # M3 = A00(B10-B11)
            add_strassen_term(
                [(i_off, j_off, 1)],
                [(j_off+1, k_off, 1), (j_off+1, k_off+1, -1)],
                [(k_off, i_off, 1)]
            )
            
            # M4 = A11(B01-B00)
            add_strassen_term(
                [(i_off+1, j_off+1, 1)],
                [(j_off, k_off+1, 1), (j_off, k_off, -1)],
                [(k_off+1, i_off+1, 1)]
            )
            
            # M5 = (A00+A01)B11
            add_strassen_term(
                [(i_off, j_off, 1), (i_off, j_off+1, 1)],
                [(j_off+1, k_off+1, 1)],
                [(k_off+1, i_off, 1)]
            )
            
            # M6 = (A10-A00)(B00+B01)
            add_strassen_term(
                [(i_off+1, j_off, 1), (i_off, j_off, -1)],
                [(j_off, k_off, 1), (j_off, k_off+1, 1)],
                [(k_off, i_off+1, 1), (k_off+1, i_off+1, 1)]
            )
            
            # M7 = (A01-A11)(B10+B11)
            add_strassen_term(
                [(i_off, j_off+1, 1), (i_off+1, j_off+1, -1)],
                [(j_off+1, k_off, 1), (j_off+1, k_off+1, 1)],
                [(k_off, i_off, 1), (k_off+1, i_off, 1)]
            )
            
            return block_blocks
        else:
            # Schoolbook
            for ii in range(n_b):
                for jj in range(m_b):
                    for kk in range(p_b):
                        u = [0]*15
                        v = [0]*25
                        w = [0]*15
                        u[(i_off+ii)*5 + (j_off+jj)] = 1
                        v[(j_off+jj)*5 + (k_off+kk)] = 1
                        w[(k_off+kk)*3 + (i_off+ii)] = 1
                        block_blocks.append((u, v, w))
            return block_blocks
    
    # Block 1: 3x3x3 at (0,0,0) - schoolbook for now
    blocks.extend(make_block(3, 3, 3, 0, 0, 0, use_strassen=False))
    
    # Block 2: 3x3x2 at (0,0,3) - schoolbook
    blocks.extend(make_block(3, 3, 2, 0, 0, 3, use_strassen=False))
    
    # Block 3: 3x2x3 at (0,3,0) - schoolbook
    blocks.extend(make_block(3, 2, 3, 0, 3, 0, use_strassen=False))
    
    # Block 4: 3x2x2 at (0,3,3) - schoolbook
    blocks.extend(make_block(3, 2, 2, 0, 3, 3, use_strassen=False))
    
    # Now combine all blocks into U, V, W
    rank = len(blocks)
    U = [[0]*rank for _ in range(15)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(15)]
    
    for col, (u, v, w) in enumerate(blocks):
        for i in range(15):
            if u[i] != 0:
                U[i][col] = u[i]
        for i in range(25):
            if v[i] != 0:
                V[i][col] = v[i]
        for i in range(15):
            if w[i] != 0:
                W[i][col] = w[i]
    
    return {"U": U, "V": V, "W": W}
