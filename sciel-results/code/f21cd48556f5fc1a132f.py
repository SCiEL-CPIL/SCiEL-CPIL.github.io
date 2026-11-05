"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # We use a block decomposition approach to reduce the rank below the trivial n*m*p = 105.
    # The idea is to decompose the matrix multiplication into smaller blocks and use
    # Strassen's algorithm (rank 7 instead of 8) for 2x2x2 blocks where possible.
    
    # We'll build the decomposition by iterating over blocks.
    # For each block of size (ni, mi, pi), we add the corresponding tensor decomposition.
    
    # Helper: add a block decomposition for M_{ni, mi, pi} using either trivial or Strassen
    def add_block(i_offset, j_offset, k_offset, ni, mi, pi, U, V, W, start_col):
        """
        Add the tensor decomposition for multiplying an ni x mi matrix by an mi x pi matrix.
        The output is stored at positions [k_offset + k][i_offset + i] in W (trace convention: W[k*n+i]).
        U rows are indexed by (i, j) -> i_offset*mi + i, j_offset*mi + j? No, U is (n*m, rank).
        Actually U[i*m + j], V[j*p + k], W[k*n + i].
        
        So for a block starting at (i_offset, j_offset, k_offset) with sizes (ni, mi, pi):
        - U rows: for i in [i_offset, i_offset+ni), j in [j_offset, j_offset+mi): row = i*m + j
        - V rows: for j in [j_offset, j_offset+mi), k in [k_offset, k_offset+pi): row = j*p + k
        - W rows: for k in [k_offset, k_offset+pi), i in [i_offset, i_offset+ni): row = k*n + i
        """
        if ni == 2 and mi == 2 and pi == 2:
            # Use Strassen's algorithm: rank 7
            # Strassen's 7 multiplications for 2x2x2
            # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
            # C = A * B
            # m1 = (a11 + a22)(b11 + b22)
            # m2 = (a21 + a22)b11
            # m3 = a11(b12 - b22)
            # m4 = a22(b21 - b11)
            # m5 = (a11 + a12)b22
            # m6 = (a21 - a11)(b11 + b12)
            # m7 = (a12 - a22)(b21 + b22)
            # c11 = m1 + m4 - m5 + m7
            # c12 = m3 + m5
            # c21 = m2 + m4
            # c22 = m1 - m2 + m3 + m6
            
            col = start_col
            # m1 = (a11 + a22)(b11 + b22)
            # U: a11 -> row (i_offset, j_offset), a22 -> row (i_offset+1, j_offset+1)
            U[i_offset * m + j_offset][col] = 1
            U[(i_offset + 1) * m + (j_offset + 1)][col] = 1
            # V: b11 -> row (j_offset, k_offset), b22 -> row (j_offset+1, k_offset+1)
            V[j_offset * p + k_offset][col] = 1
            V[(j_offset + 1) * p + (k_offset + 1)][col] = 1
            # W: contributes to c11 and c22
            W[k_offset * n + i_offset][col] = 1       # c11
            W[(k_offset + 1) * n + (i_offset + 1)][col] = 1  # c22
            col += 1
            
            # m2 = (a21 + a22)b11
            U[i_offset * m + j_offset][col] = 1       # a21
            U[(i_offset + 1) * m + (j_offset + 1)][col] = 1  # a22
            V[j_offset * p + k_offset][col] = 1       # b11
            # W: contributes to c21 and c22 (with -1)
            W[k_offset * n + (i_offset + 1)][col] = 1  # c21
            W[(k_offset + 1) * n + (i_offset + 1)][col] = -1  # c22
            col += 1
            
            # m3 = a11(b12 - b22)
            U[i_offset * m + j_offset][col] = 1       # a11
            V[j_offset * p + (k_offset + 1)][col] = 1  # b12
            V[(j_offset + 1) * p + (k_offset + 1)][col] = -1  # b22
            # W: contributes to c12 and c22
            W[(k_offset + 1) * n + i_offset][col] = 1  # c12
            W[(k_offset + 1) * n + (i_offset + 1)][col] = 1  # c22
            col += 1
            
            # m4 = a22(b21 - b11)
            U[(i_offset + 1) * m + (j_offset + 1)][col] = 1  # a22
            V[(j_offset + 1) * p + k_offset][col] = 1       # b21
            V[j_offset * p + k_offset][col] = -1       # b11
            # W: contributes to c11 and c21
            W[k_offset * n + i_offset][col] = 1       # c11
            W[k_offset * n + (i_offset + 1)][col] = 1  # c21
            col += 1
            
            # m5 = (a11 + a12)b22
            U[i_offset * m + j_offset][col] = 1       # a11
            U[i_offset * m + (j_offset + 1)][col] = 1  # a12
            V[(j_offset + 1) * p + (k_offset + 1)][col] = 1  # b22
            # W: contributes to c11 (with -1) and c12
            W[k_offset * n + i_offset][col] = -1       # c11
            W[(k_offset + 1) * n + i_offset][col] = 1  # c12
            col += 1
            
            # m6 = (a21 - a11)(b11 + b12)
            U[i_offset * m + j_offset][col] = -1       # a21 (negative)
            U[i_offset * m + j_offset][col] += 1       # Wait, a21 is at (i_offset+1, j_offset)
            # Let me redo this carefully.
            # a21 is at row (i_offset+1)*m + j_offset
            # a11 is at row i_offset*m + j_offset
            U[(i_offset + 1) * m + j_offset][col] = 1  # a21
            U[i_offset * m + j_offset][col] = -1       # a11
            V[j_offset * p + k_offset][col] = 1       # b11
            V[j_offset * p + (k_offset + 1)][col] = 1  # b12
            # W: contributes to c22
            W[(k_offset + 1) * n + (i_offset + 1)][col] = 1  # c22
            col += 1
            
            # m7 = (a12 - a22)(b21 + b22)
            U[i_offset * m + (j_offset + 1)][col] = 1  # a12
            U[(i_offset + 1) * m + (j_offset + 1)][col] = -1  # a22
            V[(j_offset + 1) * p + k_offset][col] = 1       # b21
            V[(j_offset + 1) * p + (k_offset + 1)][col] = 1  # b22
            # W: contributes to c11
            W[k_offset * n + i_offset][col] = 1       # c11
            col += 1
            
            return col - start_col  # 7
        
        else:
            # Trivial decomposition: ni * mi * pi multiplications
            col = start_col
            for ii in range(ni):
                for jj in range(mi):
                    for kk in range(pi):
                        U[(i_offset + ii) * m + (j_offset + jj)][col] = 1
                        V[(j_offset + jj) * p + (k_offset + kk)][col] = 1
                        W[(k_offset + kk) * n + (i_offset + ii)][col] = 1
                        col += 1
            return col - start_col
    
    # Now decompose 3x5x7 into blocks.
    # Split n=3 into 2+1, m=5 into 2+2+1, p=7 into 2+2+2+1
    # This gives 2*3*4 = 24 blocks. Let's be more strategic.
    
    # Let's use a recursive/block approach:
    # n=3: split into 2+1
    # m=5: split into 2+2+1
    # p=7: split into 2+2+2+1
    
    # Block sizes:
    # For i in {0, 1} (sizes 2, 1), j in {0, 1, 2} (sizes 2, 2, 1), k in {0, 1, 2, 3} (sizes 2, 2, 2, 1)
    
    i_offsets = [0, 2]
    i_sizes = [2, 1]
    j_offsets = [0, 2, 4]
    j_sizes = [2, 2, 1]
    k_offsets = [0, 2, 4, 6]
    k_sizes = [2, 2, 2, 1]
    
    # Initialize U, V, W with zeros. We don't know the total rank yet, so let's first compute it.
    total_rank = 0
    for ii in range(len(i_offsets)):
        for jj in range(len(j_offsets)):
            for kk in range(len(k_offsets)):
                ni = i_sizes[ii]
                mi = j_sizes[jj]
                pi = k_sizes[kk]
                if ni == 2 and mi == 2 and pi == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mi * pi
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ii in range(len(i_offsets)):
        for jj in range(len(j_offsets)):
            for kk in range(len(k_offsets)):
                ni = i_sizes[ii]
                mi = j_sizes[jj]
                pi = k_sizes[kk]
                col += add_block(i_offsets[ii], j_offsets[jj], k_offsets[kk], ni, mi, pi, U, V, W, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
