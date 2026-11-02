<<<<<<< SEARCH
    def _strassen_embedded():
        """
        Construct a decomposition of rank 83 by replacing the 2x2x2 schoolbook block
        (i in {0,1}, j in {0,1}, k in {0,1}) with Strassen's rank-7 decomposition.
        """
        R = 83
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        # Strassen terms for 2x2x2 block
        # Indices: i, j in {0,1}, k in {0,1}
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00, A11
        U[0*m + 0][0] = 1
        U[1*m + 1][0] = 1
        # V: B00, B11
        V[0*p + 0][0] = 1
        V[1*p + 1][0] = 1
        # W: C00 += M1, C11 += M1
        W[0*n + 0][0] = 1
        W[1*n + 1][0] = 1
        
        # Term 2: M2 = (A10+A11)B00
        # U: A10, A11
        U[1*m + 0][1] = 1
        U[1*m + 1][1] = 1
        # V: B00
        V[0*p + 0][1] = 1
        # W: C10 += M2, C11 += M2
        W[0*n + 1][1] = 1
        W[1*n + 1][1] = 1
        
        # Term 3: M3 = A00(B10-B11)
        # U: A00
        U[0*m + 0][2] = 1
        # V: B10, B11
        V[1*p + 0][2] = 1
        V[1*p + 1][2] = -1
        # W: C01 += M3, C11 -= M3
        W[1*n + 0][2] = 1
        W[1*n + 1][2] = -1
        
        # Term 4: M4 = A11(B01-B00)
        # U: A11
        U[1*m + 1][3] = 1
        # V: B01, B00
        V[0*p + 1][3] = 1
        V[0*p + 0][3] = -1
        # W: C00 -= M4, C01 += M4
        W[0*n + 0][3] = -1
        W[1*n + 0][3] = 1
        
        # Term 5: M5 = (A00+A10)B11
        # U: A00, A10
        U[0*m + 0][4] = 1
        U[1*m + 0][4] = 1
        # V: B11
        V[1*p + 1][4] = 1
        # W: C00 += M5, C10 += M5
        W[0*n + 0][4] = 1
        W[0*n + 1][4] = 1
        
        # Term 6: M6 = (A01-A00)(B00+B10)
        # U: A01, A00
        U[0*m + 1][5] = 1
        U[0*m + 0][5] = -1
        # V: B00, B10
        V[0*p + 0][5] = 1
        V[1*p + 0][5] = 1
        # W: C11 -= M6
        W[1*n + 1][5] = -1
        
        # Term 7: M7 = (A10-A11)(B01+B11)
        # U: A10, A11
        U[1*m + 0][6] = 1
        U[1*m + 1][6] = -1
        # V: B01, B11
        V[0*p + 1][6] = 1
        V[1*p + 1][6] = 1
        # W: C00 += M7
        W[0*n + 0][6] = 1
        
        # Fill remaining with schoolbook terms for the complement
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip the 2x2x2 block handled by Strassen
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Verify col matches R
        if col != R:
            # Fallback if logic error
            return _schoolbook()
            
        return {"U": U, "V": V, "W": W}
=======
    def _strassen_2x2x2(U, V, W, i0, j0, k0, col_start):
        """
        Apply Strassen's rank-7 decomposition to the 2x2x2 sub-block
        starting at global indices (i0, j0, k0).
        Returns the next column index.
        """
        # Strassen terms for 2x2x2 block
        # Local indices: ii, jj, kk in {0,1}
        # Global U index: (i0+ii)*m + (j0+jj)
        # Global V index: (j0+jj)*p + (k0+kk)
        # Global W index: (k0+kk)*n + (i0+ii)
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[(i0+0)*m + (j0+0)][col_start+0] = 1
        U[(i0+1)*m + (j0+1)][col_start+0] = 1
        V[(j0+0)*p + (k0+0)][col_start+0] = 1
        V[(j0+1)*p + (k0+1)][col_start+0] = 1
        W[(k0+0)*n + (i0+0)][col_start+0] = 1
        W[(k0+1)*n + (i0+1)][col_start+0] = 1
        
        # Term 2: M2 = (A10+A11)B00
        U[(i0+1)*m + (j0+0)][col_start+1] = 1
        U[(i0+1)*m + (j0+1)][col_start+1] = 1
        V[(j0+0)*p + (k0+0)][col_start+1] = 1
        W[(k0+0)*n + (i0+1)][col_start+1] = 1
        W[(k0+1)*n + (i0+1)][col_start+1] = 1
        
        # Term 3: M3 = A00(B10-B11)
        U[(i0+0)*m + (j0+0)][col_start+2] = 1
        V[(j0+1)*p + (k0+0)][col_start+2] = 1
        V[(j0+1)*p + (k0+1)][col_start+2] = -1
        W[(k0+1)*n + (i0+0)][col_start+2] = 1
        W[(k0+1)*n + (i0+1)][col_start+2] = -1
        
        # Term 4: M4 = A11(B01-B00)
        U[(i0+1)*m + (j0+1)][col_start+3] = 1
        V[(j0+0)*p + (k0+1)][col_start+3] = 1
        V[(j0+0)*p + (k0+0)][col_start+3] = -1
        W[(k0+0)*n + (i0+0)][col_start+3] = -1
        W[(k0+1)*n + (i0+0)][col_start+3] = 1
        
        # Term 5: M5 = (A00+A10)B11
        U[(i0+0)*m + (j0+0)][col_start+4] = 1
        U[(i0+1)*m + (j0+0)][col_start+4] = 1
        V[(j0+1)*p + (k0+1)][col_start+4] = 1
        W[(k0+0)*n + (i0+0)][col_start+4] = 1
        W[(k0+0)*n + (i0+1)][col_start+4] = 1
        
        # Term 6: M6 = (A01-A00)(B00+B10)
        U[(i0+0)*m + (j0+1)][col_start+5] = 1
        U[(i0+0)*m + (j0+0)][col_start+5] = -1
        V[(j0+0)*p + (k0+0)][col_start+5] = 1
        V[(j0+1)*p + (k0+0)][col_start+5] = 1
        W[(k0+1)*n + (i0+1)][col_start+5] = -1
        
        # Term 7: M7 = (A10-A11)(B01+B11)
        U[(i0+1)*m + (j0+0)][col_start+6] = 1
        U[(i0+1)*m + (j0+1)][col_start+6] = -1
        V[(j0+0)*p + (k0+1)][col_start+6] = 1
        V[(j0+1)*p + (k0+1)][col_start+6] = 1
        W[(k0+0)*n + (i0+0)][col_start+6] = 1
        
        return col_start + 7

    def _block_decomposition(n, m, p, block_sizes):
        """
        Construct a block decomposition using the specified block sizes.
        block_sizes: list of (ni, mj, pk) tuples specifying sub-block dimensions.
        For each sub-block, apply the best known low-rank decomposition.
        """
        R = 0
        # First pass: compute total rank
        for (ni, mj, pk) in block_sizes:
            if ni == 2 and mj == 2 and pk == 2:
                R += 7
            elif ni == 1 and mj == 1 and pk == 1:
                R += 1
            elif ni == 1 and mj == 2 and pk == 1:
                R += 2  # 1x2x1 schoolbook
            elif ni == 2 and mj == 1 and pk == 1:
                R += 2  # 2x1x1 schoolbook
            elif ni == 1 and mj == 1 and pk == 2:
                R += 2  # 1x1x2 schoolbook
            elif ni == 1 and mj == 2 and pk == 2:
                R += 4  # 1x2x2 schoolbook
            elif ni == 2 and mj == 1 and pk == 2:
                R += 4  # 2x1x2 schoolbook
            elif ni == 2 and mj == 2 and pk == 1:
                R += 4  # 2x2x1 schoolbook
            else:
                R += ni * mj * pk  # fallback to schoolbook
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        i0 = 0
        j0 = 0
        k0 = 0
        
        for (ni, mj, pk) in block_sizes:
            if ni == 2 and mj == 2 and pk == 2:
                col = _strassen_2x2x2(U, V, W, i0, j0, k0, col)
            else:
                # Schoolbook for this sub-block
                for ii in range(ni):
                    for jj in range(mj):
                        for kk in range(pk):
                            U[(i0+ii)*m + (j0+jj)][col] = 1
                            V[(j0+jj)*p + (k0+kk)][col] = 1
                            W[(k0+kk)*n + (i0+ii)][col] = 1
                            col += 1
            i0 += ni
            j0 += mj
            k0 += pk
        
        return {"U": U, "V": V, "W": W}

    def _strassen_embedded():
        """
        Construct a decomposition using block partitioning with Strassen on 2x2x2 blocks.
        Partition 3x4x7 into:
        - 2x2x2 block (i:0-1, j:0-1, k:0-1): rank 7
        - 1x2x2 block (i:2, j:0-1, k:0-1): rank 4
        - 2x1x2 block (i:0-1, j:2-3, k:0-1): rank 4
        - 1x1x2 block (i:2, j:2-3, k:0-1): rank 2
        - 2x2x1 block (i:0-1, j:0-1, k:2-6): rank 10 (schoolbook, 2*2*5=20... wait)
        
        Let me reconsider. Better partition:
        - 2x2x2 (i:0-1, j:0-1, k:0-1): rank 7
        - 1x2x2 (i:2, j:0-1, k:0-1): rank 4
        - 2x1x1 (i:0-1, j:2, k:0): rank 2
        - 2x1x1 (i:0-1, j:3, k:0): rank 2
        - 1x1x1 (i:2, j:2, k:0): rank 1
        - 1x1x1 (i:2, j:3, k:0): rank 1
        - 2x2x5 (i:0-1, j:0-1, k:2-6): rank 20 (schoolbook)
        - 1x2x5 (i:2, j:0-1, k:2-6): rank 10
        - 2x1x5 (i:0-1, j:2, k:2-6): rank 10
        - 2x1x5 (i:0-1, j:3, k:2-6): rank 10
        - 1x1x5 (i:2, j:2, k:2-6): rank 5
        - 1x1x5 (i:2, j:3, k:2-6): rank 5
        Total: 7+4+2+2+1+1+20+10+10+10+5+5 = 77
        
        Alternative: use 2x2x2 for k:0-1 only, and handle k:2-6 separately.
        Let's try a different partition to maximize Strassen usage.
        """
        # Partition 3x4x7:
        # Block 1: i=0-1, j=0-1, k=0-1 (2x2x2) -> rank 7
        # Block 2: i=2, j=0-1, k=0-1 (1x2x2) -> rank 4
        # Block 3: i=0-1, j=2-3, k=0-1 (2x2x2) -> rank 7
        # Block 4: i=2, j=2-3, k=0-1 (1x2x2) -> rank 4
        # Block 5: i=0-1, j=0-1, k=2-6 (2x2x5) -> rank 20
        # Block 6: i=2, j=0-1, k=2-6 (1x2x5) -> rank 10
        # Block 7: i=0-1, j=2-3, k=2-6 (2x2x5) -> rank 20
        # Block 8: i=2, j=2-3, k=2-6 (1x2x5) -> rank 10
        # Total: 7+4+7+4+20+10+20+10 = 82
        
        block_sizes = [
            (2, 2, 2),  # i:0-1, j:0-1, k:0-1
            (1, 2, 2),  # i:2, j:0-1, k:0-1
            (2, 2, 2),  # i:0-1, j:2-3, k:0-1
            (1, 2, 2),  # i:2, j:2-3, k:0-1
            (2, 2, 5),  # i:0-1, j:0-1, k:2-6
            (1, 2, 5),  # i:2, j:0-1, k:2-6
            (2, 2, 5),  # i:0-1, j:2-3, k:2-6
            (1, 2, 5),  # i:2, j:2-3, k:2-6
        ]
        
        return _block_decomposition(n, m, p, block_sizes)
>>>>>>> REPLACE

