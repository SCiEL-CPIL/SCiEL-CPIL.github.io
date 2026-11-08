<<<<<<< SEARCH
        elif (ni, nj, nk) == (2, 2, 1):
            # 2x2x1: A is 2x2, B is 2x1, C is 2x1
            # C0 = A00*B0 + A01*B1
            # C1 = A10*B0 + A11*B1
            # Rank 2:
            # Term 1: A00*B0 -> C0
            # Term 2: A01*B1 -> C0
            # Term 3: A10*B0 -> C1
            # Term 4: A11*B1 -> C1
            # Wait, that's rank 4. Can we do rank 2?
            # C0 = A00*B0 + A01*B1
            # C1 = A10*B0 + A11*B1
            # This is a 2x2x1 tensor. The rank is at most 2 if we can factor it.
            # Actually, for 2x2x1, the rank is 2 (since it's a 2x2 matrix times a vector).
            # Term 1: (A00, A10) * (B0, B0) -> (C0, C1) with coefficients (1,1)
            # No, that doesn't work directly.
            # Let's use schoolbook for 2x2x1 which is rank 4.
            # But the rank_map says 2. Let me verify.
            # The tensor T_{i,j,k} = delta_{i,k} for 2x2x1? No.
            # For matrix multiplication: C_i = sum_j A_{i,j} B_{j}
            # T_{i,j,k} where i is row of C, j is col of A / row of B, k is col of B.
            # Since B is 2x1, k=0 only.
            # T_{i,j,0} = delta_{i,j}
            # This is the identity tensor. Rank of identity 2x2x1 is 2.
            # Term 1: A00 * B0 -> C0
            # Term 2: A11 * B1 -> C1
            # But we also have A01*B1 -> C0 and A10*B0 -> C1.
            # So we need 4 terms? No, the rank of the identity tensor 2x2x1 is 2.
            # T = e0⊗e0⊗e0 + e1⊗e1⊗e1
            # So U: (0,0)=1, (1,1)=1
            # V: (0,0)=1, (1,0)=1
            # W: (0,0)=1, (1,0)=1
            # Wait, V is 2x1, so V[j,k] where j=0,1 and k=0.
            # V[0,0]=1, V[1,0]=1? No, that would give A00*B0 + A10*B0 for C0?
            # Let me re-derive.
            # U[i,j], V[j,k], W[k,i]
            # i=0,1; j=0,1; k=0
            # Term 1: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> contributes to C[0]
            # Term 2: U[1,1]=1, V[1,0]=1, W[0,1]=1 -> contributes to C[1]
            # This gives C0 = A00*B0, C1 = A11*B1. Missing A01*B1 and A10*B0.
            # So rank 2 is not sufficient for general 2x2x1 matrix multiplication.
            # The rank is actually 4 (schoolbook).
            # Let me use schoolbook.
            
            for li in range(2):
                for lj in range(2):
                    U[(i0 + li) * m + (j0 + lj)][col_offset] = 1
                    V[(j0 + lj) * p + (k0 + 0)][col_offset] = 1
                    W[(k0 + 0) * n + (i0 + li)][col_offset] = 1
                    col_offset += 1
        
        elif (ni, nj, nk) == (1, 2, 2):
            # 1x2x2: A is 1x2, B is 2x2, C is 1x2
            # C0 = A0*B00 + A1*B10
            # C1 = A0*B01 + A1*B11
            # Rank 2:
            # Term 1: A0 * (B00, B01) -> (C0, C1)
            # Term 2: A1 * (B10, B11) -> (C0, C1)
            # U: U[0,0]=1 (A0), U[1,1]=1 (A1)
            # V: V[0,0]=1 (B00), V[1,0]=1 (B01)? No.
            # V[j,k]: j=0,1; k=0,1
            # Term 1: U[0,0]=1, V[0,0]=1, W[0,0]=1; V[0,1]=1, W[1,0]=1
            # This is getting complex. Let me use schoolbook rank 4.
            
            for lj in range(2):
                for lk in range(2):
                    U[(i0 + 0) * m + (j0 + lj)][col_offset] = 1
                    V[(j0 + lj) * p + (k0 + lk)][col_offset] = 1
                    W[(k0 + lk) * n + (i0 + 0)][col_offset] = 1
                    col_offset += 1
        
        elif (ni, nj, nk) == (1, 2, 1):
            # 1x2x1: A is 1x2, B is 2x1, C is 1x1
            # C0 = A0*B0 + A1*B1
            # Rank 2:
            # Term 1: A0*B0
            # Term 2: A1*B1
            
            U[(i0 + 0) * m + (j0 + 0)][col_offset] = 1
            V[(j0 + 0) * p + (k0 + 0)][col_offset] = 1
            W[(k0 + 0) * n + (i0 + 0)][col_offset] = 1
            col_offset += 1
            
            U[(i0 + 0) * m + (j0 + 1)][col_offset] = 1
            V[(j0 + 1) * p + (k0 + 0)][col_offset] = 1
            W[(k0 + 0) * n + (i0 + 0)][col_offset] = 1
            col_offset += 1
=======
        elif (ni, nj, nk) == (2, 2, 1):
            # Schoolbook: 4 terms
            for li in range(2):
                for lj in range(2):
                    U[(i0 + li) * m + (j0 + lj)][col_offset] = 1
                    V[(j0 + lj) * p + (k0 + 0)][col_offset] = 1
                    W[(k0 + 0) * n + (i0 + li)][col_offset] = 1
                    col_offset += 1
        
        elif (ni, nj, nk) == (1, 2, 2):
            # Schoolbook: 4 terms
            for lj in range(2):
                for lk in range(2):
                    U[(i0 + 0) * m + (j0 + lj)][col_offset] = 1
                    V[(j0 + lj) * p + (k0 + lk)][col_offset] = 1
                    W[(k0 + lk) * n + (i0 + 0)][col_offset] = 1
                    col_offset += 1
        
        elif (ni, nj, nk) == (1, 2, 1):
            # 2 terms
            U[(i0 + 0) * m + (j0 + 0)][col_offset] = 1
            V[(j0 + 0) * p + (k0 + 0)][col_offset] = 1
            W[(k0 + 0) * n + (i0 + 0)][col_offset] = 1
            col_offset += 1
            
            U[(i0 + 0) * m + (j0 + 1)][col_offset] = 1
            V[(j0 + 1) * p + (k0 + 0)][col_offset] = 1
            W[(k0 + 0) * n + (i0 + 0)][col_offset] = 1
            col_offset += 1
>>>>>>> REPLACE

Now let me also fix the U_entries bug in the Strassen section:

