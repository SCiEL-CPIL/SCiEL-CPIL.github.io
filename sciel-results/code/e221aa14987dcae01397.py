"""Optimized tensor decomposition for 3x4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-66 decomposition of the 3x4x6 matrix multiplication tensor.
    Uses Strassen for 2x2x2 blocks and direct for 1x2x2 blocks.
    
    Block structure:
    - Split 4 columns into 2 blocks of 2
    - Split 6 rows into 3 blocks of 2
    - 6 blocks total, each 3x2x2:
      * 2x2x2 part (rows 1-2 of A): Strassen, rank 7
      * 1x2x2 part (row 3 of A): direct, rank 4
      * Total per block: 11
    - Grand total: 6 * 11 = 66
    """
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 66
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 66
    W = [[0.0] * rank for _ in range(p * n)]  # 18 x 66
    
    # Strassen formulas for 2x2x2 (indices 0-based):
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Direct formulas for 1x2x2:
    # A = [a0, a1] (1x2)
    # B = [[b00, b01], [b10, b11]]
    # N1 = a0*b00
    # N2 = a1*b10
    # N3 = a0*b01
    # N4 = a1*b11
    # C0 = N1 + N2
    # C1 = N3 + N4
    
    # For block (col_block cb in {0,1}, row_block rb in {0,1,2}):
    # A rows: 0,1,2; A cols: 2*cb, 2*cb+1
    # B rows: 2*rb, 2*rb+1; B cols: 0,1,2
    # Output C rows: 0,1,2; C cols: 2*rb, 2*rb+1
    
    col_blocks = 2
    row_blocks = 3
    
    for cb in range(col_blocks):
        for rb in range(row_blocks):
            # Start column for this block's contributions
            base = (cb * row_blocks + rb) * 11
            
            # --- Strassen part (rows 0,1 of A, both cols) ---
            # A indices: row 0,1; cols 2*cb, 2*cb+1
            # B indices: rows 2*rb, 2*rb+1; cols 0,1,2 (all 3 output cols)
            
            # M1 = (A[0,2cb] + A[1,2cb+1]) * (B[2rb,0] + B[2rb+1,1])
            # Wait, need to be careful with B structure.
            # B is (m x p) = (4 x 6). B[j, k] where j is row (0..3), k is col (0..5)
            # For block rb, we use B rows 2*rb, 2*rb+1 and B cols 0,1,2 (output cols)
            
            # Actually, let me reconsider the tensor structure.
            # The multiplication tensor T[i,j,k] = 1 means C[i,k] += A[i,j]*B[j,k]
            # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
            # So for each term: A[i,j] * B[j,k] contributes to C[i,k]
            
            # For Strassen on A[0:2, 2cb:2cb+2] * B[2rb:2rb+2, :]:
            # But B has 6 columns (p=6), and we want output C[0:2, 2rb:2rb+2]
            # Wait, that's not right either.
            
            # Let me re-examine: C = A @ B where A is 3x4, B is 4x6, C is 3x6
            # C[i,k] = sum_j A[i,j] * B[j,k]
            
            # Block (cb, rb): 
            # A subblock: rows 0..2, cols 2cb..2cb+1 (3x2)
            # B subblock: rows 2rb..2rb+1, cols 0..5 (2x6) -- but we only need cols 2rb..2rb+1?
            # No! B is 4x6. The rows of B are indexed by the shared dimension (m=4).
            # B[j,k] where j=0..3 (matches A's columns), k=0..5 (output columns)
            
            # So for block cb (A's column block) and rb (B's row block = A's column block? No!)
            
            # Actually, the shared dimension is j (0..3 for A's cols and B's rows).
            # A[i,j] * B[j,k] -> C[i,k]
            
            # If I split A's columns (j) into blocks of 2: cb=0 uses j=0,1; cb=1 uses j=2,3
            # If I split B's rows (j) into blocks of 2: rb=0 uses j=0,1; rb=1 uses j=2,3
            # But cb and rb must match! A's column block and B's row block are the same j indices.
            
            # So it's not 6 independent blocks. It's:
            # For each j-block (2 blocks of size 2):
            #   A[:, j_block] (3x2) * B[j_block, :] (2x6) -> C (3x6)
            # Each such product is 3x2x6.
            
            # Let me redo: 2 blocks, each 3x2x6.
            # Split 3x2x6 into:
            #   - Rows 0,1 of A (2x2x6): can further split into 3 sub-blocks of 2x2x2
            #   - Row 2 of A (1x2x6): can further split into 3 sub-blocks of 1x2x2
            
            # For 2x2x6: split 6 cols into 3 groups of 2 -> 3 blocks of 2x2x2, each rank 7 -> 21
            # For 1x2x6: split 6 cols into 3 groups of 2 -> 3 blocks of 1x2x2, each rank 4 -> 12
            # Total per j-block: 21 + 12 = 33
            # Total: 2 * 33 = 66
            
            pass  # Will implement below
    
    # Let me implement properly now
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for j_block in range(2):  # j=0,1 or j=2,3
        # 2x2x6 part: A[0:2, j_block*2:(j_block+1)*2] * B[j_block*2:(j_block+1)*2, :]
        for k_block in range(3):  # output cols 2*k_block, 2*k_block+1
            # Strassen for 2x2x2
            a00 = 0; a01 = 1  # A row 0, cols j_block*2, j_block*2+1
            a10 = 2; a11 = 3  # A row 1, cols j_block*2, j_block*2+1
            b00 = 0; b01 = 1  # B row j_block*2, cols 2*k_block, 2*k_block+1
            b10 = 0; b11 = 1  # B row j_block*2+1, cols 2*k_block, 2*k_block+1
            c00 = 0; c01 = 1  # C row 0, cols 2*k_block, 2*k_block+1
            c10 = 0; c11 = 1  # C row 1, cols 2*k_block, 2*k_block+1
            
            # M1 = (A[0,2cb]+A[1,2cb+1])*(B[2rb,0]+B[2rb+1,1])
            # In terms of flat indices:
            # U index for A[i,j]: i*m+j
            # V index for B[j,k]: j*p+k
            # W index for C[k,i]: k*n+i
            
            aj0 = j_block * 2      # A col index for first col in block
            aj1 = j_block * 2 + 1  # A col index for second col in block
            bj0 = j_block * 2      # B row index for first row in block
            bj1 = j_block * 2 + 1  # B row index for second row in block
            ck0 = k_block * 2      # C col index for first col in block
            ck1 = k_block * 2 + 1  # C col index for second col in block
            
            # M1 = (A[0,aj0]+A[1,aj1])*(B[bj0,ck0]+B[bj1,ck1])
            U[0*m+aj0][col] += 1
            U[1*m+aj1][col] += 1
            V[bj0*p+ck0][col] += 1
            V[bj1*p+ck1][col] += 1
            W[ck0*n+0][col] += 1   # C[0, ck0]
            W[ck1*n+1][col] += 1   # C[1, ck1]
            # But we need C[0,ck0] += M1, C[0,ck1] += M1, C[1,ck0] += M1, C[1,ck1] += M1?
            # No, Strassen combines them. Let me just add the multiplication terms.
            
            # Actually for tensor decomposition, each rank-1 term is U[i,j,c]*V[j,k,c]*W[k,i,c]
            # So I just need to set U, V, W for each multiplication.
            
            # Let me redo this more carefully. For each rank-1 term:
            # U[a_idx][c] = coefficient for A element
            # V[b_idx][c] = coefficient for B element  
            # W[c_idx][c] = coefficient for C element
            # The product U[a_idx][c]*V[b_idx][c]*W[c_idx][c] contributes to the tensor.
            
            # For Strassen M1 = (A[0,aj0]+A[1,aj1])*(B[bj0,ck0]+B[bj1,ck1])
            # This expands to 4 terms, but in tensor decomposition we can have:
            # One rank-1 term where U has 1 at both A[0,aj0] and A[1,aj1]
            # and V has 1 at both B[bj0,ck0] and B[bj1,ck1]
            # and W has appropriate coefficients for the 4 C elements.
            
            # Actually no. A rank-1 term in the tensor is:
            # sum_{a,b,c} U[a,c]*V[b,c]*W[c_idx,c] * e_a ⊗ e_b ⊗ e_c
            # where a indexes A, b indexes B, c_idx indexes C.
            
            # For M1 contributing to C[0,ck0], C[0,ck1], C[1,ck0], C[1,ck1]:
            # We need 4 separate contributions or use the linearity.
            
            # The correct way: each Strassen "multiplication" M_i is a bilinear form.
            # M1 = (A[0,aj0]+A[1,aj1])*(B[bj0,ck0]+B[bj1,ck1])
            # This is a single rank-1 term in the decomposition if we allow sums in U and V.
            # U[0,aj0] = 1, U[1,aj1] = 1, V[bj0,ck0] = 1, V[bj1,ck1] = 1
            # Then W needs to distribute to the 4 C positions.
            # But W[k*n+i] for k=ck0,i=0 and k=ck1,i=0 and k=ck0,i=1 and k=ck1,i=1
            # can each be set to 1, giving M1 added to all 4 positions.
            # But that's not what Strassen wants.
            
            # I think the issue is that in tensor decomposition, a rank-1 term is:
            # u_a * v_b * w_c for specific a, b, c. You can't have sums in u, v, w
            # for a single rank-1 term.
            
            # So each Strassen multiplication actually requires 4 rank-1 terms
            # (for the 4 products in the expansion), OR we use the fact that
            # the tensor entries are what matter.
            
            # Let me think differently. The multiplication tensor T has entries:
            # T[a,b,c] = 1 if A element a, B element b, C element c are related
            # Specifically, T[i*m+j, j*p+k, k*n+i] = 1
            
            # A decomposition of rank r means T = sum_{t=1}^r U[:,t] ⊗ V[:,t] ⊗ W[:,t]
            # i.e., T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
            
            # For Strassen, we want to express the 2x2x2 multiplication tensor
            # as 7 rank-1 terms. The standard construction gives 7 terms where
            # each term has specific nonzero entries in U, V, W.
            
            # Let me just hardcode the 7 terms for 2x2x2 and 4 terms for 1x2x2.
            
            col -= 1  # undo the increment, will manage col properly
            
    # Let me start fresh with a clean implementation
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    for j_block in range(2):
        aj0 = j_block * 2
        aj1 = j_block * 2 + 1
        bj0 = j_block * 2
        bj1 = j_block * 2 + 1
        
        for k_block in range(3):
            ck0 = k_block * 2
            ck1 = k_block * 2 + 1
            
            # Strassen 2x2x2: A[0:2, aj0:aj1+1] * B[bj0:bj1+1, ck0:ck1+1] -> C[0:2, ck0:ck1+1]
            
            # M1 = (A[0,aj0]+A[1,aj1])*(B[bj0,ck0]+B[bj1,ck1])
            # Contributes to C[0,ck0], C[0,ck1], C[1,ck0], C[1,ck1]
            # But with coefficients: C[0,ck0]+=1, C[0,ck1]+=1, C[1,ck0]+=1, C[1,ck1]+=1
            # Wait, that's not right. Let me use the standard Strassen output formulas.
            
            # Standard Strassen (0-indexed):
            # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1 (with signs in final combo)
            # Actually the final C is linear combination of M1..M7.
            
            # The tensor decomposition gives 7 rank-1 terms. Let me define them:
            
            # Term 1: M1 = (A[0,aj0]+A[1,aj1])*(B[bj0,ck0]+B[bj1,ck1])
            # In the final C: c00 += M1, c01 -= M1? No...
            
            # Let me just use the explicit rank-1 terms for Strassen.
            # From literature, the 7 terms are:
            
            # t1: U: a00=1, a11=1; V: b00=1, b11=1; W: c00=1, c11=1
            # t2: U: a10=1, a11=1; V: b00=1; W: c10=1
            # t3: U: a00=1; V: b01=1, b11=-1; W: c01=1
            # t4: U: a11=1; V: b10=1, b00=-1; W: c10=1
            # t5: U: a00=1, a01=1; V: b11=1; W: c01=1
            # t6: U: a10=1, a00=-1; V: b00=1, b01=1; W: c11=1
            # t7: U: a01=1, a11=-1; V: b10=1, b11=1; W: c00=1
            
            # Then:
            # c00 = t1 + t4 - t5 + t7
            # c01 = t3 + t5
            # c10 = t2 + t4
            # c11 = t1 - t2 + t3 + t6
            
            # This matches the standard formulas!
            
            # Term 1
            U[0*m+aj0][col] = 1
            U[1*m+aj1][col] = 1
            V[bj0*p+ck0][col] = 1
            V[bj1*p+ck1][col] = 1
            W[ck0*n+0][col] = 1
            W[ck1*n+1][col] = 1
            col += 1
            
            # Term 2
            U[1*m+aj0][col] = 1
            U[1*m+aj1][col] = 1
            V[bj0*p+ck0][col] = 1
            W[ck0*n+1][col] = 1
            col += 1
            
            # Term 3
            U[0*m+aj0][col] = 1
            V[bj0*p+ck1][col] = 1
            V[bj1*p+ck1][col] = -1
            W[ck1*n+0][col] = 1
            col += 1
            
            # Term 4
            U[1*m+aj1][col] = 1
            V[bj1*p+ck0][col] = 1
            V[bj0*p+ck0][col] = -1
            W[ck0*n+1][col] = 1
            col += 1
            
            # Term 5
            U[0*m+aj0][col] = 1
            U[0*m+aj1][col] = 1
            V[bj1*p+ck1][col] = 1
            W[ck1*n+0][col] = 1
            col += 1
            
            # Term 6
            U[1*m+aj0][col] = 1
            U[0*m+aj0][col] = -1
            V[bj0*p+ck0][col] = 1
            V[bj0*p+ck1][col] = 1
            W[ck1*n+1][col] = 1
            col += 1
            
            # Term 7
            U[0*m+aj1][col] = 1
            U[1*m+aj1][col] = -1
            V[bj1*p+ck0][col] = 1
            V[bj1*p+ck1][col] = 1
            W[ck0*n+0][col] = 1
            col += 1
            
            # Direct 1x2x2: A[2, aj0:aj1+1] * B[bj0:bj1+1, ck0:ck1+1] -> C[2, ck0:ck1+1]
            # N1 = A[2,aj0]*B[bj0,ck0] -> C[2,ck0]
            U[2*m+aj0][col] = 1
            V[bj0*p+ck0][col] = 1
            W[ck0*n+2][col] = 1
            col += 1
            
            # N2 = A[2,aj1]*B[bj1,ck0] -> C[2,ck0]
            U[2*m+aj1][col] = 1
            V[bj1*p+ck0][col] = 1
            W[ck0*n+2][col] = 1
            col += 1
            
            # N3 = A[2,aj0]*B[bj0,ck1] -> C[2,ck1]
            U[2*m+aj0][col] = 1
            V[bj0*p+ck1][col] = 1
            W[ck1*n+2][col] = 1
            col += 1
            
            # N4 = A[2,aj1]*B[bj1,ck1] -> C[2,ck1]
            U[2*m+aj1][col] = 1
            V[bj1*p+ck1][col] = 1
            W[ck1*n+2][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

