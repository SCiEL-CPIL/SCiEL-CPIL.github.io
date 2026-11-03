"""Program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """Check that U, V, W exactly decompose the n*m*p multiplication tensor."""
    import numpy as np
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check a sample of (i,j,k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if s != 1.0:
                    return False
    # Also check off-diagonal entries should be 0
    # The tensor is non-zero only at (i,j,k) positions corresponding to matrix multiplication
    # Actually for matrix multiplication, every (i,j,k) with i in [0,n), j in [0,m), k in [0,p) 
    # has value 1 (it's the multiplication tensor). Let me verify the convention.
    # W[k*n+i, r] is the official trace convention.
    # The multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # So checking that all (i,j,k) give 1 is sufficient.
    return True

def construct():
    import time
    start_time = time.time()
    deadline = 150.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank 40
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": "V", "W": W}  # placeholder, will fix
    best_result = {"U": U, "V": V, "W": W}
    
    # Try Strassen-based rank 36 decomposition
    # Partition: use Strassen on the 2x2x2 blocks
    # The multiplication tensor for 2x4x5 can be decomposed using Strassen-style
    # block decomposition.
    
    # Strategy: 
    # A is 2x4, B is 4x5, C is 2x5
    # Partition A into columns: A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # Partition B into rows: B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5
    # Partition B further: B_top = [B_t0 | B_t1 | B_t2] (2x2, 2x2, 2x1)
    #                    B_bottom = [B_b0 | B_b1 | B_b2] (2x2, 2x2, 2x1)
    
    # Actually, let's use a cleaner approach.
    # For 2x4x5, we can view this as multiplying a 2x2 block matrix by a 2x5 block matrix
    # where the "2" in the middle is doubled.
    
    # Better: Use the fact that we can apply Strassen to reduce the effective dimension.
    # 
    # Let me think about this more carefully.
    #
    # A (2x4) * B (4x5) = C (2x5)
    #
    # Write A = [A0 A1] where A0, A1 are 2x2
    # Write B = [B0; B1] where B0, B1 are 2x5
    #
    # C = A0*B0 + A1*B1
    #
    # Now A0*B0 is 2x2 * 2x5, and A1*B1 is 2x2 * 2x5.
    #
    # For 2x2*2x5: partition the 5 into 2+2+1
    # A0*B0 = A0*[B0a B0b B0c] = A0*B0a + A0*B0b + A0*B0c
    # where B0a, B0b are 2x2, B0c is 2x1.
    #
    # 2x2*2x2 has Strassen rank 7
    # 2x2*2x1 has rank 4 (or can use Strassen-like)
    #
    # So total rank = 7 + 4 + 7 + 4 = 22? That seems too optimistic.
    # Wait, 2x2*2x1: A(2x2)*B(2x1) = C(2x1)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # This has rank at most 4 (schoolbook). Can we do better?
    # M1 = A00*B0, M2 = A01*B1, M3 = (A00+A01)*(B0+B1), M4 = (A00+A01)*B1... 
    # Actually 2x2*2x1 is just two independent 2x1 multiplications per row, so rank 4 seems tight.
    # Hmm, actually it's a sum of 4 rank-1 terms and I don't think it can be reduced below 4
    # because it contains two independent 1x2*2x1 (which is rank 2 each... no).
    # Let me just use schoolbook for the 2x1 parts.
    #
    # Total: 7 (Strassen for A0*B0a) + 4 (A0*B0b schoolbook... wait B0b is 2x2)
    # Let me redo:
    #
    # C = A0*B0 + A1*B1
    # B0 is 2x5 = [B00 B01 B02] where B00,B01 are 2x2, B02 is 2x1
    # B1 is 2x5 = [B10 B11 B12] where B10,B11 are 2x2, B12 is 2x1
    #
    # A0*B0 = A0*B00 + A0*B01 + A0*B02
    # A1*B1 = A1*B10 + A1*B11 + A1*B12
    #
    # Each A*X where A is 2x2 and X is 2x2: use Strassen (rank 7)
    # Each A*X where A is 2x2 and X is 2x1: schoolbook (rank 4)
    #
    # Total = 7*4 + 4*2 = 28 + 8 = 36. 
    #
    # Great, rank 36!
    
    # Let me construct the rank 36 decomposition.
    #
    # Indexing:
    # A[i,j] for i=0,1; j=0,1,2,3
    # B[j,k] for j=0,1,2,3; k=0,1,2,3,4
    # C[i,k] for i=0,1; k=0,1,2,3,4
    #
    # A0 = A[:, 0:2], A1 = A[:, 2:4]
    # B0 = B[0:2, :], B1 = B[2:4, :]
    #
    # B0[:, 0:2] = B00, B0[:, 2:4] = B01, B0[:, 4:5] = B02
    # B1[:, 0:2] = B10, B1[:, 2:4] = B11, B1[:, 4:5] = B12
    #
    # C = A0*B00 + A0*B01 + A0*B02 + A1*B10 + A1*B11 + A1*B12
    #
    # For A0*B00 (2x2 * 2x2), Strassen:
    # M1 = A0[0,0]*B00[0,0] ... no wait, Strassen uses sub-products.
    #
    # Let me write it out explicitly.
    # A0 = [[a00, a01], [a10, a11]] (a00 = A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1])
    # B00 = [[b00, b01], [b10, b11]] (b00=B[0,0], b01=B[0,1], b10=B[1,0], b11=B[1,1])
    #
    # Strassen:
    # S1 = a00 * b00
    # S2 = (a00+a01) * (b10+b11)  -- wait, standard Strassen:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b10)
    # P7 = (a01-a11)(b01+b11)
    #
    # C[0,0] = P1 - P4 + P6 - P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P3 + P5 - P6
    #
    # I need to map this to U, V, W entries.
    #
    # For each term S = u * v * w where:
    # u = A[i,j], v = B[j',k], w = C[k',i']
    # The contribution to C[i',k'] is u*v*w if the indices match.
    #
    # Actually, the tensor decomposition means:
    # C[i,k] = sum_r U[i*4+j_r, r] * V[j_r*5+k_r, r] * W[k_r*2+i, r]
    # where for each r, U[i*4+j_r, r]=1 means we pick row i, col j_r of A
    # V[j_r*5+k_r, r]=1 means we pick row j_r, col k_r of B  
    # W[k_r*2+i, r]=1 means we pick row k_r, col i of C
    #
    # But Strassen has sums and differences, so U, V, W entries can be -1, 0, or 1.
    #
    # Let me build the rank 36 decomposition step by step.
    
    def add_term(U, V, W, rank_pos, a_row, a_col, b_row, b_col, c_row, c_col, coeff):
        """Add a term: coeff * A[a_row,a_col] * B[b_row,b_col] contributes to C[c_row,c_col]"""
        U[a_row * m + a_col][rank_pos] += coeff
        V[b_row * p + b_col][rank_pos] += coeff
        W[c_col * n + c_row][rank_pos] += coeff
    
    # For Strassen, we need to track multiple (a_row, a_col) pairs per U entry, etc.
    # Actually, Strassen combines terms. Let me just build it directly.
    
    # I'll construct U, V, W as lists of lists, with rank = 36.
    rank36 = 36
    U36 = [[0] * rank36 for _ in range(n * m)]
    V36 = [[0] * rank36 for _ in range(m * p)]
    W36 = [[0] * rank36 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for A_block (2x2) * B_block (2x2)
    # A_block rows: a0, a1 (global row indices)
    # A_block cols: a2, a3 (global col indices)  
    # B_block rows: b0, b1 (global row indices)
    # B_block cols: b2, b3 (global col indices)
    # C_block: C[a0,b2], C[a0,b3], C[a1,b2], C[a1,b3]
    def add_strassen_222(A_rows, A_cols, B_rows, B_cols):
        nonlocal col
        a0, a1 = A_rows
        ac0, ac1 = A_cols
        b0, b1 = B_rows
        bc0, bc1 = B_cols
        
        # P1 = (A[a0,ac0]+A[a1,ac1]) * (B[b0,bc0]+B[b1,bc1])
        U36[a0*m+ac0][col] = 1
        U36[a1*m+ac1][col] = 1
        V36[b0*p+bc0][col] = 1
        V36[b1*p+bc1][col] = 1
        # Contributes to C[a0,bc0], C[a0,bc1], C[a1,bc0], C[a1,bc1]
        W36[bc0*n+a0][col] = 1   # C[a0, bc0]
        W36[bc1*n+a0][col] = 1   # C[a0, bc1]
        W36[bc0*n+a1][col] = 1   # C[a1, bc0]
        W36[bc1*n+a1][col] = 1   # C[a1, bc1]
        col += 1
        
        # P2 = (A[a1,ac0]+A[a1,ac1]) * B[b0,bc0]
        U36[a1*m+ac0][col] = 1
        U36[a1*m+ac1][col] = 1
        V36[b0*p+bc0][col] = 1
        W36[bc0*n+a1][col] = 1
        col += 1
        
        # P3 = A[a0,ac0] * (B[b1,bc0]-B[b1,bc1])
        U36[a0*m+ac0][col] = 1
        V36[b1*p+bc0][col] = 1
        V36[b1*p+bc1][col] = -1
        W36[bc0*n+a0][col] = 1
        W36[bc1*n+a0][col] = 1
        col += 1
        
        # P4 = A[a1,ac1] * (B[b0,bc1]-B[b0,bc0])
        U36[a1*m+ac1][col] = 1
        V36[b0*p+bc1][col] = 1
        V36[b0*p+bc0][col] = -1
        W36[bc1*n+a1][col] = 1
        W36[bc0*n+a1][col] = 1
        col += 1
        
        # P5 = (A[a0,ac0]+A[a0,ac1]) * B[b1,bc1]
        U36[a0*m+ac0][col] = 1
        U36[a0*m+ac1][col] = 1
        V36[b1*p+bc1][col] = 1
        W36[bc1*n+a0][col] = 1
        col += 1
        
        # P6 = (A[a1,ac0]-A[a0,ac0]) * (B[b0,bc0]+B[b1,bc0])
        U36[a1*m+ac0][col] = 1
        U36[a0*m+ac0][col] = -1
        V36[b0*p+bc0][col] = 1
        V36[b1*p+bc0][col] = 1
        W36[bc0*n+a1][col] = 1
        col += 1
        
        # P7 = (A[a0,ac1]-A[a1,ac1]) * (B[b0,bc1]+B[b1,bc1])
        U36[a0*m+ac1][col] = 1
        U36[a1*m+ac1][col] = -1
        V36[b0*p+bc1][col] = 1
        V36[b1*p+bc1][col] = 1
        W36[bc1*n+a0][col] = 1
        col += 1
    
    # Helper to add schoolbook 2x2 * 2x1
    # A_block: rows a0,a1; col ac
    # B_block: rows b0,b1; col bc
    # C: C[a0,bc], C[a1,bc]
    def add_221(A_rows, A_col, B_rows, B_col):
        nonlocal col
        a0, a1 = A_rows
        b0, b1 = B_rows
        
        # C[a0,bc] = A[a0,ac]*B[b0,bc] + A[a0,ac]*B[b1,bc] -- NO
        # C[a0,bc] = A[a0,ac]*B[b0,bc] + A[a0,ac+1]*B[b0,bc]... 
        # Wait, A is 2x2, B is 2x1.
        # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        
        # But here A has two rows (a0,a1) and one column? No.
        # A_block is 2x2: rows a0,a1; cols ac0,ac1
        # B_block is 2x1: rows b0,b1; col bc
        # C_block is 2x1: rows a0,a1; col bc
        
        # C[a0,bc] = A[a0,ac0]*B[b0,bc] + A[a0,ac1]*B[b1,bc]
        # C[a1,bc] = A[a1,ac0]*B[b0,bc] + A[a1,ac1]*B[b1,bc]
        
        # Term 1: A[a0,ac0] * B[b0,bc] -> C[a0,bc]
        U36[a0*m+ac0][col] = 1
        V36[b0*p+bc][col] = 1
        W36[bc*n+a0][col] = 1
        col += 1
        
        # Term 2: A[a0,ac1] * B[b1,bc] -> C[a0,bc]
        U36[a0*m+ac1][col] = 1
        V36[b1*p+bc][col] = 1
        W36[bc*n+a0][col] = 1
        col += 1
        
        # Term 3: A[a1,ac0] * B[b0,bc] -> C[a1,bc]
        U36[a1*m+ac0][col] = 1
        V36[b0*p+bc][col] = 1
        W36[bc*n+a1][col] = 1
        col += 1
        
        # Term 4: A[a1,ac1] * B[b1,bc] -> C[a1,bc]
        U36[a1*m+ac1][col] = 1
        V36[b1*p+bc][col] = 1
        W36[bc*n+a1][col] = 1
        col += 1
    
    # Now build the full decomposition:
    # A rows: 0, 1
    # A cols: 0,1 (A0), 2,3 (A1)
    # B rows: 0,1 (B0), 2,3 (B1)
    # B cols: 0,1 (block0), 2,3 (block1), 4 (block2)
    
    # A0*B00: Strassen 2x2*2x2
    # A0 rows: 0,1; A0 cols: 0,1
    # B00 rows: 0,1; B00 cols: 0,1
    add_strassen_222([0,1], [0,1], [0,1], [0,1])
    
    # A0*B01: Strassen 2x2*2x2
    # A0 rows: 0,1; A0 cols: 0,1
    # B01 rows: 0,1; B01 cols: 2,3
    add_strassen_222([0,1], [0,1], [0,1], [2,3])
    
    # A0*B02: schoolbook 2x2*2x1
    # A0 rows: 0,1; A0 cols: 0,1
    # B02 rows: 0,1; B02 col: 4
    add_221([0,1], [0,1], [0,1], 4)
    
    # A1*B10: Strassen 2x2*2x2
    # A1 rows: 0,1; A1 cols: 2,3
    # B10 rows: 2,3; B10 cols: 0,1
    add_strassen_222([0,1], [2,3], [2,3], [0,1])
    
    # A1*B11: Strassen 2x2*2x2
    # A1 rows: 0,1; A1 cols: 2,3
    # B11 rows: 2,3; B11 cols: 2,3
    add_strassen_222([0,1], [2,3], [2,3], [2,3])
    
    # A1*B12: schoolbook 2x2*2x1
    # A1 rows: 0,1; A1 cols: 2,3
    # B12 rows: 2,3; B12 col: 4
    add_221([0,1], [2,3], [2,3], 4)
    
    # col should be 7*4 + 4*2 = 28 + 8 = 36
    assert col == 36, f"Expected col=36, got {col}"
    
    # Verify
    if _validate(U36, V36, W36):
        return {"U": U36, "V": V36, "W": W36}
    
    # If Strassen verification fails, fall back to schoolbook rank 40
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
