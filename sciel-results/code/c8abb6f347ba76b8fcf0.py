"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # A is 2x3, B is 3x6.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (2x6) and B2 (1x6).
    # AB = A1*B1 + A2*B2.
    
    # Part 1: A2 (2x1) * B2 (1x6)
    # This is an outer product of a 2x1 vector and a 1x6 vector.
    # Rank is 2*6 = 12.
    # U indices for A2: rows 0,1 of A (cols 2). Wait, A is 2x3.
    # A indices: i in 0..1, j in 0..2.
    # A2 corresponds to j=2.
    # B2 corresponds to k=2 (row 2 of B). B is 3x6. k in 0..2, l in 0..5.
    # B2 is row 2 of B.
    # U for A2: row i*m+j. i in 0..1, j=2.
    # V for B2: row j*p+k. j=2, k in 0..5? No.
    # B is 3x6. Rows are j (0..2), cols are k (0..5).
    # In the problem statement: V has shape (m*p, rank). Index is j*p+k.
    # So for B2 (row 2 of B), the index in V is 2*6 + k = 12 + k.
    # W for output: C is 2x6. Index is k*n+i. k in 0..5, i in 0..1.
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal rank
        U.append([0] * rank + [val]) # This is inefficient, let's build lists
        # Actually, let's build the matrices properly.
        pass

    # Let's build the lists directly.
    # Total rank = 12 + 21 = 33.
    total_rank = 33
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A2 * B2 (Rank 12)
    # A2 is col 2 of A. Indices i=0,1; j=2.
    # B2 is row 2 of B. Indices j=2; k=0..5.
    # Output C is 2x6. Indices i=0,1; k=0..5.
    for i in range(n):
        for k in range(p):
            # U[i*m + 2][col] = 1
            # V[2*p + k][col] = 1  -> V[12 + k]
            # W[k*n + i][col] = 1
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A1 * B1 (Rank 21)
    # A1 is cols 0,1 of A. 2x2 matrix.
    # B1 is rows 0,1 of B. 2x6 matrix.
    # Split B1 into three 2x2 blocks: B1a (cols 0,1), B1b (cols 2,3), B1c (cols 4,5).
    # For each block, use Strassen (Rank 7).
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # We need to map these to the global indices.
    # A1 indices: i in 0..1, j in 0..1.
    # B1a indices: j in 0..1, k in 0..1.
    # B1b indices: j in 0..1, k in 2..3.
    # B1c indices: j in 0..1, k in 4..5.
    
    def add_strassen_block(a_base_j, b_base_k):
        # a_base_j is 0 for A1 (cols 0,1)
        # b_base_k is 0, 2, or 4 for B1 blocks
        nonlocal col
        # U indices for A1: i*m + j. i in 0..1, j in 0..1.
        # V indices for B1: j*p + k. j in 0..1, k in b_base_k..b_base_k+1.
        # W indices for output: k*n + i. k in b_base_k..b_base_k+1, i in 0..1.
        
        # M1
        # U: A00 (i=0,j=0), A11 (i=1,j=1)
        # V: B00 (j=0,k=b), B11 (j=1,k=b+1)
        # W: C00 (k=b,i=0), C11 (k=b+1,i=1), C01 (k=b+1,i=0), C10 (k=b,i=1)
        # Strassen formula:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[0*m + 0][col] += 1
        U[1*m + 1][col] += 1
        V[0*p + b_base_k][col] += 1
        V[1*p + (b_base_k+1)][col] += 1
        W[(b_base_k)*n + 0][col] += 1
        W[(b_base_k+1)*n + 1][col] += 1
        W[(b_base_k+1)*n + 0][col] -= 1
        W[(b_base_k)*n + 1][col] -= 1
        col += 1
        
        # Term 2: M2 = (A10+A11)(B00)
        U[1*m + 0][col] += 1
        U[1*m + 1][col] += 1
        V[0*p + b_base_k][col] += 1
        W[(b_base_k)*n + 1][col] += 1
        W[(b_base_k+1)*n + 1][col] -= 1
        col += 1
        
        # Term 3: M3 = (A00)(B10-B11)
        U[0*m + 0][col] += 1
        V[1*p + b_base_k][col] += 1
        V[1*p + (b_base_k+1)][col] -= 1
        W[(b_base_k+1)*n + 0][col] += 1
        W[(b_base_k)*n + 1][col] += 1
        col += 1
        
        # Term 4: M4 = (A11)(B01-B10)
        U[1*m + 1][col] += 1
        V[0*p + (b_base_k+1)][col] += 1
        V[1*p + b_base_k][col] -= 1
        W[(b_base_k)*n + 1][col] += 1
        W[(b_base_k+1)*n + 0][col] -= 1
        col += 1
        
        # Term 5: M5 = (A00+A01)(B11)
        U[0*m + 0][col] += 1
        U[0*m + 1][col] += 1
        V[1*p + (b_base_k+1)][col] += 1
        W[(b_base_k+1)*n + 1][col] += 1
        W[(b_base_k)*n + 0][col] -= 1
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        U[1*m + 0][col] += 1
        U[0*m + 0][col] -= 1
        V[0*p + b_base_k][col] += 1
        V[1*p + b_base_k][col] += 1
        W[(b_base_k+1)*n + 1][col] += 1
        col += 1
        
        # Term 7: M7 = (A01-A11)(B01+B11)
        U[0*m + 1][col] += 1
        U[1*m + 1][col] -= 1
        V[0*p + (b_base_k+1)][col] += 1
        V[1*p + (b_base_k+1)][col] += 1
        W[(b_base_k+1)*n + 0][col] += 1
        col += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    add_strassen_block(0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
