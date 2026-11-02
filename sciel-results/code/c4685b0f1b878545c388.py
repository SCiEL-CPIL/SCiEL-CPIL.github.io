"""Matrix multiplication 2x3x8 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-44 decomposition for 2x3x8 matrix multiplication.
    
    Strategy: Split the 3x8 product into two independent 3x4 blocks.
    For each 3x4 block (multiplying 2x3 by 3x4), use:
    - Strassen rank-7 decomposition for the 2x2x2 sub-tensor (columns 0,1 of A with rows 0,1 of B)
    - 4 rank-1 terms for the third column of A (column 2) with all 4 rows of B
    This gives 7 + 4 = 11 per block, total 22... wait, let me reconsider.
    
    Actually, for 2x3 * 3x4:
    - The 2x3 matrix A has 6 entries, the 3x4 matrix B has 12 entries
    - Schoolbook would be 2*3*4 = 24
    - We can split B into two 3x2 blocks: B = [B1, B2]
    - For each 3x2 block, we multiply 2x3 * 3x2
    - For 2x3 * 3x2: split A into two 2x2 blocks plus one 2x1 column? No, A is 2x3.
    
    Let me reconsider. For 2x3 * 3x4:
    - A is 2x3: rows i=0,1; cols j=0,1,2
    - B is 3x4: rows j=0,1,2; cols k=0,1,2,3
    - C is 2x4: rows i=0,1; cols k=0,1,2,3
    
    We can group the 4 columns of B into two groups of 2: {k=0,1} and {k=2,3}
    For each group of 2 columns, we have a 2x3 * 3x2 subproblem.
    
    For 2x3 * 3x2:
    - A is 2x3, B is 3x2, C is 2x2
    - Split A's columns: A = [A_left, A_right] where A_left is 2x2 (cols 0,1) and A_right is 2x1 (col 2)
    - Then C = A_left * B_left + A_right * B_right where B_left is 3x2 (rows 0,1) and B_right is 3x2 (row 2)
    
    Wait, that doesn't work cleanly. Let me think differently.
    
    For 2x3 * 3x2:
    C[i,k] = sum_j A[i,j] * B[j,k] for j=0,1,2
    
    We can use the following approach:
    - For the 2x2 part (j=0,1), use Strassen's algorithm which gives rank 7 for 2x2*2x2
    - For the j=2 part, we need A[i,2] * B[2,k] for i=0,1 and k=0,1, which is 4 rank-1 terms
    
    So 2x3 * 3x2 has rank at most 7 + 4 = 11.
    
    Then 2x3 * 3x4 = 2x3 * 3x2 (cols 0,1) + 2x3 * 3x2 (cols 2,3) has rank at most 11 + 11 = 22.
    
    And 2x3 * 3x8 = 2x3 * 3x4 (cols 0,3) + 2x3 * 3x4 (cols 4,7) has rank at most 22 + 22 = 44.
    
    Let me verify the Strassen decomposition for 2x2*2x2:
    Given A = [[a,b],[c,d]] and B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    Strassen's 7 products:
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    C[0,0] = P1 + P4 - P5 + P7
    C[0,1] = P3 + P5
    C[1,0] = P2 + P4
    C[1,1] = P1 - P2 + P3 + P6
    
    Each P is a product of a linear combination of A entries and a linear combination of B entries.
    So in tensor decomposition terms, each P corresponds to one rank-1 term.
    
    For the 2x2*2x2 subproblem (A_left is 2x2, B_top is 2x2):
    A_left = A[:, 0:2] (2x2), B_top = B[0:2, :] (2x2 for the 2 columns we're considering)
    
    Let me build the full decomposition:
    
    For 2x3 * 3x4 (one block of 4 columns):
    - Strassen part: 7 terms for the 2x2*2x2 using A[:,0:2] and B[0:2,:]
    - Direct part: 4 terms for A[:,2] * B[2,:]
    
    Total per 3x4 block: 11
    Total for 3x8: 22... 
    
    Wait, I made an error. Let me recount.
    
    2x3 * 3x8:
    - Split into two 2x3 * 3x4 blocks
    - Each 2x3 * 3x4: split into two 2x3 * 3x2 sub-blocks
    - Each 2x3 * 3x2: Strassen(7) + direct(4) = 11
    - Each 2x3 * 3x4: 2 * 11 = 22
    - Total: 2 * 22 = 44
    
    Yes, rank 44.
    
    Now let me construct the actual U, V, W matrices.
    
    n=2, m=3, p=8
    U has shape (n*m, rank) = (6, 44)
    V has shape (m*p, rank) = (24, 44)
    W has shape (p*n, rank) = (16, 44)
    
    Indexing:
    U[i*m + j, c] for i in 0..1, j in 0..2
    V[j*p + k, c] for j in 0..2, k in 0..7
    W[k*n + i, c] for k in 0..7, i in 0..1
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # For each of the two 3x4 blocks (k offset 0 and 4)
    for k_off in [0, 4]:
        # For each of the two 3x2 sub-blocks within the 3x4 (k offset 0 and 2)
        for k_sub_off in [0, 2]:
            # Strassen part: 7 terms for 2x2*2x2 (A cols 0,1; B rows 0,1)
            # A indices: i in {0,1}, j in {0,1}
            # B indices: j in {0,1}, k in {k_off+k_sub_off, k_off+k_sub_off+1}
            
            # P1 = (a00+a11)(e00+e11)
            U[0*m+0][col] = 1; U[1*m+1][col] = 1
            V[0*p+(k_off+k_sub_off)][col] = 1; V[1*p+(k_off+k_sub_off+1)][col] = 1
            W[(k_off+k_sub_off)*n+0][col] = 1; W[(k_off+k_sub_off+1)*n+1][col] = 1
            W[(k_off+k_sub_off)*n+1][col] = 1; W[(k_off+k_sub_off+1)*n+0][col] = 1
            col += 1
            
            # P2 = (c00+c11)*e00  => (a10+a11)*b00... wait, let me be careful with indexing
            # A[i,j]: i=row, j=col. a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
            # B[j,k]: j=row, k=col. e00=B[0,0], e01=B[0,1], e10=B[1,0], e11=B[1,1]
            # C[i,k]: C[0,0]=a00*e00+a01*e10, C[0,1]=a00*e01+a01*e11
            #          C[1,0]=a10*e00+a11*e10, C[1,1]=a10*e01+a11*e11
            
            # P2 = (a10+a11)*e00 = (A[1,0]+A[1,1])*B[0,0]
            U[1*m+0][col] = 1; U[1*m+1][col] = 1
            V[0*p+(k_off+k_sub_off)][col] = 1
            W[(k_off+k_sub_off)*n+1][col] = 1
            col += 1
            
            # P3 = a00*(e01-e11) = A[0,0]*(B[0,1]-B[1,1])
            U[0*m+0][col] = 1
            V[0*p+(k_off+k_sub_off+1)][col] = 1; V[1*p+(k_off+k_sub_off+1)][col] = -1
            W[(k_off+k_sub_off+1)*n+0][col] = 1
            col += 1
            
            # P4 = a11*(e10-e00) = A[1,1]*(B[1,0]-B[0,0])
            U[1*m+1][col] = 1
            V[1*p+(k_off+k_sub_off)][col] = 1; V[0*p+(k_off+k_sub_off)][col] = -1
            W[(k_off+k_sub_off)*n+1][col] = 1
            col += 1
            
            # P5 = (a00+a01)*e11 = (A[0,0]+A[0,1])*B[1,1]
            U[0*m+0][col] = 1; U[0*m+1][col] = 1
            V[1*p+(k_off+k_sub_off+1)][col] = 1
            W[(k_off+k_sub_off+1)*n+0][col] = 1
            col += 1
            
            # P6 = (a10-a00)*(e00+e01) = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
            U[1*m+0][col] = 1; U[0*m+0][col] = -1
            V[0*p+(k_off+k_sub_off)][col] = 1; V[0*p+(k_off+k_sub_off+1)][col] = 1
            W[(k_off+k_sub_off)*n+1][col] = 1; W[(k_off+k_sub_off+1)*n+1][col] = 1
            col += 1
            
            # P7 = (a01-a11)*(e10+e11) = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
            U[0*m+1][col] = 1; U[1*m+1][col] = -1
            V[1*p+(k_off+k_sub_off)][col] = 1; V[1*p+(k_off+k_sub_off+1)][col] = 1
            W[(k_off+k_sub_off)*n+0][col] = 1; W[(k_off+k_sub_off+1)*n+0][col] = 1
            col += 1
            
            # Direct part: 4 terms for A[:,2] * B[2,:]
            # For i in {0,1}, k in {k_off+k_sub_off, k_off+k_sub_off+1}:
            # C[i,k] += A[i,2] * B[2,k]
            for i in range(2):
                for k_local in range(2):
                    k_idx = k_off + k_sub_off + k_local
                    U[i*m+2][col] = 1
                    V[2*p+k_idx][col] = 1
                    W[k_idx*n+i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
