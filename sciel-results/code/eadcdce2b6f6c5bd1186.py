"""Rank-97 exact tensor decomposition for 4x4 x 4x7 matrix multiplication.
The 4x4x4 sub-block uses a rank-49 Strassen-style decomposition (7 tensor products
each of rank 7), and the 4x4x3 sub-block uses schoolbook rank 48."""

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7

    # Build a rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    # Returns U (4 x 7), V (4 x 7), W (4 x 7) where W[k*n+i] for 2x2 output.
    def strassen_222():
        # 7 products: each is (linear form in A) * (linear form in B) -> output C
        # A is 2x2, B is 2x2, C = A*B
        # We need U[ai+bj, r] * V[bj+ck, r] * W[ck+ai, r] to reconstruct C[ai, ck]
        # where a in {0,1}, b in {0,1}, c in {0,1}
        
        # U index: a*2 + b (row of A, col of A) -> 4 rows
        # V index: b*2 + c (col of A, row of B) -> 4 rows  
        # W index: c*2 + a (col of B, row of C... wait)
        # Actually for the tensor: T[ai, bj, ck] = delta_{a,c} delta_{b,d} ... no.
        # Matrix mult: C[ai, ck] = sum_b A[ai, bj] * B[bj, ck]
        # Tensor T[ai, bj, ck] = 1 if this is the (ai,bj,ck) term in the sum.
        # But the standard tensor form: sum_r U[ai, r] V[bj, r] W[ck, r] = C[ai, ck] when summed over b.
        # Wait, the tensor T has indices (ai, bj, ck) and T[ai,bj,ck] = delta(ai, ck_row) ... 
        # Let me re-read the contract.
        # U shape: (n*m, rank) = (4, 7) for 2x2x2
        # V shape: (m*p, rank) = (4, 7)
        # W shape: (p*n, rank) = (4, 7)
        # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (A*B)[i,k]
        # For 2x2x2: i in {0,1}, j in {0,1}, k in {0,1}
        # U index: i*2+j (0..3), V index: j*2+k (0..3), W index: k*2+i (0..3)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen's 7 products for 2x2:
        # P1 = (A00 + A11)(B00 + B11)
        # P2 = (A10 + A11)B00
        # P3 = A00(B01 - B11)
        # P4 = A11(B10 - B00)
        # P5 = (A00 + A01)B11
        # P6 = (A10 - A00)(B00 + B01)
        # P7 = (A01 - A11)(B10 + B11)
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # U[i*2+j, r]: coefficient of A[i,j] in product r
        # V[j*2+k, r]: coefficient of B[j,k] in product r
        # W[k*2+i, r]: coefficient in output C[i,k] from product r
        
        # Product 1: (A00 + A11)(B00 + B11)
        U[0*2+0][0] = 1  # A00
        U[1*2+1][0] = 1  # A11
        V[0*2+0][0] = 1  # B00
        V[1*2+1][0] = 1  # B11
        W[0*2+0][0] = 1  # C00
        W[0*2+1][0] = 0  # C01 (none)
        W[1*2+0][0] = 0  # C10 (none)
        W[1*2+1][0] = 1  # C11
        
        # Product 2: (A10 + A11)B00
        U[1*2+0][1] = 1  # A10
        U[1*2+1][1] = 1  # A11
        V[0*2+0][1] = 1  # B00
        W[0*2+0][1] = 0
        W[0*2+1][1] = 0
        W[1*2+0][1] = 1  # C10
        W[1*2+1][1] = 0
        
        # Product 3: A00(B01 - B11)
        U[0*2+0][2] = 1  # A00
        V[0*2+1][2] = 1  # B01
        V[1*2+1][2] = -1 # B11
        W[0*2+0][2] = 0
        W[0*2+1][2] = 1  # C01
        W[1*2+0][2] = 0
        W[1*2+1][2] = 0
        
        # Product 4: A11(B10 - B00)
        U[1*2+1][3] = 1  # A11
        V[1*2+0][3] = 1  # B10
        V[0*2+0][3] = -1 # B00
        W[0*2+0][3] = 1  # C00
        W[0*2+1][3] = 0
        W[1*2+0][3] = 1  # C10
        W[1*2+1][3] = 0
        
        # Product 5: (A00 + A01)B11
        U[0*2+0][4] = 1  # A00
        U[0*2+1][4] = 1  # A01
        V[1*2+1][4] = 1  # B11
        W[0*2+0][4] = -1 # C00
        W[0*2+1][4] = 1  # C01
        W[1*2+0][4] = 0
        W[1*2+1][4] = 0
        
        # Product 6: (A10 - A00)(B00 + B01)
        U[1*2+0][5] = 1  # A10
        U[0*2+0][5] = -1 # A00
        V[0*2+0][5] = 1  # B00
        V[0*2+1][5] = 1  # B01
        W[0*2+0][5] = 0
        W[0*2+1][5] = 0
        W[1*2+0][5] = 0
        W[1*2+1][5] = 1  # C11
        
        # Product 7: (A01 - A11)(B10 + B11)
        U[0*2+1][6] = 1  # A01
        U[1*2+1][6] = -1 # A11
        V[1*2+0][6] = 1  # B10
        V[1*2+1][6] = 1  # B11
        W[0*2+0][6] = 1  # C00
        W[0*2+1][6] = 0
        W[1*2+0][6] = 0
        W[1*2+1][6] = 0
        
        return U, V, W
    
    # Now build the full 4x4x7 decomposition.
    # Split the 4x4 A and 4x7 B into blocks.
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22
    # B is 4x7, split rows into 2x2 blocks (4 rows), columns into 7 columns.
    # For the 4x4x4 part (first 4 columns of B): use Strassen with 7 rank-7 sub-decompositions.
    # For the 4x4x3 part (last 3 columns of B): use schoolbook.
    
    # First, let's build the 4x4x4 part using Strassen.
    # We have 7 Strassen products, each involving a 2x2 matrix multiply.
    # Each 2x2 multiply has rank 7, so total rank = 7*7 = 49.
    
    sU, sV, sW = strassen_222()
    
    # For the 4x4x4 part:
    # A block indices: (a1, b1) where a1 in {0,1} is block-row, b1 in {0,1} is block-col
    # B block indices: (b1, c1) where b1 in {0,1} is block-row, c1 in {0,1} is block-col (for 4 cols)
    # C block indices: (a1, c1)
    
    # The 4x4x4 tensor uses indices:
    # i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}
    # Block mapping: i = a1*2 + ai, j = b1*2 + bj, k = c1*2 + ck
    
    # For each Strassen product r (0..6), and each 2x2x2 decomposition (0..6):
    # We need to place the coefficients.
    
    rank_4x4x4 = 49
    U_444 = [[0]*rank_4x4x4 for _ in range(16)]  # 4*4
    V_444 = [[0]*rank_4x4x4 for _ in range(16)]  # 4*4
    W_444 = [[0]*rank_4x4x4 for _ in range(16)]  # 4*4
    
    for r in range(7):  # Strassen product
        for s in range(7):  # 2x2x2 decomposition
            col = r * 7 + s
            # U: index i*4+j, where i=a1*2+ai, j=b1*2+bj
            # sU[ai*2+bj, s] gives coefficient of A[ai,bj] in 2x2 product s
            # But we need to map to the 4x4 index.
            for a1 in range(2):
                for b1 in range(2):
                    # The 2x2 sub-matrix is at block (a1, b1) of A
                    # So A block (a1,b1) has elements A[a1*2+ai, b1*2+bj]
                    # sU[ai*2+bj, s] is the coefficient
                    for ai in range(2):
                        for bj in range(2):
                            coeff = sU[ai*2+bj][s]
                            if coeff != 0:
                                i = a1*2 + ai
                                j = b1*2 + bj
                                U_444[i*4+j][col] = coeff
            
            # V: index j*4+k, where j=b1*2+bj, k=c1*2+ck
            for b1 in range(2):
                for c1 in range(2):
                    for bj in range(2):
                        for ck in range(2):
                            coeff = sV[bj*2+ck][s]
                            if coeff != 0:
                                j = b1*2 + bj
                                k = c1*2 + ck
                                V_444[j*4+k][col] = coeff
            
            # W: index k*4+i, where k=c1*2+ck, i=a1*2+ai
            for c1 in range(2):
                for a1 in range(2):
                    for ck in range(2):
                        for ai in range(2):
                            coeff = sW[ck*2+ai][s]
                            if coeff != 0:
                                k = c1*2 + ck
                                i = a1*2 + ai
                                W_444[k*4+i][col] = coeff
    
    # Now build the 4x4x3 part using schoolbook (rank 48).
    rank_443 = 48
    U_443 = [[0]*rank_443 for _ in range(16)]
    V_443 = [[0]*rank_443 for _ in range(28)]  # 4*7
    W_443 = [[0]*rank_443 for _ in range(28)]  # 7*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):  # Only last 3 columns
                kk = 4 + k  # k in {4,5,6}
                U_443[i*4+j][col] = 1
                V_443[j*7+kk][col] = 1
                W_443[kk*4+i][col] = 1
                col += 1
    
    # Combine: total rank = 49 + 48 = 97
    total_rank = 97
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # Copy 4x4x4 part (first 49 columns)
    for i in range(16):
        for c in range(rank_4x4x4):
            U[i][c] = U_444[i][c]
            W[c*4 + i%4][c] = 0  # placeholder, will fill properly
    
    # Actually, let me be more careful with W for the 444 part.
    # W_444 has shape (16, 49) with index k*4+i for k in {0,1,2,3}, i in {0,1,2,3}
    # But the full W has shape (28, 97) with index k*4+i for k in {0,...,6}, i in {0,...,3}
    
    for i in range(16):
        for c in range(rank_4x4x4):
            U[i][c] = U_444[i][c]
    
    for j in range(16):
        for c in range(rank_4x4x4):
            V[j][c] = V_444[j][c]
    
    for k in range(16):
        for c in range(rank_4x4x4):
            W[k][c] = W_444[k][c]
    
    # Copy 4x4x3 part (columns 49..96)
    for i in range(16):
        for c in range(rank_443):
            U[i][rank_4x4x4 + c] = U_443[i][c]
    
    for j in range(28):
        for c in range(rank_443):
            V[j][rank_4x4x4 + c] = V_443[j][c]
    
    for k in range(28):
        for c in range(rank_443):
            W[k][rank_4x4x4 + c] = W_443[k][c]
    
    return {"U": U, "V": V, "W": W}
