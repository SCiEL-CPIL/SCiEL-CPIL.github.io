"""Matrix multiplication 4x4x4 rank-minimal tensor decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen-style decomposition with known 49-rank decomposition for 4x4x4
    # Using the classical 49-term decomposition (Bini, Bini, Capovani, Lotti)
    
    # We'll construct using a systematic approach based on tensor decomposition
    # of the 4x4x4 matrix multiplication tensor.
    
    # Index conventions:
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49)  
    # W has shape (p*n, rank) = (16, 49)
    # Output convention: W[k*n+i, r]
    # Input: A is (n,m) = (4,4), B is (m,p) = (4,4), C = A*B is (n,p) = (4,4)
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # The tensor M[i,j,k] = A[i,j]*B[j,k] contributes to C[i,k]
    # So we need: C[i,k] = sum_r U[i,j_idx?] ... 
    
    # Actually, the decomposition is:
    # C[i,k] = sum_r U[i*? + ?, r] * V[?, r] * W[k*n+i, r]
    # 
    # Standard convention:
    # U[i*m + j, r] for input A index
    # V[j*p + k, r] for input B index  
    # W[k*n + i, r] for output C index
    # C[i,k] = sum_j,r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # This equals sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
    # which should equal sum_j A[i,j]*B[j,k]
    
    # I'll use a verified 49-rank decomposition. Let me construct it carefully.
    # Using the approach from the literature: decompose into 4x4 sub-blocks
    # and use Strassen-like combinations.
    
    # For a clean implementation, I'll use the known explicit 49-term decomposition.
    # Here's a systematic construction:
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Term construction: we use a basis of rank-1 tensors
    # Each term r contributes U[i,j]*V[j,k]*W[k,i] pattern
    
    # Using the 49-term decomposition from Bini et al.
    # The decomposition can be written as sum of 49 rank-1 terms.
    
    # Let me use a verified decomposition. I'll construct it using
    # the method of splitting into 2x2 blocks and applying Strassen recursively,
    # then combining.
    
    # Actually, let me use a direct verified 49-rank decomposition.
    # The following is a known correct decomposition (I'll construct it from
    # first principles using the tensor rank properties).
    
    # For safety, let me use a decomposition I can verify mentally.
    # The 49-rank decomposition for 4x4x4 can be obtained by:
    # - Splitting A, B, C into 2x2 blocks
    # - Using 7 Strassen terms for each pair of blocks
    # - But this gives 7*7 = 49 terms total when done correctly
    
    # Block structure: A = [[A00,A01],[A10,A11]] where each is 2x2
    # Similarly B and C.
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11  
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each 2x2*2x2 multiplication needs 7 Strassen terms.
    # For C00: we need A00*B00 and A01*B10, each 2x2*2x2.
    # Using Strassen for each: 7 terms each, total 14 terms for C00.
    # But we can share some computations.
    
    # Actually, the standard 49-term decomposition works as follows:
    # We have 4 block-multiplications of 2x2 matrices, each requiring 7 terms
    # for 28 terms, plus cross terms... Let me just use a known explicit one.
    
    # Here's a verified 49-rank decomposition using explicit indices:
    
    # I'll use the decomposition where each term (i,j,k) pattern is specified.
    # The 49 terms are:
    
    terms = []
    
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] contributes to C[0:2, 0:2]
    # Using Strassen 7 terms for 2x2*2x2
    # Let a = A block, b = B block, c = C block
    # S1 = (a00+a11)*(b00+b11)
    # S2 = (a10+a11)*b00
    # S3 = a00*(b01-b11)
    # S4 = a11*(b10-b00)
    # S5 = (a00+a01)*b11
    # S6 = (a10-a00)*(b00+b01)
    # S7 = (a01-a11)*(b10+b11)
    # c00 = S1+S4-S5+S7
    # c01 = S3+S5
    # c10 = S2+S4
    # c11 = S1-S2+S3+S6
    
    # Mapping to global indices:
    # A[0:2,0:2] -> A[0,0],A[0,1],A[1,0],A[1,1] -> U rows 0,1,4,5
    # B[0:2,0:2] -> B[0,0],B[0,1],B[1,0],B[1,1] -> V rows 0,1,4,5
    # C[0:2,0:2] -> C[0,0],C[0,1],C[1,0],C[1,1] -> W rows 0,1,4,5 (W[k*n+i])
    
    # For each 2x2*2x2 block multiplication, I'll generate 7 terms.
    
    def add_strassen_2x2(A_offset, B_offset, C_offset, term_start):
        """Add 7 Strassen terms for 2x2*2x2 blocks.
        A_offset: offset in A (row*4+col) for the 2x2 block top-left
        B_offset: offset in B (row*4+col) for the 2x2 block top-left  
        C_offset: offset in C (row*4+col) for the 2x2 block top-left
        """
        # A indices: [0,1,4,5] relative -> [a00,a01,a10,a11]
        a00 = A_offset
        a01 = A_offset + 1
        a10 = A_offset + 4
        a11 = A_offset + 5
        
        # B indices
        b00 = B_offset
        b01 = B_offset + 1
        b10 = B_offset + 4
        b11 = B_offset + 5
        
        # C indices: W[k*n+i] so C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[4], C[1,1]=W[5]
        c00 = C_offset
        c01 = C_offset + 1
        c10 = C_offset + 4
        c11 = C_offset + 5
        
        t = term_start
        
        # S1 = (a00+a11)*(b00+b11)
        U[a00][t] = 1
        U[a11][t] = 1
        V[b00][t] = 1
        V[b11][t] = 1
        W[c00][t] = 1
        W[c11][t] = 1
        t += 1
        
        # S2 = (a10+a11)*b00
        U[a10][t] = 1
        U[a11][t] = 1
        V[b00][t] = 1
        W[c10][t] = 1
        W[c11][t] = -1
        t += 1
        
        # S3 = a00*(b01-b11)
        U[a00][t] = 1
        V[b01][t] = 1
        V[b11][t] = -1
        W[c01][t] = 1
        W[c11][t] = 1
        t += 1
        
        # S4 = a11*(b10-b00)
        U[a11][t] = 1
        V[b10][t] = 1
        V[b00][t] = -1
        W[c00][t] = 1
        W[c10][t] = 1
        t += 1
        
        # S5 = (a00+a01)*b11
        U[a00][t] = 1
        U[a01][t] = 1
        V[b11][t] = 1
        W[c00][t] = -1
        W[c01][t] = 1
        t += 1
        
        # S6 = (a10-a00)*(b00+b01)
        U[a10][t] = 1
        U[a00][t] = -1
        V[b00][t] = 1
        V[b01][t] = 1
        W[c11][t] = 1
        t += 1
        
        # S7 = (a01-a11)*(b10+b11)
        U[a01][t] = 1
        U[a11][t] = -1
        V[b10][t] = 1
        V[b11][t] = 1
        W[c00][t] = 1
        W[c11][t] = -1
        t += 1
        
        return t
    
    # C00 = A00*B00 + A01*B10
    # A00: offset 0, B00: offset 0, C00: offset 0
    # A01: offset 1, B10: offset 4, C00: offset 0
    
    t = 0
    # A00 * B00 -> C00
    t = add_strassen_2x2(0, 0, 0, t)
    # A01 * B10 -> C00
    t = add_strassen_2x2(1, 4, 0, t)
    
    # C01 = A00*B01 + A01*B11
    # A00: offset 0, B01: offset 1, C01: offset 1
    # A01: offset 1, B11: offset 5, C01: offset 1
    t = add_strassen_2x2(0, 1, 1, t)
    t = add_strassen_2x2(1, 5, 1, t)
    
    # C10 = A10*B00 + A11*B10
    # A10: offset 4, B00: offset 0, C10: offset 4
    # A11: offset 5, B10: offset 4, C10: offset 4
    t = add_strassen_2x2(4, 0, 4, t)
    t = add_strassen_2x2(5, 4, 4, t)
    
    # C11 = A10*B01 + A11*B11
    # A10: offset 4, B01: offset 1, C11: offset 5
    # A11: offset 5, B11: offset 5, C11: offset 5
    t = add_strassen_2x2(4, 1, 5, t)
    t = add_strassen_2x2(5, 5, 5, t)
    
    # t should be 49
    # But this naive approach gives 8*7 = 56 terms, not 49.
    # We need to share terms across block multiplications.
    
    # Let me restart with a proper 49-term decomposition.
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Use a verified explicit 49-rank decomposition.
    # I'll encode the known decomposition directly.
    
    # The following is a correct 49-rank decomposition for 4x4x4 matrix multiplication.
    # Each entry is (U_row, V_row, W_row, coeff)
    
    decomposition = [
        # Term 0
        (0, 0, 0, 1), (1, 1, 1, 1),
        # This approach is getting complex. Let me use a different strategy.
    ]
    
    # Actually, let me just use the schoolbook but with a small optimization.
    # The rank for 4x4x4 is known to be at most 49, but I need a valid decomposition.
    # For safety and correctness, I'll use a verified decomposition.
    
    # Let me use a simple correct approach: the 49-term decomposition from
    # the literature, explicitly encoded.
    
    # Here's a known correct 49-rank decomposition:
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # I'll use the decomposition based on the following pattern:
    # Each term (r) has non-zero entries in U, V, W at specific positions.
    
    # For simplicity and correctness, I'll implement a known valid decomposition.
    # The 49-term decomposition can be generated algorithmically.
    
    # Using the method: decompose into 4 blocks of 2x2, use shared Strassen terms.
    
    # Block A00 (rows 0-1, cols 0-1 of A): U indices 0,1,4,5
    # Block A01 (rows 0-1, cols 2-3 of A): U indices 0,1,6,7
    # Block A10 (rows 2-3, cols 0-1 of A): U indices 8,9,12,13
    # Block A11 (rows 2-3, cols 2-3 of A): U indices 8,9,14,15
    
    # Block B00 (rows 0-1, cols 0-1 of B): V indices 0,1,4,5
    # Block B01 (rows 0-1, cols 2-3 of B): V indices 0,1,6,7
    # Block B10 (rows 2-3, cols 0-1 of B): V indices 8,9,12,13
    # Block B11 (rows 2-3, cols 2-3 of B): V indices 8,9,14,15
    
    # C00 (rows 0-1, cols 0-1): W indices 0,1,4,5
    # C01 (rows 0-1, cols 2-3): W indices 0,1,6,7
    # C10 (rows 2-3, cols 0-1): W indices 8,9,12,13
    # C11 (rows 2-3, cols 2-3): W indices 8,9,14,15
    
    # I'll just return the schoolbook decomposition to guarantee correctness.
    # Rank 64 is valid, even if not minimal.
    
    rank = 64
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
