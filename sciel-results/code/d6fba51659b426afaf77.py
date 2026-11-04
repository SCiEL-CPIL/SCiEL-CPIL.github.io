"""Program for matrix multiplication 4x4 * 4x9 tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy: Use Strassen's algorithm (rank 7) for 2x2 blocks.
    Decompose the 4x4 * 4x9 product using Kronecker structure.
    
    For 4x4 * 4x4: rank 7*7 = 49
    For 4x4 * 4x1: rank 7*1 = 7 (or use schoolbook 4*1*4=16, but 7 is better)
    Total rank: 49 + 49 + 7 = 105? Wait, let me think more carefully.
    
    Actually, let's use the block decomposition:
    Split 4x9 into three blocks: 4x4, 4x4, 4x1.
    
    For each 4x4 * 4x4 block: use Strassen rank 7 for 2x2, Kronecker with 2x2 identity structure.
    Actually, 4x4 can be seen as 2x2 block matrix where each entry is 2x2.
    Using Strassen: rank 7 for the 2x2 scalar matrix, then each of the 7 terms is a 2x2 * 2x2 product.
    But wait, that's not quite right. Let me reconsider.
    
    Better approach: 
    4x4 * 4x4: The standard Strassen-based decomposition has rank 7*4 = 28? No.
    
    Let me use a cleaner approach. For 2x2 * 2x2, Strassen gives rank 7.
    For 4x4 * 4x4, we can view it as a 2x2 block matrix where each block is 2x2.
    The multiplication involves 8 block multiplications (schoolbook) or 7 block multiplications (Strassen-like).
    Each block multiplication is 2x2 * 2x2, which has rank 7 (Strassen).
    So total rank for 4x4 * 4x4 would be 7 * 7 = 49.
    
    For 4x4 * 4x1: We can view this similarly. The result is 4x1.
    Split into two 2x1 blocks. Each 2x4 * 2x1 product... 
    
    Actually, let's use a simpler and verified approach. Let me use the known rank for these dimensions.
    
    Known results:
    - 2x2 * 2x2: rank 7 (Strassen)
    - 4x4 * 4x4: rank 49 (Strassen Kronecker) or possibly lower
    - 4x4 * 4x1: rank 28 (using Strassen-like structure)
    
    Let me construct explicitly using the Kronecker product approach.
    
    For 4x4 * 4x4 with rank 49:
    Let S be the Strassen 2x2*2x2 tensor (rank 7).
    Then 4x4*4x4 tensor = S ⊗ S (Kronecker product), rank 7*7 = 49.
    
    For 4x4 * 4x1:
    We can write this as 4x4 * (4x1). 
    Using Strassen for 2x2 * 2x1: rank 7? Let's see.
    2x2 * 2x1: A(2x2) * B(2x1) = C(2x1).
    Schoolbook rank: 2*2*1 = 4.
    Strassen for 2x2 * 2x1: We can use 7 multiplications? 
    Actually, for 2x2 * 2x1, the minimal rank is 7? No, it's known to be 7 for square, but for rectangular it might differ.
    
    Let me just use schoolbook for the 4x1 part: rank = 4*4*1 = 16.
    Total: 49 + 49 + 16 = 114.
    
    Or better: use Strassen for all parts.
    4x4 * 4x1: Split 4x4 into 2x2 blocks. 
    Result 4x1 split into 2x1 blocks.
    Top-left block: 2x4 * 2x1 -> 2x1
    Top-right: 2x4 * 2x1 -> 2x1  
    Wait, the B matrix is 4x1, so it's a single column.
    
    Let me just implement the Kronecker product of Strassen for the 4x4*4x4 parts and schoolbook for 4x4*4x1.
    """
    n, m, p = 4, 4, 9
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper: Strassen 2x2*2x2 decomposition (rank 7)
    # Returns U, V, W each of shape (4, 7) for 2x2*2x2
    def strassen_2x2():
        # A(2x2) * B(2x2) = C(2x2)
        # Strassen's 7 products:
        # P1 = (A00 + A11) * (B00 + B11)
        # P2 = (A10 + A11) * B00
        # P3 = A00 * (B10 - B11)
        # P4 = A11 * (B01 - B00)
        # P5 = (A00 + A01) * B11
        # P6 = (A10 - A00) * (B00 + B10)
        # P7 = (A01 - A11) * (B01 + B11)
        # 
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        #
        # U: rows indexed by i*2+j (A's entries), columns by product number
        # V: rows indexed by j*2+k (B's entries), columns by product number  
        # W: rows indexed by k*2+i (C's entries), columns by product number
        # Wait, the trace convention: W[k*n+i, r] for n=2, so W[k*2+i, r]
        
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # A entries: A00=row0, A01=row1, A10=row2, A11=row3 (i*2+j: i=0,j=0->0; i=0,j=1->1; i=1,j=0->2; i=1,j=1->3)
        # B entries: B00=row0, B01=row1, B10=row2, B11=row3 (j*2+k: j=0,k=0->0; j=0,k=1->1; j=1,k=0->2; j=1,k=1->3)
        # C entries: C00=row0, C01=row1, C10=row2, C11=row3 (k*2+i: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3)
        
        # P1 = (A00+A11)*(B00+B11)
        U_s[0][0] = 1; U_s[3][0] = 1  # A00, A11
        V_s[0][0] = 1; V_s[3][0] = 1  # B00, B11
        W_s[0][0] = 1; W_s[3][0] = 1  # C00, C11
        
        # P2 = (A10+A11)*B00
        U_s[2][1] = 1; U_s[3][1] = 1  # A10, A11
        V_s[0][1] = 1  # B00
        W_s[2][1] = 1; W_s[3][1] = 1  # C10, C11
        
        # P3 = A00*(B10-B11)
        U_s[0][2] = 1  # A00
        V_s[2][2] = 1; V_s[3][2] = -1  # B10, -B11
        W_s[1][2] = 1; W_s[3][2] = 1  # C01, C11
        
        # P4 = A11*(B01-B00)
        U_s[3][3] = 1  # A11
        V_s[1][3] = 1; V_s[0][3] = -1  # B01, -B00
        W_s[0][3] = 1; W_s[2][3] = 1  # C00, C10
        
        # P5 = (A00+A01)*B11
        U_s[0][4] = 1; U_s[1][4] = 1  # A00, A01
        V_s[3][4] = 1  # B11
        W_s[0][4] = -1; W_s[1][4] = 1  # -C00, C01
        
        # P6 = (A10-A00)*(B00+B10)
        U_s[2][5] = 1; U_s[0][5] = -1  # A10, -A00
        V_s[0][5] = 1; V_s[2][5] = 1  # B00, B10
        W_s[3][5] = 1  # C11
        
        # P7 = (A01-A11)*(B01+B11)
        U_s[1][6] = 1; U_s[3][6] = -1  # A01, -A11
        V_s[1][6] = 1; V_s[3][6] = 1  # B01, B11
        W_s[0][6] = 1; W_s[1][6] = 1  # C00, C01
        
        return U_s, V_s, W_s
    
    U_2, V_2, W_2 = strassen_2x2()
    
    # For 4x4 * 4x4: Kronecker product of Strassen 2x2 with itself
    # The 4x4 matrix A can be seen as 2x2 blocks of 2x2 matrices
    # Block (I,J) of A is A[I*2:(I+1)*2, J*2:(J+1)*2]
    # Similarly for B and C.
    # Using Strassen on the block structure: 7 block products
    # Each block product is 2x2 * 2x2, which uses Strassen (rank 7)
    # Total rank: 7 * 7 = 49
    
    def kronecker_decomposition(U1, V1, W1, U2, V2, W2):
        """Kronecker product of two decompositions."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        n1 = len(U1)
        m1 = len(V1)
        p1 = len(W1)
        n2 = len(U2)
        m2 = len(V2)
        p2 = len(W2)
        
        rank = r1 * r2
        U_k = [[0]*rank for _ in range(n1*n2)]
        V_k = [[0]*rank for _ in range(m1*m2)]
        W_k = [[0]*rank for _ in range(p1*p2)]
        
        for a in range(r1):
            for b in range(r2):
                col = a * r2 + b
                for i1 in range(n1):
                    for i2 in range(n2):
                        U_k[i1*n2+i2][col] = U1[i1][a] * U2[i2][b]
                for j1 in range(m1):
                    for j2 in range(m2):
                        V_k[j1*m2+j2][col] = V1[j1][a] * V2[j2][b]
                for k1 in range(p1):
                    for k2 in range(p2):
                        W_k[k1*p2+k2][col] = W1[k1][a] * W2[k2][b]
        
        return U_k, V_k, W_k
    
    U_4x4, V_4x4, W_4x4 = kronecker_decomposition(U_2, V_2, W_2, U_2, V_2, W_2)
    # U_4x4: 16 x 49, V_4x4: 16 x 49, W_4x4: 16 x 49
    
    # For 4x4 * 4x1: Use schoolbook decomposition
    # A(4x4) * B(4x1) = C(4x1)
    # Schoolbook rank: 4*4*1 = 16
    U_4x1 = [[0]*16 for _ in range(16)]
    V_4x1 = [[0]*16 for _ in range(16)]  # m*p = 4*1 = 4? No, V is (m*p) x rank = 4*1 x 16? 
    # Wait, V shape is (m*p, rank). m=4, p=1, so V is 4 x 16.
    # But I wrote 16 rows. Let me fix.
    
    # Actually, for 4x4 * 4x1:
    # n=4, m=4, p=1
    # U: (n*m) x rank = 16 x 16
    # V: (m*p) x rank = 4 x 16
    # W: (p*n) x rank = 1*4 = 4 x 16
    
    U_4x1 = [[0]*16 for _ in range(16)]
    V_4x1 = [[0]*16 for _ in range(4)]
    W_4x1 = [[0]*16 for _ in range(4)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(1):
                U_4x1[i*4+j][col] = 1  # A[i,j]
                V_4x1[j*1+k][col] = 1  # B[j,k]
                W_4x1[k*4+i][col] = 1  # C[k,i] -> W[k*4+i]
                col += 1
    
    # Now combine the three blocks:
    # Block 1: 4x4 * 4x4 (columns 0-3 of B) -> rank 49
    # Block 2: 4x4 * 4x4 (columns 4-7 of B) -> rank 49
    # Block 3: 4x4 * 4x1 (column 8 of B) -> rank 16
    # Total rank: 49 + 49 + 16 = 114
    
    total_rank = 49 + 49 + 16
    
    U_full = [[0]*total_rank for _ in range(16)]
    V_full = [[0]*total_rank for _ in range(36)]  # m*p = 4*9 = 36
    W_full = [[0]*total_rank for _ in range(36)]  # p*n = 9*4 = 36
    
    # Block 1: columns 0-3 of B
    offset = 0
    for i in range(16):
        for c in range(49):
            U_full[i][offset + c] = U_4x4[i][c]
            # V: row j*9+k, for k=0..3 (columns 0-3 of B)
            # U_4x4 has rows i*4+j where j is 0..3 for the 4x4*4x4 case
            # Wait, I need to be more careful with indexing.
    
    # Let me redo this more carefully.
    # For 4x4 * 4x4 block using columns k=0..3 of B:
    # A is 4x4, B_block is 4x4 (columns 0-3), C_block is 4x4
    # U_4x4: rows i*4+j (i=0..3, j=0..3) -> 16 rows
    # V_4x4: rows j*4+k (j=0..3, k=0..3) -> 16 rows
    # W_4x4: rows k*4+i (k=0..3, i=0..3) -> 16 rows
    
    # In the full problem:
    # U: rows i*4+j (i=0..3, j=0..3) -> same
    # V: rows j*9+k (j=0..3, k=0..8) -> 36 rows
    # W: rows k*4+i (k=0..8, i=0..3) -> 36 rows
    
    # Block 1 (k=0..3):
    # V_4x4 row j*4+k maps to V_full row j*9+k
    # W_4x4 row k*4+i maps to W_full row k*4+i
    
    offset = 0
    for c in range(49):
        for i in range(4):
            for j in range(4):
                U_full[i*4+j][offset+c] = U_4x4[i*4+j][c]
        for j in range(4):
            for k in range(4):
                V_full[j*9+k][offset+c] = V_4x4[j*4+k][c]
        for k in range(4):
            for i in range(4):
                W_full[k*4+i][offset+c] = W_4x4[k*4+i][c]
    
    # Block 2 (k=4..7):
    offset = 49
    for c in range(49):
        for i in range(4):
            for j in range(4):
                U_full[i*4+j][offset+c] = U_4x4[i*4+j][c]
        for j in range(4):
            for k in range(4):
                V_full[j*9+(4+k)][offset+c] = V_4x4[j*4+k][c]
        for k in range(4):
            for i in range(4):
                W_full[(4+k)*4+i][offset+c] = W_4x4[k*4+i][c]
    
    # Block 3 (k=8):
    offset = 98
    for c in range(16):
        for i in range(4):
            for j in range(4):
                U_full[i*4+j][offset+c] = U_4x1[i*4+j][c]
        for j in range(4):
            V_full[j*9+8][offset+c] = V_4x1[j][c]
        for i in range(4):
            W_full[8*4+i][offset+c] = W_4x1[i][c]
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
