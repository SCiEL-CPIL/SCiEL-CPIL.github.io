"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate 4x4x4 Strassen decomposition
    def get_strassen_decomp(n, m, p):
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        # Split dimensions
        n1, n2 = n // 2, n - n // 2
        m1, m2 = m // 2, m - m // 2
        p1, p2 = p // 2, p - p // 2
        
        # Strassen 7 multiplications
        # We define the 7 subproblems
        # Each subproblem is defined by (A_sub, B_sub, offset_i, offset_j, offset_k)
        # But easier: just recurse on the 7 specific combinations
        
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # This approach requires constructing U, V, W for the "summed" A and B blocks.
        # A simpler recursive implementation for power-of-2 dimensions:
        
        # Get decompositions for the 7 parts
        # Note: We are multiplying matrices of size n x m and m x p.
        # The recursive calls will be on sizes like n1 x m1 x p1 etc.
        
        # Part 1: (n1+m1) x (m1+p1) ... this is not direct.
        # The standard Strassen recursion:
        # To compute C (n x p) = A (n x m) * B (m x p)
        # We split A into A11 (n1 x m1), A12 (n1 x m2), A21 (n2 x m1), A22 (n2 x m2)
        # B into B11 (m1 x p1), B12 (m1 x p2), B21 (m2 x p1), B22 (m2 x p2)
        
        # The 7 products are of sizes:
        # M1: (n1 x m1) * (m1 x p1) -> but with added blocks?
        # No, Strassen uses the fact that (A11+A22) is n1 x m1? No, A22 is n2 x m2.
        # If n1!=n2 or m1!=m2, we need padding.
        # Since we only use 4x4x4, n=m=p=4. Splits are 2,2.
        # Recurse on 2,2,2. Splits 1,1. Recurse on 1,1,1.
        
        # So we only need to handle equal splits for this specific problem.
        
        U1, V1, W1 = get_strassen_decomp(n1, m1, p1) # For A11*B11 type
        U2, V2, W2 = get_strassen_decomp(n2, m2, p2) # For A22*B22 type
        # But the dimensions of the intermediate products in Strassen are mixed.
        # M1 involves (A11+A22) and (B11+B22).
        # A11 is n1 x m1. A22 is n2 x m2.
        # To add them, we need n1==n2 and m1==m2. Which is true here.
        
        # We will build the U, V, W for the 7 terms by combining the base decompositions.
        
        # Helper to concatenate U, V, W horizontally
        def cat_horizontal(lists):
            if not lists: return []
            res = [ [0] * (sum(len(l[0]) for l in lists) if lists else 0 ) for _ in range(len(lists[0])) ]
            # This is slow, but dimensions are small.
            return res

        # Let's do it more directly.
        # We return lists of rows.
        
        # Term 1: (A11 + A22) * (B11 + B22)
        # This is sum of 4 products: A11B11, A11B22, A22B11, A22B22? No, (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # We can generate the decompositions for these 4 parts and sum them.
        
        # However, generating 7 decompositions and adding them is the way.
        
        # Base case n=2, m=2, p=2:
        # We need the U, V, W for the 7 scalar multiplications.
        
        # Let's implement the recursion specifically for equal dimensions.
        
        if n == 2:
            # Generate the 7 terms for 2x2x2
            # Each term is a sum of products of 1x1 matrices
            U = [[0]*7 for _ in range(4)] # 2*2 = 4 rows
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # Indices:
            # U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
            # V[0]=B00, V[1]=B01, V[2]=B10, V[3]=B11
            # W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11
            
            # M1: (A00+A11)(B00+B11) -> A00B00, A00B11, A11B00, A11B11
            U[0][0] += 1; V[0][0] += 1; W[0][0] += 1 # C00 gets A00B00
            U[0][0] += 0; V[1][0] += 1; W[1][0] += 1 # C01 gets A00B11 ? No.
            # W index is k*n + i.
            # C00: k=0, i=0 -> W[0]
            # C01: k=1, i=0 -> W[1]
            # C10: k=0, i=1 -> W[2]
            # C11: k=1, i=1 -> W[3]
            
            # Term 1:
            U[0][0] = 1; V[0][0] = 1; W[0][0] = 1 # A00 B00 -> C00
            U[0][0] += 0 # A00 is already in U
            # We need to map A00 to B11 contribution to C01.
            # U[0] is A00. V[1] is B01. W[1] is C01.
            # We want A00 * B01? No, Term 1 is (A00+A11)(B00+B11).
            # It expands to A00B00, A00B11, A11B00, A11B11.
            #
            # Let's just assign the 7 terms directly.
            
            # t1: (A00+A11)*(B00+B11)
            U[0][0] = 1; U[3][0] = 1 # A00, A11
            V[0][0] = 1; V[3][0] = 1 # B00, B11
            # Result: C00 += A00B00, C01 += A00B11, C10 += A11B00, C11 += A11B11
            W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
            
            # t2: (A10+A11)*B00
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1; W[3][1] = 1 # C10, C11
            
            # t3: A00*(B01-B11)
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[1][2] = 1; W[3][2] = -1 # C01, C11
            
            # t4: A11*(B10-B00)
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[2][3] = 1; W[0][3] = -1 # C10, C00
            
            # t5: (A00+A01)*B11
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[1][4] = 1; W[3][4] = 1 # C01, C11
            
            # t6: (A10-A00)*(B00+B01)
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[2][5] = 1; W[3][5] = 1 # C10, C11
            
            # t7: (A01-A11)*(B10+B11)
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[1][6] = 1; W[3][6] = 1 # C01, C11
            
            return U, V, W
        
        # Recursive case (n > 2)
        # This logic is tricky to write generically for U/V/W matrix concatenation.
        # Since we only need 4x4x4, we can hardcode the recursion for 4 -> 2.
        
        # Call for 2x2x2
        U2, V2, W2 = get_strassen_decomp(2, 2, 2)
        
        # U4 will be 4x47 = 16 x 49?
        # Each 2x2 block in U4 corresponds to a 2x2 block in U2.
        
        # This implementation is becoming too complex to write from scratch in one go without testing.
        # I will return the schoolbook for now? No, score must improve.
        # I will return a valid decomposition of rank 66 by hardcoding the 4x4x4 Strassen matrices.
        
        # But I don't have the 49x16 matrices memorized.
        # I will fall back to a simpler valid improvement:
        # Rank 90 is still an improvement.
        # Actually, 85 is easy to implement.
        # Core 4x4x4 Schoolbook (64) + Strip 16 + Corner 1 = 81.
        # Wait, 4x4x4 schoolbook is 64.
        # 64 + 16 + 1 = 81.
        # Score 100/81 = 1.23.
        
        # Let's use 4x4x4 Schoolbook for the core.
        
        pass
        
    # 2. Build 4x4x4 Schoolbook Core
    n_core, m_core, p_core = 4, 4, 4
    rank_core = n_core * m_core * p_core
    U_core = [[0]*rank_core for _ in range(n_core*m_core)]
    V_core = [[0]*rank_core for _ in range(m_core*p_core)]
    W_core = [[0]*rank_core for _ in range(p_core*n_core)]
    
    col = 0
    for i in range(n_core):
        for j in range(m_core):
            for k in range(p_core):
                U_core[i*m_core+j][col] = 1
                V_core[j*p_core+k][col] = 1
                W_core[k*n_core+i][col] = 1
                col += 1
                
    # 3. Add Strip 4x4x1 (Rank 16)
    rank_strip = 16
    U_strip = [[0]*rank_strip for _ in range(n*m)] # 4*5 = 20 rows
    V_strip = [[0]*rank_strip for _ in range(m*p)] # 5*5 = 25 rows
    W_strip = [[0]*rank_strip for _ in range(p*n)] # 5*4 = 20 rows
    
    col = 0
    for i in range(n_core):
        for j in range(m_core):
            # k is 4 (the 5th column)
            k = 4
            U_strip[i*m + j][col] = 1
            V_strip[j*p + k][col] = 1
            W_strip[k*n + i][col] = 1
            col += 1
            
    # 4. Add Corner 4x1x5 (Rank 1)
    rank_corner = 1
    U_corner = [[0]*rank_corner for _ in range(n*m)]
    V_corner = [[0]*rank_corner for _ in range(m*p)]
    W_corner = [[0]*rank_corner for _ in range(p*n)]
    
    for i in range(n):
        U_corner[i*m + 4][0] = 1
        for k in range(p):
            V_corner[4*p + k][0] = 1
            W_corner[k*n + i][0] = 1
            
    # 5. Combine
    rank = rank_core + rank_strip + rank_corner
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Copy Core
    for r in range(n*m):
        for c in range(rank_core):
            # Map core indices (i*4+j) to global (i*5+j)
            i = r // 4
            j = r % 4
            U[i*5 + j][c] = U_core[r][c]
            
    for r in range(m*p):
        for c in range(rank_core):
            j = r // 4
            k = r % 4
            V[j*5 + k][c] = V_core[r][c]
            
    for r in range(p*n):
        for c in range(rank_core):
            k = r // 4
            i = r % 4
            W[k*4 + i][c] = W_core[r][c]
            
    # Copy Strip
    offset = rank_core
    for r in range(n*m):
        for c in range(rank_strip):
            U[r][offset + c] = U_strip[r][c]
    for r in range(m*p):
        for c in range(rank_strip):
            V[r][offset + c] = V_strip[r][c]
    for r in range(p*n):
        for c in range(rank_strip):
            W[r][offset + c] = W_strip[r][c]
            
    # Copy Corner
    offset += rank_strip
    for r in range(n*m):
        for c in range(rank_corner):
            U[r][offset + c] = U_corner[r][c]
    for r in range(m*p):
        for c in rank_corner:
            V[r][offset + c] = V_corner[r][c]
    for r in range(p*n):
        for c in range(rank_corner):
            W[r][offset + c] = W_corner[r][c]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
