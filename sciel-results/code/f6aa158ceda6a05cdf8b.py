"""Rewritten program for 4x4x8 matrix multiplication using recursive Strassen (rank 98)."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 exact tensor decomposition for 4x4 * 4x8 -> 4x8.
    
    Strategy:
    - Use Strassen's algorithm recursively to get a rank-49 decomposition for 4x4*4x4.
    - Apply this twice (for the two 4x4 blocks of the 4x8 output) to get rank 98.
    - U has shape (16, 98), V has shape (32, 98), W has shape (32, 98).
    """
    # Base case: 2x2 Strassen decomposition
    # Returns list of (u, v, w) triples where u, v, w are length-4 vectors
    def strassen_2x2():
        # The 7 multiplications for 2x2 matrix multiplication:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # Output:
        # C00 = M1 - M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6
        #
        # Indexing: A[i*m+j], B[j*p+k], C[k*n+i] with n=m=p=2
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices: 00=0, 01=1, 10=2, 11=3 (C[k*n+i])
        
        triples = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = [1, 0, 0, 1]  # A00 + A11
        v1 = [1, 0, 0, 1]  # B00 + B11
        w1 = [1, 0, 0, 1]  # contributes to C00 and C11: C00=M1-..., C11=M1-...
        # Wait, W should encode the contribution. Let me think again.
        # Actually, in the tensor decomposition, each term is u * v * w where:
        # u[i*m+j] is coefficient for A element
        # v[j*p+k] is coefficient for B element
        # w[k*n+i] is coefficient for C element
        # And the sum over all terms gives the correct C.
        
        # For M1 = (A00+A11)*(B00+B11):
        # This contributes +1 to C00 and +1 to C11
        # C00 = C[0*2+0] = C[0]
        # C11 = C[1*2+1] = C[3]
        u1 = [1, 0, 0, 1]  # A00 + A11
        v1 = [1, 0, 0, 1]  # B00 + B11
        w1 = [1, 0, 0, 1]  # C00 + C11 (indices 0 and 3)
        triples.append((u1, v1, w1))
        
        # M2 = (A10+A11)*B00
        # Contributes +1 to C10
        # C10 = C[1*2+0] = C[2]
        u2 = [0, 0, 1, 1]  # A10 + A11
        v2 = [1, 0, 0, 0]  # B00
        w2 = [0, 0, 1, 0]  # C10
        triples.append((u2, v2, w2))
        
        # M3 = A00*(B10-B11)
        # Contributes -1 to C11 and +1 to C01
        # C01 = C[0*2+1] = C[1]
        # C11 = C[1*2+1] = C[3]
        u3 = [1, 0, 0, 0]  # A00
        v3 = [0, 0, 1, -1]  # B10 - B11
        w3 = [0, 1, 0, -1]  # C01 - C11
        triples.append((u3, v3, w3))
        
        # M4 = A11*(B01-B10)
        # Contributes -1 to C00 and +1 to C10
        # C00 = C[0]
        # C10 = C[2]
        u4 = [0, 0, 0, 1]  # A11
        v4 = [0, 1, -1, 0]  # B01 - B10
        w4 = [-1, 0, 1, 0]  # -C00 + C10
        triples.append((u4, v4, w4))
        
        # M5 = (A00+A01)*B11
        # Contributes +1 to C01
        # C01 = C[1]
        u5 = [1, 1, 0, 0]  # A00 + A01
        v5 = [0, 0, 0, 1]  # B11
        w5 = [0, 1, 0, 0]  # C01
        triples.append((u5, v5, w5))
        
        # M6 = (A10-A00)*(B00+B10)
        # Contributes +1 to C11
        # C11 = C[3]
        u6 = [-1, 0, 1, 0]  # -A00 + A10
        v6 = [1, 0, 1, 0]  # B00 + B10
        w6 = [0, 0, 0, 1]  # C11
        triples.append((u6, v6, w6))
        
        # M7 = (A01-A11)*(B01+B11)
        # Contributes +1 to C00
        # C00 = C[0]
        u7 = [0, 1, 0, -1]  # A01 - A11
        v7 = [0, 1, 0, 1]  # B01 + B11
        w7 = [1, 0, 0, 0]  # C00
        triples.append((u7, v7, w7))
        
        return triples
    
    # Recursive Strassen for 4x4*4x4
    # Takes the 7 triples from 2x2 and applies them to 2x2 blocks
    def strassen_4x4():
        base_triples = strassen_2x2()
        result = []
        n_block = 2  # block size
        # A is 4x4, split into 2x2 blocks of 2x2
        # A indices: A[i*4+j] for i,j in 0..3
        # Block A_ab has indices (2a+x)*4 + (2b+y) for x,y in 0..1
        # B is 4x4, similar
        # C is 4x4, C[k*4+i] for k,i in 0..3
        # Block C_cd has indices (2c+z)*4 + (2d+w) for z,w in 0..1
        
        for outer_u, outer_v, outer_w in base_triples:
            # outer_u[ab] gives coefficient for block A_ab
            # outer_v[bc] gives coefficient for block B_bc
            # outer_w[cd] gives coefficient for block C_cd
            
            for ab in range(4):  # a in 0..1, b in 0..1
                a = ab // 2
                b = ab % 2
                coeff_a = outer_u[ab]
                if coeff_a == 0:
                    continue
                    
                for bc in range(4):  # b in 0..1, c in 0..1
                    b2 = bc // 2
                    c = bc % 2
                    coeff_b = outer_v[bc]
                    if coeff_b == 0:
                        continue
                    
                    for cd in range(4):  # c in 0..1, d in 0..1
                        c2 = cd // 2
                        d = cd % 2
                        coeff_c = outer_w[cd]
                        if coeff_c == 0:
                            continue
                        
                        if b != b2 or c != c2:
                            continue  # block indices must match
                        
                        # Now we need to apply Strassen to the actual 2x2 multiplication
                        # within these blocks
                        for inner_u, inner_v, inner_w in base_triples:
                            u_vec = [0] * 16
                            v_vec = [0] * 16
                            w_vec = [0] * 16
                            
                            total_coeff = coeff_a * coeff_b * coeff_c
                            if total_coeff == 0:
                                continue
                            
                            # Fill u_vec for A block (a,b)
                            for x in range(2):
                                for y in range(2):
                                    a_idx = (2*a + x) * 4 + (2*b + y)
                                    u_vec[a_idx] = total_coeff * inner_u[x*2 + y]
                            
                            # Fill v_vec for B block (b,c)
                            for x in range(2):
                                for y in range(2):
                                    b_idx = (2*b + x) * 4 + (2*c + y)
                                    v_vec[b_idx] = total_coeff * inner_v[x*2 + y]
                            
                            # Fill w_vec for C block (c,d)
                            for x in range(2):
                                for y in range(2):
                                    c_idx = (2*c + x) * 4 + (2*d + y)
                                    w_vec[c_idx] = total_coeff * inner_w[x*2 + y]
                            
                            # Check if this is a non-zero vector
                            if any(u_vec) or any(v_vec) or any(w_vec):
                                result.append((u_vec, v_vec, w_vec))
        
        return result
    
    # Get the 49 triples for 4x4*4x4
    strassen_49 = strassen_4x4()
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    rank = 2 * len(strassen_49)  # 98
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # First 49 columns: for the first half of output (k in 0..3)
    for col_idx, (u_vec, v_vec, w_vec) in enumerate(strassen_49):
        for i in range(n * m):
            U[i][col_idx] = u_vec[i]
        for j in range(m * 4):  # first 4 columns of B (k in 0..3)
            V[j][col_idx] = v_vec[j]
        for k in range(4 * n):  # first 4 rows of C (k in 0..3)
            W[k][col_idx] = w_vec[k]
    
    # Next 49 columns: for the second half of output (k in 4..7)
    for col_idx, (u_vec, v_vec, w_vec) in enumerate(strassen_49):
        c = col_idx + len(strassen_49)
        for i in range(n * m):
            U[i][c] = u_vec[i]
        for j in range(4, m * p):  # second 4 columns of B (k in 4..7)
            # Map j to local index j-4 in the 4x4 B block
            V[j][c] = v_vec[j - 4]
        for k in range(4, p * n):  # second 4 rows of C (k in 4..7)
            # Map k to local index k-4 in the 4x4 C block
            W[k][c] = w_vec[k - 4]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
