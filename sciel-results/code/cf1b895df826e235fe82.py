"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    U_list = []
    V_list = []
    W_list = []

    def add_term(u, v, w):
        """u: dict (i,j)->coeff, v: dict (j,k)->coeff, w: dict (k,i)->coeff"""
        for (i, j), uc in u.items():
            U_list.append((i * m + j, uc))
        for (j, k), vc in v.items():
            V_list.append((j * p + k, vc))
        for (k, i), wc in w.items():
            W_list.append((k * n + i, wc))

    # We want to multiply A (3x4) by B (4x8) to get C (3x8).
    # C[i,k] = sum_j A[i,j] B[j,k].

    # Block 1: i in {0, 1}, j in {0, 1}, k in {0, 1} (2x2x2)
    # Using Strassen for 2x2 matrices.
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], etc.
    # But here j only goes 0..1 for this block.
    # Wait, m=4. So j in {0,1} is half.
    # Let's use Strassen on the 2x2 block of indices (i in {0,1}, j in {0,1}, k in {0,1}).
    # This contributes to C[0:2, 0:2].
    # A[i,j] is 2x2, B[j,k] is 2x2.
    # C[i,k] = A11 B11 + A12 B21? No.
    # C[0:2, 0:2] = A[0:2, 0:2] B[0:2, 0:2] + A[0:2, 2:4] B[2:4, 0:2].
    # Let A_L = A[0:2, 0:2], A_R = A[0:2, 2:4].
    # Let B_T = B[0:2, 0:2], B_B = B[2:4, 0:2].
    # C_block = A_L B_T + A_R B_B.
    # This is not a standard 2x2x2 multiplication of two 2x2 matrices.
    # It is (2x2)(2x2) + (2x2)(2x2).
    # We can view this as multiplying a 2x4 matrix [A_L A_R] by a 4x2 matrix [B_T; B_B].
    # Rank of 2x4x2 multiplication is 2*4*2 = 16 schoolbook.
    # Is there a faster way?
    # Actually, we can just use schoolbook for each 2x2 block.
    # Let's try to be clever with the whole 3x4x8.
    # Maybe just use Strassen on 2x2 blocks where possible.
    # For i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}.
    # This is 2x4x8.
    # We can split k into two 4s.
    # C[0:2, 0:4] = A[0:2, :] B[:, 0:4].
    # This is 2x4x4. Rank 2*4*4 = 32 schoolbook.
    # Can we do better?
    # Let's stick to a simpler reduction first to ensure correctness.
    # Use Strassen for 2x2x2 subproblems.
    # There are 3*2*4 = 24 such 2x2x2 subproblems?
    # i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}.
    # We can partition i into {0,1} and {2}.
    # For i in {0,1}, we have 2x4x8.
    # Partition j into {0,1} and {2,3}.
    # Partition k into {0,1,2,3} and {4,5,6,7}.
    # This gives 4 blocks of 2x2x4.
    # 2x2x4 multiplication: C[2x4] = A[2x2] B[2x4].
    # Rank 2*2*4 = 16 schoolbook.
    # We can use Strassen-like on 2x2 part?
    # C[:, 0:4] = A[:,0:2] B[0:2, 0:4] + A[:,2:4] B[2:4, 0:4].
    # Still two 2x2x4 multiplications.
    
    # Let's just implement a generic rank-reduction using known algorithms for small matrices.
    # Actually, the simplest improvement is Strassen on 2x2 blocks.
    # Let's iterate over all 2x2 blocks of i and j.
    # i in {0,1,2}. j in {0,1,2,3}.
    # We can pair i={0,1} and j={0,1}, i={0,1} and j={2,3}, i={2} and j={0,1}, i={2} and j={2,3}.
    # For each (i_pair, j_pair), we have a 2x2 A block.
    # We need to multiply by B (2x8) to get C (2x8).
    # This is 2x2x8.
    # We can use Strassen if we split k into 2x2 blocks?
    # 2x2x8 = 4 * (2x2x2).
    # Each 2x2x2 can be done in 7 rank terms.
    # So 2x2x8 can be done in 4 * 7 = 28 rank terms.
    # Schoolbook is 2*2*8 = 32.
    # Saving 4 per block.
    # Number of (i_pair, j_pair) blocks: 2 * 2 = 4.
    # Total saving 16.
    # New rank 96 - 16 = 80.

    # Let's implement this.
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    def strassen_term(i0, i1, j0, j1, k0, k1, sign):
        # This adds 7 rank terms for the 2x2x2 block at i0,i1; j0,j1; k0,k1.
        # A is i x j, B is j x k.
        # We need to map these to global indices.
        pass

    # This is getting complicated to manage signs and indices.
    # Let's write a helper to add a scalar multiplication term.
    terms = []
    
    # Schoolbook baseline
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       add term
    
    # Instead, let's construct the rank 80 decomposition.
    # We will use schoolbook for most, and Strassen for the 2x2x2 blocks in the i={0,1} range.
    
    # Define Strassen combinations for 2x2 matrices.
    # We will process blocks of size 2x2x2.
    # i indices: (0,1), (2)
    # j indices: (0,1), (2,3)
    # k indices: (0,1), (2,3), (4,5), (6,7)
    
    # For i_pair (0,1), we use Strassen.
    # For i=2, we use schoolbook.
    
    # Let's just implement the Strassen reduction for all 2x2x2 sub-tensors where i has 2 elements.
    # There are 2 pairs of i (0,1) and... wait, i only has 3. So only (0,1) can be paired.
    # j has 4, so (0,1) and (2,3).
    # k has 8, so (0,1), (2,3), (4,5), (6,7).
    
    # So we have 2 (j pairs) * 4 (k pairs) = 8 blocks of 2x2x2.
    # Each block saves 1 rank term (8 -> 7).
    # Total saving 8.
    # Rank 96 - 8 = 88.
    
    # Is there more?
    # What about i=2? It's a 1x4x8 multiplication.
    # C[2, k] = sum_j A[2, j] B[j, k].
    # This is 1x4x8. Rank 32.
    # Can we do better?
    # 1x4x8 is equivalent to 4 parallel 1x1x8 dot products? No.
    # It's a linear combination of rows of B.
    # The rank of the tensor T_{j,k} is 4*8 = 32.
    # It is known that rank(1 x m x p) = m*p.
    # So no saving there.
    
    # So total rank 88.
    # Let's verify the Strassen part.
    # We replace 8 schoolbook terms with 7 Strassen terms.
    # We need to carefully construct the U, V, W vectors.
    
    U = {}
    V = {}
    W = {}
    
    def add_rank_term(u, v, w):
        # u, v, w are dicts {index: coeff}
        # We store them in lists to later convert to matrices.
        terms.append((u, v, w))
        
    # Strassen coefficients
    # M1: (A00+A11)(B00+B11)
    # U: A00, A11. V: B00, B11. W: C00, C11.
    # Wait, Strassen uses intermediate products.
    # M1 = (A00+A11)(B00+B11).
    # This is a sum of 4 terms if expanded, but in tensor decomposition,
    # (A00+A11) is a new "matrix" in the U factor.
    # So U has entries for A00 and A11 with coeff 1.
    # V has entries for B00 and B11 with coeff 1.
    # W has entries for C00 and C11 with coeff 1 (partially).
    
    # Let's define the 7 Strassen multiplications for a 2x2 block.
    # Indices: i0, i1 (rows of A), j0, j1 (cols of A, rows of B), k0, k1 (cols of B).
    # A00=A[i0,j0], A01=A[i0,j1], A10=A[i1,j0], A11=A[i1,j1].
    # B00=B[j0,k0], B01=B[j0,k1], B10=B[j1,k0], B11=B[j1,k1].
    # C00=C[i0,k0], C01=C[i0,k1], C10=C[i1,k0], C11=C[i1,k1].
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M5 + M7 + M4
    # C01 = M5 + M3
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        # M1
        u = {(i0,j0):1, (i1,j1):1}
        v = {(j0,k0):1, (j1,k1):1}
        w = {(k0,i0):1, (k1,i1):1} # C00 and C11 get +1
        # We need to accumulate W contributions.
        # It's better to define the 7 terms and then sum their W contributions.
        
        terms_M = []
        # M1
        terms_M.append((
            {(i0,j0):1, (i1,j1):1},
            {(j0,k0):1, (j1,k1):1},
            {(k0,i0):1, (k1,i1):1}
        ))
        # M2
        terms_M.append((
            {(i1,j0):1, (i1,j1):1},
            {(j0,k0):1},
            {(k0,i1):1}
        ))
        # M3
        terms_M.append((
            {(i0,j0):1},
            {(j1,k0):1, (j1,k1):-1},
            {(k1,i0):1, (k0,i1):1} # C01 gets M3, C10? No.
            # C01 = M5+M3. C10 = M2+M4.
            # M3 contributes to C01.
            # W index is (k, i).
            # C01 is i=0, k=1. So (1,0).
            # Wait, W[k*n+i].
            # C[i,k] is at index (k, i).
            # So C01 is (k1, i0).
            # M3 = A00(B10-B11).
            # A00 is (i0,j0). B10 is (j1,k0), B11 is (j1,k1).
            # Product is A00 * (B10 - B11).
            # This contributes to C[i0, k0] and C[i0, k1]?
            # No, A00 is scalar. B10 is scalar.
            # M3 is a scalar product of two linear combos?
            # No, in Strassen, A00, B10 etc are scalars (elements of the 2x2 blocks).
            # So M3 is a scalar.
            # It contributes to C01?
            # Formula: C01 = M5 + M3.
            # So M3 contributes to C[i0, k1]. Index (k1, i0).
            # My term above had (k1,i0):1 and (k0,i1):1. That's wrong.
            # Let's re-verify Strassen formulas.
            
        ))
        # Let's look up standard Strassen.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A22-A21)(B11+B21)
        # M7 = (A12-A22)(B12+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Mapping:
        # i0=1, i1=2. j0=1, j1=2. k0=1, k1=2.
        # A11 -> (i0,j0). A12 -> (i0,j1). A21 -> (i1,j0). A22 -> (i1,j1).
        # B11 -> (j0,k0). B12 -> (j0,k1). B21 -> (j1,k0). B22 -> (j1,k1).
        # C11 -> (k0,i0). C12 -> (k1,i0). C21 -> (k0,i1). C22 -> (k1,i1).
        
        # M1: (A11+A22)(B11+B22).
        # U: (i0,j0):1, (i1,j1):1.
        # V: (j0,k0):1, (j1,k1):1.
        # W: contributes to C11 and C22?
        # C11 has +M1. C22 has +M1.
        # So W: (k0,i0):1, (k1,i1):1.
        
        # M2: (A21+A22)B11.
        # U: (i1,j0):1, (i1,j1):1.
        # V: (j0,k0):1.
        # W: C21 has +M2.
        # So W: (k0,i1):1.
        
        # M3: A11(B21-B22).
        # U: (i0,j0):1.
        # V: (j1,k0):1, (j1,k1):-1.
        # W: C12 has +M3.
        # So W: (k1,i0):1.
        
        # M4: A22(B12-B11).
        # U: (i1,j1):1.
        # V: (j0,k1):1, (j0,k0):-1.
        # W: C11 has +M4. C21 has +M4.
        # So W: (k0,i0):1, (k0,i1):1.
        
        # M5: (A11+A12)B22.
        # U: (i0,j0):1, (i0,j1):1.
        # V: (j1,k1):1.
        # W: C11 has -M5. C12 has +M5.
        # So W: (k0,i0):-1, (k1,i0):1.
        
        # M6: (A22-A21)(B11+B21).
        # U: (i1,j1):1, (i1,j0):-1.
        # V: (j0,k0):1, (j1,k0):1.
        # W: C22 has +M6.
        # So W: (k1,i1):1.
        
        # M7: (A12-A22)(B12+B22).
        # U: (i0,j1):1, (i1,j1):-1.
        # V: (j0,k1):1, (j1,k1):1.
        # W: C11 has +M7.
        # So W: (k0,i0):1.
        
        # Now sum W contributions:
        # C11 (k0,i0): M1(+1) + M4(+1) + M5(-1) + M7(+1) = 2.
        # Wait. C11 = M1 + M4 - M5 + M7.
        # Sum: 1 + 1 - 1 + 1 = 2.
        # This is correct.
        
        # C12 (k1,i0): M3(+1) + M5(+1) = 2.
        # C21 (k0,i1): M2(+1) + M4(+1) = 2.
        # C22 (k1,i1): M1(+1) + M6(+1) = 2.
        
        # Wait, is it 2?
        # Yes, because the diagonal entries of C are sums of 4 terms.
        # But in the tensor decomposition, each term is a rank 1 tensor.
        # The sum of these 7 rank 1 tensors equals the multiplication tensor for the 2x2 block.
        
        # Let's add these 7 terms to the global list.
        terms.append((
            {(i0,j0):1, (i1,j1):1},
            {(j0,k0):1, (j1,k1):1},
            {(k0,i0):1, (k1,i1):1}
        ))
        terms.append((
            {(i1,j0):1, (i1,j1):1},
            {(j0,k0):1},
            {(k0,i1):1}
        ))
        terms.append((
            {(i0,j0):1},
            {(j1,k0):1, (j1,k1):-1},
            {(k1,i0):1}
        ))
        terms.append((
            {(i1,j1):1},
            {(j0,k1):1, (j0,k0):-1},
            {(k0,i0):1, (k0,i1):1}
        ))
        terms.append((
            {(i0,j0):1, (i0,j1):1},
            {(j1,k1):1},
            {(k0,i0):-1, (k1,i0):1}
        ))
        terms.append((
            {(i1,j1):1, (i1,j0):-1},
            {(j0,k0):1, (j1,k0):1},
            {(k1,i1):1}
        ))
        terms.append((
            {(i0,j1):1, (i1,j1):-1},
            {(j0,k1):1, (j1,k1):1},
            {(k0,i0):1}
        ))

    # Apply Strassen to i in {0,1}.
    # j pairs: (0,1), (2,3).
    # k pairs: (0,1), (2,3), (4,5), (6,7).
    
    for jp in [(0,1), (2,3)]:
        for kp in [(0,1), (2,3), (4,5), (6,7)]:
            strassen_2x2x2(0, 1, jp[0], jp[1], kp[0], kp[1])
            
    # For i=2, use schoolbook.
    for j in range(m):
        for k in range(p):
            terms.append((
                {(2,j):1},
                {(j,k):1},
                {(k,2):1}
            ))
            
    # Convert terms to matrices
    rank = len(terms)
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, coeff in u.items():
            U_mat[idx[0]*m + idx[1]][r] += coeff
        for idx, coeff in v.items():
            V_mat[idx[0]*p + idx[1]][r] += coeff
        for idx, coeff in w.items():
            W_mat[idx[0]*n + idx[1]][r] += coeff
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
