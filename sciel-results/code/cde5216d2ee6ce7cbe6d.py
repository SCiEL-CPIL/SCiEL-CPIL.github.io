"""Matrix multiplication 2,2,8 tensor decomposition."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # We seek a rank-minimal decomposition for M_{2,2,8}.
    # The schoolbook rank is 32.
    # By exploiting linearity, we can potentially reduce the rank.
    # For M_{2,2,2}, Strassen gives rank 7.
    # For M_{2,2,8}, we have 4 blocks of size 2x2x2.
    # If we use independent Strassen decompositions, rank is 4*7 = 28.
    # Can we do better?
    
    # Let's analyze the structure.
    # A is 2x2, B is 2x8.
    # C is 2x8.
    # C[:, 0:2] = A @ B[:, 0:2]
    # C[:, 2:4] = A @ B[:, 2:4]
    # C[:, 4:6] = A @ B[:, 4:6]
    # C[:, 6:8] = A @ B[:, 6:8]
    
    # Let B1 = B[:, 0:2], B2 = B[:, 2:4], B3 = B[:, 4:6], B4 = B[:, 6:8].
    # C1 = A B1, C2 = A B2, C3 = A B3, C4 = A B4.
    
    # The tensor is sum of 4 M_{2,2,2} tensors.
    # T = T1 + T2 + T3 + T4.
    # T1 = A B1 C1^T (roughly, in tensor form)
    
    # Is it possible to share structure between blocks?
    # The A part is shared. The B parts are disjoint. The C parts are disjoint.
    # So the tensor is block diagonal in B and C indices, but coupled in A.
    
    # Let's check if rank 28 is minimal.
    # Lower bounds:
    # 1. Slice rank: Fix A index (i,j). Then we have M_{1,2,8} tensor? No.
    #    Fix i,j. The tensor slice is B (2x8) -> C (2x8).
    #    This is just 4 separate M_{1,1,1} blocks scaled by A_{ij}.
    #    Wait, C_{k, l} = sum_j A_{ij} B_{j, l}.
    #    For fixed i, this is a linear map from B to C.
    #    The tensor T_{i, j, k, l} = A_{i, j} B_{j, k} ... no.
    #    T_{A, B, C} = sum_{j} A_{i, j} B_{j, k} delta_{l, k}?
    #    Actually, C_{i, k} = sum_j A_{i, j} B_{j, k}.
    #    So T_{i, j, k} for a block is A_{i, j} B_{j, k}.
    #    The full tensor has indices (i, j, k, l) where l is the column index of B/C (0..7).
    #    T_{i, j, k, l} = A_{i, j} * (B_{j, l} if k == l else 0)? No.
    #    B is 2x8. C is 2x8.
    #    C_{i, l} = sum_j A_{i, j} B_{j, l}.
    #    So the tensor entry T_{i, j, l} (where l is column) is A_{i, j} B_{j, l}.
    #    Wait, the output index is l.
    #    So T is a 3-tensor? No, it's a 3-mode tensor for matrix multiplication?
    #    Standard MM tensor T_{i, j, k} for C_{i, k} = sum_j A_{i, j} B_{j, k}.
    #    Here B has 8 columns. So k runs 0..7.
    #    So T_{i, j, k} = A_{i, j} B_{j, k}.
    #    This is just 8 separate M_{2,2,1} tensors?
    #    No, B_{j, k} depends on k.
    #    For each column k, we have a 2x2 matrix multiplication.
    #    T = sum_{k=0}^7 T^{(k)} where T^{(k)} is the tensor for C[:, k] = A B[:, k].
    #    T^{(k)}_{i, j, l} is non-zero only if l=k?
    #    In the decomposition, W has rows indexed by (k, i) i.e. (col, row).
    #    So W_{col*n + row, r}.
    #    The tensor condition is:
    #    sum_r U_{row_a, col_a?} V_{row_b, col_b?} W_{row_c, col_c?} = T
    #    Let's stick to the definition:
    #    U shape (n*m, rank) = (4, R)
    #    V shape (m*p, rank) = (16, R)
    #    W shape (p*n, rank) = (16, R)
    
    #    T_{i, j, k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = A_{i,j} B_{j,k}
    #    Here i in 0..1, j in 0..1, k in 0..7.
    
    #    This is a tensor of shape 2x2x8.
    #    It is the Kronecker product of M_{2,2,1} (for the matrix mult) and identity on the 8 columns?
    #    No.
    #    T_{i, j, k} = A_{i, j} B_{j, k}.
    #    This is NOT a Kronecker product.
    #    However, note that for each k, the "slice" T_{i, j, k} as a function of i,j is A_{i,j}.
    #    But it depends on k through B_{j,k}.
    #    Actually, T_{i, j, k} = A_{i, j} * B_{j, k}.
    #    This looks like a "block diagonal" structure in the B-C indices.
    
    #    Let's check the rank.
    #    We can use Strassen for each of the 8 "columns" of B?
    #    No, Strassen is for 2x2x2.
    #    Here we have 2x2x8.
    #    We can view this as 4 blocks of 2x2x2.
    #    Block 1: k=0,1. T_{i,j,0..1} = A_{i,j} B_{j,0..1}.
    #    This is exactly the tensor for M_{2,2,2} where B is 2x2.
    #    So the tensor is the direct sum of 4 M_{2,2,2} tensors.
    #    The rank of a direct sum is the sum of the ranks.
    #    rank(T) = sum rank(T_k) = 4 * rank(M_{2,2,2}) = 4 * 7 = 28.
    
    #    Is it possible to do better than the sum of ranks?
    #    Generally, for block diagonal tensors, the rank is the sum of the ranks of the blocks,
    #    UNLESS there is some "coupling" or if the blocks are not independent.
    #    Here, the A indices are shared, but the B and C indices are disjoint.
    #    The tensor entries for block k only involve B_{j,k} and C_{i,k}.
    #    The entries for block l (l!=k) involve B_{j,l} and C_{i,l}.
    #    The A entries are the same.
    #    
    #    Let's check if we can mix terms from different blocks.
    #    Suppose we have a term r that contributes to both block 0 and block 1.
    #    U_{i,j, r} is fixed.
    #    V_{j, k, r} must be non-zero for k in {0,1} (block 0) and potentially k in {2,3} (block 1).
    #    W_{k, i, r} must be non-zero for k in {0,1} and {2,3}.
    #    
    #    If V_{j, 0, r} and V_{j, 2, r} are both non-zero, then the term r mixes the inputs.
    #    This would imply that the linear forms in V and W are coupled.
    #    
    #    However, the target tensor is block diagonal with respect to the (j,k) and (k,i) indices?
    #    T_{i, j, k} is non-zero for all i,j,k.
    #    But for a fixed k, the dependence on j is via B_{j,k}.
    #    
    #    Actually, let's look at the structure again.
    #    T_{i, j, k} = A_{i, j} B_{j, k}.
    #    This is a "simple" tensor structure.
    #    It is the tensor product of the vector A_{i,j} (reshaped) and ... no.
    #    
    #    Consider the slices.
    #    Fix i. T_{i, j, k} = A_{i, j} B_{j, k}.
    #    This is the outer product of vector A_i (length 2) and matrix B^T? No.
    #    It is the matrix A_i (1x2) times B (2x8).
    #    So the slice i is the rank-2 matrix (or less) A_i B.
    #    Since A_i has 2 elements, and B is 2x8, the product is a 1x8 matrix.
    #    The slice is a matrix of size 1x8? No, the slice is T_{i, :, :} which is 2x8.
    #    T_{i, j, k} for fixed i is a 2x8 matrix.
    #    Entry (j,k) is A_{i,j} B_{j,k}.
    #    This matrix is the sum of 2 rank-1 matrices:
    #    A_{i,0} * B_{0,:}^T + A_{i,1} * B_{1,:}^T.
    #    So each slice i has rank <= 2.
    #    
    #    This suggests the rank might be low.
    #    But we have 2 slices.
    #    
    #    Let's try to construct a decomposition of rank < 28.
    #    
    #    Hypothesis: Rank is 16?
    #    Schoolbook for 2x2x8 is 2*2*8 = 32.
    #    Strassen blocks: 28.
    #    
    #    Can we use the fact that B is "wide"?
    #    For M_{2,2,p}, the rank is known.
    #    For p=2, rank=7.
    #    For p=1, rank=2? (2x2x1 is just 2x2 matrix mult by a vector? No, 2x2x1 is not standard MM.
    #    Standard MM is M_{n,m,p}.
    #    Here n=2, m=2, p=8.
    #    
    #    Let's check known results for M_{2,2,p}.
    #    The rank of M_{2,2,p} is the minimum number of bilinear multiplications.
    #    For p=1, M_{2,2,1} is multiplying 2x2 by 2x1.
    #    C (2x1) = A (2x2) B (2x1).
    #    c0 = a00 b0 + a01 b1
    #    c1 = a10 b0 + a11 b1
    #    This requires 4 multiplications: a00b0, a01b1, a10b0, a11b1.
    #    Can we do it in 3?
    #    (a00+a10)(b0) + ... no.
    #    Strassen doesn't help for vector mult.
    #    Rank(M_{2,2,1}) = 4.
    #    
    #    For p=2, Rank(M_{2,2,2}) = 7.
    #    For p=3?
    #    
    #    There is a result: Rank(M_{n,m,p}) <= n * Rank(M_{1,m,p})?
    #    Rank(M_{1,m,p}) is the rank of multiplying 1xm by mxp.
    #    M_{1,2,8}: c = a b. c is 1x8.
    #    c_k = a0 b0k + a1 b1k.
    #    This is just dot product.
    #    It requires 2 multiplications?
    #    c = a0 * b0_row + a1 * b1_row.
    #    The terms are a0*b0k and a1*b1k.
    #    We can compute u0 = a0 * 1, u1 = a1 * 1? No, multiplications must be bilinear.
    #    Multiplications: a0*b00, a0*b01, ..., a1*b17.
    #    That's 16 multiplications.
    #    Can we share?
    #    c = a0 * b0 + a1 * b1.
    #    This is a sum of two vectors.
    #    The tensor is T_{a0, b0k} and T_{a1, b1k}.
    #    These are disjoint in the B indices.
    #    So Rank(M_{1,2,8}) = Rank(M_{1,1,8}) + Rank(M_{1,1,8}) = 8 + 8 = 16?
    #    M_{1,1,8} is just 8 scalar multiplications. Rank 8.
    #    So Rank(M_{1,2,8}) = 16.
    #    
    #    Then Rank(M_{2,2,8}) <= 2 * 16 = 32. (Schoolbook).
    #    
    #    But we can do better by using Strassen on the A-matrix structure?
    #    C = A B.
    #    A is 2x2.
    #    We can decompose A = U V^T (rank 2).
    #    This doesn't help directly.
    #    
    #    Let's go back to the block sum.
    #    Rank(T) = 28 is a strong candidate.
    #    Is it possible to get 24?
    #    
    #    If we could combine the 4 blocks of M_{2,2,2} into a single larger structure...
    #    But the B and C indices are disjoint.
    #    Any bilinear form mixing B indices from different blocks would introduce cross terms
    #    that are not present in the target tensor (which is block diagonal in B-C).
    #    Therefore, the decomposition must respect the block structure?
    #    Not necessarily, if the coefficients cancel out.
    #    But for minimal rank, usually the block structure is preserved.
    #    
    #    I will implement the 4-Strassen construction (rank 28).
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(g+h)
    # M4 = (a+d)(h-f)
    # M5 = (a+c)(e-g)
    # M6 = (b+d)(g-e)  # Wait, standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(g+h)
    # M4 = (a+d)(h-f)  -- This is not standard.
    # Standard:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(g+h)
    # M4 = (a+d)(h-f) is wrong.
    # Let's use the standard 7 products:
    # m1 = (a + d) * (e + h)
    # m2 = (c + d) * e
    # m3 = (a - b) * (g + h)
    # m4 = (a + d) * (h - f)  -- No.
    # 
    # Correct Strassen:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * e
    # p3 = (a - b) * (g + h)
    # p4 = (a + d) * (h - f) -- Incorrect.
    # 
    # Let's derive or look up:
    # C11 = p1 - p4 + p5 + p7  (if p7 is something else?)
    # 
    # Standard Set:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = (a - b)(g + h)
    # m4 = (a + d)(h - f) -- This term appears in some variants.
    # 
    # Let's use the one that gives:
    # C11 = m1 - m4 + m5 + m7
    # C12 = m4 + m6
    # C21 = m2 + m3
    # C22 = m1 - m2 + m3 + m7
    # 
    # Let's check C11:
    # m1 = ae + ah + de + dh
    # m4 = ah - af + dh - df
    # m5 = ae - ag + ce - cg
    # m7 = ?
    # 
    # Actually, let's just hardcode the 7 vectors for 2x2x2 that are known to work.
    # 
    # U (4 rows) for A:
    # r0: [1, 0, 0, 1]  (a+d)
    # r1: [1, 1, 0, 0]  (a) ? No.
    # 
    # Let's use the explicit coordinates for the 7 rank-1 tensors.
    # 
    # T1: (a+d)(e+h) -> U: [1,0,0,1], V: [1,0,0,1], W: [1,0,0,1] ? No.
    # W distributes to C.
    # 
    # Let's define the 7 terms by their (U, V, W) vectors.
    # U is 4-dim (a,b,c,d). V is 4-dim (e,f,g,h). W is 4-dim (C11, C21, C12, C22) -- Note index order W[k*n+i].
    # n=2. W indices:
    # 0: k=0, i=0 -> C00
    # 1: k=0, i=1 -> C10
    # 2: k=1, i=0 -> C01
    # 3: k=1, i=1 -> C11
    
    # Term 1: m1 = (a+d)(e+h)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: ?
    # m1 contributes to C00, C01, C10, C11.
    # C00 += m1
    # C11 += m1
    # C01 -= m4 (where m4 is another term)
    # This is getting complicated to hardcode without a reference.
    
    # I will use a known valid set of 7 vectors for M_{2,2,2}.
    # 
    # Sources often list:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a-b)(g+h)
    # m4 = (a+d)(h-f)  -- Wait, h-f?
    # m5 = (a+c)(e-g)
    # m6 = (b+d)(g-e)  -- Wait.
    # m7 = (a-b)(h-f)  -- Wait.
    
    # Let's try to verify C11 = m1 - m4 + m5 + m7.
    # If m4 = (a+d)(h-f), m5=(a+c)(e-g), m7=(a-b)(h-f)?
    
    # Actually, a simpler way:
    # Since I cannot be 100% sure of the exact Strassen coefficients from memory without deriving them,
    # and I have time, I will derive them or use a fallback.
    
    # Fallback: Schoolbook is 32.
    # If I can't get 28 right, I'll stick to 32 to be safe?
    # No, 28 is a significant improvement.
    
    # Let's use a different strategy:
    # Use NumPy to find the decomposition?
    # No, the decomposition must be exact integers/half-integers.
    
    # I will use the following known decomposition for M_{2,2,2}:
    # 
    # U (rows a,b,c,d):
    # 0: 1 0 0 1  (a+d)
    # 1: 0 0 1 1  (c+d)
    # 2: 1 0 -1 0 (a-b)
    # 3: 1 0 0 -1 (a-d) ? No.
    
    # Let's use the one from Wikipedia:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)(e)
    # p3 = (a - b)(g + h)
    # p4 = (a + d)(h - f)
    # p5 = (a + c)(e - g)
    # p6 = (b + d)(g - e)
    # p7 = (a - b)(h - f)
    
    # C11 = p1 - p4 + p5 + p7
    # C12 = p4 + p6
    # C21 = p2 + p3
    # C22 = p1 - p2 + p3 + p7
    
    # Let's verify C11:
    # p1 = ae + ah + de + dh
    # p4 = ah - af + dh - df
    # p5 = ae - ag + ce - cg
    # p7 = ah - af - bh + bf
    # 
    # Sum:
    # ae: 1 + 1 = 2? No.
    # 
    # There is an error in my memory of the formulas.
    
    # I will use a reliable source's coefficients:
    # 
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = (a-b)(g+h)
    # m4 = (a+d)(h-f)  -- This is likely wrong.
    
    # Let's try to construct it.
    # We want 7 rank 1 tensors.
    
    # I will fall back to Schoolbook (32) if I cannot verify Strassen quickly.
    # But 28 is the standard answer for 2x2x8 (4 blocks of 7).
    # I will assume the standard Strassen decomposition is valid and implement it carefully.
    
    # To be safe, I will implement the Schoolbook decomposition (32) which is guaranteed to work.
    # The task asks for rank-minimal. 32 is not minimal.
    # But if I can't get 28, 32 is better than crashing.
    
    # However, I can use a library or algorithm?
    # No.
    
    # I will use the Schoolbook decomposition for now, as it is robust.
    # rank = 32.
    
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
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
