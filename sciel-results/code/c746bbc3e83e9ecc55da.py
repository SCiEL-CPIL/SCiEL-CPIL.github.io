"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-structured Strassen decomposition for 6x6 matrix multiplication.
    The 6x6 matrices are partitioned into 3x3 blocks of 2x2 matrices.
    Each 2x2 block multiplication uses the rank-7 Strassen algorithm.
    This gives rank = 3^2 * 7 = 63... wait, 3x3 blocks means 9 block positions,
    each with 7 Strassen multiplications, so rank = 9 * 7 = 63?
    
    Actually, let me reconsider. The 6x6 matrices A and B are divided into
    a 3x3 grid of 2x2 blocks. So A has blocks A_{ij} for i,j in {0,1,2},
    each being 2x2. The product C = A*B has blocks C_{ij} = sum_k A_{ik} * B_{kj}.
    
    For each pair (i,j), we need to compute C_{ij} = sum_{k=0}^{2} A_{ik} B_{kj}.
    This is a sum of 3 matrix multiplications of 2x2 matrices.
    
    Each 2x2 multiplication A_{ik} * B_{kj} can be done with 7 multiplications
    using Strassen. But we have 3 such products summed together.
    
    Actually, the standard approach is: for each (i,j) pair, we compute
    C_{ij} = A_{i0}B_{0j} + A_{i1}B_{1j} + A_{i2}B_{2j}.
    
    We can combine the three 2x2 multiplications into a single computation.
    For fixed i and j, define:
      X = [A_{i0}, A_{i1}, A_{i2}]  (a 2x6 matrix)
      Y = [B_{0j}; B_{1j}; B_{2j}]  (a 6x2 matrix)
    
    Then C_{ij} = X * Y, which is a 2x6 times 6x2 product, i.e., a 2x2 result
    from a 12-term dot product. But this doesn't directly help with Strassen.
    
    The better approach: treat the entire 6x6 multiplication as a block matrix
    multiplication where each block is 2x2. The number of scalar multiplications
    needed is the same as multiplying two 3x3 matrices of 2x2 blocks.
    
    The standard Strassen recursion: to multiply two 3x3 block matrices where
    each block is 2x2, we can use Strassen to reduce the number of block
    multiplications from 9 to 7. Each block multiplication is a 2x2 x 2x2 product,
    which itself can be done with 7 scalar multiplications.
    
    So total rank = 7 (Strassen for 3x3 blocks) * 7 (Strassen for 2x2 within each block)
    = 49? No, that's not right either.
    
    Let me think more carefully. The Strassen algorithm for n x n matrix multiplication
    (where n is a power of 2) has rank R(n) = 7^{log2(n)}. For n=8, R(8) = 343.
    For n=4, R(4) = 49. For n=2, R(2) = 7.
    
    But 6 is not a power of 2. The known best simple construction for 6x6 matrix
    multiplication uses the following: partition into 2x2 blocks in a 3x3 grid.
    Use Strassen-like methods at the block level.
    
    Actually, the measured result from the knowledge says: "Implementing the correct
    block-structured Strassen decomposition (3x3 blocks of 2x2, schoolbook block
    structure with rank-7 Strassen per block) reduces rank from 216 to 189."
    
    189 = 27 * 7. So it seems like there are 27 block multiplications, each using
    rank-7 Strassen. But 27 = 3^3, which suggests a 3x3x3 structure.
    
    Wait, I think the construction is: we have 3x3 = 9 block positions for A and 9
    for B. The product has 9 block positions. For each output block C_{ij}, we need
    to compute sum over k of A_{ik} B_{kj}. That's 3 terms. So we have 9 * 3 = 27
    individual 2x2 x 2x2 multiplications. Each is done with Strassen (rank 7).
    Total rank = 27 * 7 = 189.
    
    But this isn't using Strassen at the block level to reduce the 9 block multiplications
    to 7. It's just doing 27 separate 2x2 multiplications, each with Strassen.
    
    Hmm, but 27 separate 2x2 multiplications is just the schoolbook method at the block
    level (9 output blocks, each requiring 3 input block multiplications). So this is
    schoolbook at the block level, Strassen within each block.
    
    Rank = 27 * 7 = 189. That matches the known result.
    
    So the construction is:
    1. For each (i, j, k) where i,j,k in {0,1,2}, we compute A_{ik} * B_{kj} using
       Strassen (7 multiplications).
    2. Sum these contributions to get C_{ij}.
    
    Let me implement this.
    
    Strassen for 2x2:
    Given P = [[a, b], [c, d]] and Q = [[e, f], [g, h]],
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    R11 = M1 + M4 - M5 + M7
    R12 = M3 + M5
    R21 = M2 + M4
    R22 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block size is 2, number of blocks in each dimension is 3
    bs = 2
    nb = 3  # number of blocks per dimension
    
    # Total rank: 3*3*3 block positions * 7 Strassen multiplications per 2x2 product
    total_rank = 27 * 7
    
    # Initialize factor matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    
    # Strassen coefficients for the 2x2 product
    # For each of the 7 Strassen multiplications, we have:
    # M_t = (linear combo of A entries) * (linear combo of B entries)
    # The result contributes to C entries with specific coefficients.
    
    # Strassen: 
    # M1 = (a+d)(e+h) -> coeff in A: [1,0,0,1] (a,d), coeff in B: [1,1,0,0] (e,f->no, e and h)
    # Let's be precise. A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h): A coeffs: a=1, b=0, c=0, d=1; B coeffs: e=1, f=0, g=0, h=1
    # M2 = (c+d)(e): A coeffs: a=0, b=0, c=1, d=1; B coeffs: e=1, f=0, g=0, h=0
    # M3 = (a)(f-h): A coeffs: a=1, b=0, c=0, d=0; B coeffs: e=0, f=1, g=0, h=-1
    # M4 = (d)(g-e): A coeffs: a=0, b=0, c=0, d=1; B coeffs: e=-1, f=0, g=1, h=0
    # M5 = (a+b)(h): A coeffs: a=1, b=1, c=0, d=0; B coeffs: e=0, f=0, g=0, h=1
    # M6 = (c-a)(e+f): A coeffs: a=-1, b=0, c=1, d=0; B coeffs: e=1, f=1, g=0, h=0
    # M7 = (b-d)(g+h): A coeffs: a=0, b=1, c=0, d=-1; B coeffs: e=0, f=0, g=1, h=1
    
    # Output coefficients:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A entry indices within a 2x2 block: 0=a, 1=b, 2=c, 3=d
    # B entry indices within a 2x2 block: 0=e, 1=f, 2=g, 3=h
    
    strassen_A = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1], # M7: b-d
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [-1, 0, 1, 0], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1],  # M7: g+h
    ]
    
    # Output coefficient matrix: output[c_idx] gets contribution from strassen mult t
    # c_idx: 0=C11, 1=C12, 2=C21, 3=C22
    output_coeffs = [
        [1, 0, 0, 0],  # M1 contributes to C11 with coeff 1, C12:0, C21:0, C22:0
        [0, 0, 1, 0],  # M2 contributes to C21 with coeff 1
        [0, 1, 0, 1],  # M3 contributes to C12 with coeff 1, C22 with coeff 1
        [1, 0, 1, 0],  # M4 contributes to C11 with coeff 1, C21 with coeff 1
        [-1, 1, 0, 0], # M5 contributes to C11 with coeff -1, C12 with coeff 1
        [0, 0, 0, 1],  # M6 contributes to C22 with coeff 1
        [1, 0, 0, 0],  # M7 contributes to C11 with coeff 1
    ]
    # Wait, let me re-check:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # M1: C11+=1, C22+=1
    # M2: C21+=1, C22-=1
    # M3: C12+=1, C22+=1
    # M4: C11+=1, C21+=1
    # M5: C11-=1, C12+=1
    # M6: C22+=1
    # M7: C11+=1
    
    output_coeffs = [
        [1, 0, 0, 1],   # M1: C11=1, C12=0, C21=0, C22=1
        [0, 0, 1, -1],  # M2: C11=0, C12=0, C21=1, C22=-1
        [0, 1, 0, 1],   # M3: C11=0, C12=1, C21=0, C22=1
        [1, 0, 1, 0],   # M4: C11=1, C12=0, C21=1, C22=0
        [-1, 1, 0, 0],  # M5: C11=-1, C12=1, C21=0, C22=0
        [0, 0, 0, 1],   # M6: C11=0, C12=0, C21=0, C22=1
        [1, 0, 0, 0],   # M7: C11=1, C12=0, C21=0, C22=0
    ]
    
    # For each block position (i, j, k) and Strassen mult t:
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for t in range(7):
                    # A block A_{ik}: row block i, col block k
                    # A entry (r, c) in block (i,k) is at global position (i*bs + r, k*bs + c)
                    # U index: (i*bs + r) * m + (k*bs + c) = (i*2+r)*6 + (k*2+c)
                    
                    # B block B_{kj}: row block k, col block j
                    # B entry (r, c) in block (k,j) is at global position (k*bs + r, j*bs + c)
                    # V index: (k*bs + r) * p + (j*bs + c) = (k*2+r)*6 + (j*2+c)
                    
                    # C block C_{ij}: row block i, col block j
                    # C entry (r, c) in block (i,j) is at global position (i*bs + r, j*bs + c)
                    # W index: (j*bs + c) * n + (i*bs + r) = (j*2+c)*6 + (i*2+r)
                    # Note: W[k*n+i, r] where k is column index of C, i is row index of C
                    # So W index = col_C * n + row_C = (j*2+c)*6 + (i*2+r)
                    
                    for r in range(bs):
                        for c in range(bs):
                            # A entry contribution
                            a_idx = r * bs + c  # local index in 2x2 block: 0,1,2,3
                            a_coeff = strassen_A[t][a_idx]
                            if a_coeff != 0:
                                U[(i * bs + r) * m + (k * bs + c)][col] = a_coeff
                    
                    for r in range(bs):
                        for c in range(bs):
                            # B entry contribution
                            b_idx = r * bs + c
                            b_coeff = strassen_B[t][b_idx]
                            if b_coeff != 0:
                                V[(k * bs + r) * p + (j * bs + c)][col] = b_coeff
                    
                    for r in range(bs):
                        for c in range(bs):
                            # C entry contribution
                            c_local_idx = r * bs + c
                            c_coeff = output_coeffs[t][c_local_idx]
                            if c_coeff != 0:
                                W[(j * bs + c) * n + (i * bs + r)][col] = c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
