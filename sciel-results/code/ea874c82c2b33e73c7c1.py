"""Matrix multiplication 4x4x8 with Strassen-based decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Split the 4x8 product into two 4x4 products:
      C = A * B = A * [B1 | B2] = [A*B1 | A*B2]
    Each 4x4x4 multiplication uses Strassen's algorithm with rank 7.
    Total rank = 2 * 7 = 14... wait, that's not right.
    
    Actually, for 4x4x4, Strassen gives rank 7 per 2x2x2 block, and 4x4x4 
    can be done with rank 49 using recursive Strassen (or known minimal rank).
    
    Let me think more carefully. The known minimal rank for 4x4x4 matrix 
    multiplication is 49. So for 4x4x8, we can split B into two 4x4 blocks,
    giving rank 49 + 49 = 98.
    
    The 4x4x4 rank-49 decomposition comes from applying Strassen recursively:
    4x4x4 -> 2 blocks of 2x2x2, each with rank 7, total 14? No, that's wrong.
    
    Actually, 4x4x4 can be decomposed using 49 rank-1 terms. This is achieved
    by viewing 4x4x4 as 8 blocks of 2x2x2, but with sharing. The standard 
    result is that R(4,4,4) = 49.
    
    Let me implement the 4x4x4 rank-49 decomposition explicitly.
    
    For a 4x4 matrix product C = A * B where A, B are 4x4:
    We can use the following approach based on the work of Strassen and others.
    
    Actually, let me use a simpler approach. The 4x4x4 multiplication tensor 
    has rank 49. One way to construct this is:
    
    Split indices i, j, k in {0,1,2,3} into high/low bits:
    i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0
    
    Then C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using the Kronecker product structure, we can write this as a sum of 
    49 rank-1 terms.
    
    Let me use the explicit construction from the literature. The 4x4x4 
    multiplication can be done with 49 multiplications using the following 
    scheme (based on the recursive Strassen approach):
    
    Actually, the simplest correct approach: use the known fact that 
    4x4x4 has rank 49, and construct it via 2x2x2 Strassen blocks.
    
    For 2x2x2, Strassen's 7 multiplications are:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    For 4x4x4, we can view it as a 2x2 block matrix where each block is 2x2:
    A = [[A00, A01], [A10, A11]]  (each Ai is 2x2)
    B = [[B00, B01], [B10, B11]]  (each Bj is 2x2)
    
    C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
         [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Using Strassen at the block level (7 block multiplications of 2x2 matrices),
    and then Strassen at the 2x2 level (7 scalar multiplications each),
    we get 7 * 7 = 49 scalar multiplications.
    
    So the rank-49 decomposition for 4x4x4 is:
    - 7 block-level Strassen combinations
    - Each block multiplication uses 7 scalar Strassen combinations
    - Total: 49 rank-1 terms
    
    Now for 4x4x8:
    C = A * B where A is 4x4, B is 4x8.
    Split B into B = [B_left | B_right] where each is 4x4.
    C = [A * B_left | A * B_right]
    
    Each A * B_left and A * B_right is a 4x4x4 multiplication with rank 49.
    Total rank = 98.
    
    Let me now construct the explicit U, V, W matrices.
    
    For the 4x4x4 part, I need to map the 49 Strassen multiplications to 
    the U, V, W format where:
    U has shape (4*4, 49) = (16, 49)  -- rows indexed by (i,j)
    V has shape (4*4, 49) = (16, 49)  -- rows indexed by (j,k)
    W has shape (4*4, 49) = (16, 49)  -- rows indexed by (k,i)
    
    The multiplication tensor is: T[i,j,k] = delta[i,j] * delta[j,k] ... no.
    
    Actually, the matrix multiplication tensor for (n,m,p) is:
    T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    
    So the tensor has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p),
    and the decomposition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    (since every A[i,j]*B[j,k] term appears exactly once in C[k,i]).
    
    Wait, that's not right either. Let me reconsider.
    
    The matrix multiplication tensor for multiplying an n x m matrix by an 
    m x p matrix to get an n x p result is:
    
    C[r, c] = sum_j A[r, j] * B[j, c]
    
    So the tensor T has indices (r, j, c) and T[r, j, c] = 1.
    
    In the problem's notation with W[k*n+i, r]:
    - i corresponds to row index of A (and output C): i in [0, n)
    - j corresponds to column of A / row of B: j in [0, m)  
    - k corresponds to column of B (and output C): k in [0, p)
    
    U[i*m + j, r] -- row indexed by (i, j)
    V[j*p + k, r] -- row indexed by (j, k)
    W[k*n + i, r] -- row indexed by (k, i)
    
    And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    This is the standard matrix multiplication tensor.
    
    Now let me construct the Strassen-based decomposition.
    
    For 4x4x4 (n=m=p=4):
    Split i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0 where i1,j1,k1 in {0,1} and i0,j0,k0 in {0,1}.
    
    A[i,j] = A[i1*2+i0, j1*2+j0]
    B[j,k] = B[j1*2+j0, k1*2+k0]
    
    Block structure:
    A = [[A00, A01], [A10, A11]] where Aab is 2x2 with indices (i0, j0) for a=b
    B = [[B00, B01], [B10, B11]] where Bcd is 2x2 with indices (j0, k0) for c=d
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Block level:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Strassen at block level (7 block multiplications):
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    Then:
    C00 = M1 - M4 + M2 - M7  ... wait, let me get the standard Strassen formulas right.
    
    Standard Strassen for 2x2:
    C00 = M1 - M4 + M2 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M5 - M6
    
    where:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    Now each Mx is a 2x2 matrix product, which we decompose using 2x2 Strassen (7 scalar mults).
    
    For a 2x2 product P = X * Y where X, Y are 2x2:
    P00 = S1 - S4 + S2 - S7
    P01 = S3 + S5
    P10 = S2 + S4
    P11 = S1 - S3 + S5 - S6
    
    where:
    S1 = (X00+X11)*(Y00+Y11)
    S2 = (X10+X11)*Y00
    S3 = X00*(Y01-Y11)
    S4 = X11*(Y10-Y00)
    S5 = (X00+X01)*Y11
    S6 = (X10-X00)*(Y00+Y01)
    S7 = (X01-X11)*(Y10+Y11)
    
    So for each block-level Mx, we have 7 scalar multiplications.
    Total: 7 * 7 = 49 scalar multiplications for 4x4x4.
    
    Now I need to trace through all 49 scalar multiplications and determine 
    their coefficients in U, V, W.
    
    Let me define the 49 terms. For block-level term b in {1..7} and 2x2-level 
    term s in {1..7}, we have scalar multiplication term (b,s).
    
    For each such term, I need to find:
    - The linear combination of A entries (gives U contributions)
    - The linear combination of B entries (gives V contributions)  
    - The linear combination of C entries (gives W contributions)
    
    This is getting complex. Let me just build it programmatically.
    
    I'll represent each of the 49 terms as (coeff_U, coeff_V, coeff_W) where 
    coeff_U is a vector in R^16 (for A entries), coeff_V in R^16 (for B entries),
    and coeff_W in R^16 (for C entries).
    
    Actually, let me just construct the full 49-term decomposition by 
    composing the two levels of Strassen.
    """
    import time
    start_time = time.time()
    deadline = 150.0  # seconds
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll build the 4x4x4 rank-49 decomposition, then duplicate for 4x4x8.
    # For 4x4x4: U4 (16x49), V4 (16x49), W4 (16x49)
    
    # Let's build it by composing Strassen levels.
    
    # First, define the 2x2x2 Strassen decomposition.
    # For 2x2x2: U2 (4x7), V2 (4x7), W2 (4x7)
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    # i,j,k in {0,1}
    
    # Strassen 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 + M2 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    U2 = [[0]*7 for _ in range(4)]  # rows: (i,j) -> i*2+j
    V2 = [[0]*7 for _ in range(4)]  # rows: (j,k) -> j*2+k
    W2 = [[0]*7 for _ in range(4)]  # rows: (k,i) -> k*2+i
    
    # M1: (A00+A11)*(B00+B11)
    # U: A00=1, A11=1 -> U[0*2+0][0]=1, U[1*2+1][0]=1
    U2[0][0] = 1  # A00
    U2[3][0] = 1  # A11
    V2[0][0] = 1  # B00
    V2[3][0] = 1  # B11
    # W: contributes to C00, C11 with +1, C01 with -1, C10 with -1
    # C00: k=0,i=0 -> W[0*2+0][0] = +1
    # C11: k=1,i=1 -> W[1*2+1][0] = +1
    # C01: k=0,i=1 -> W[0*2+1][0] = -1
    # C10: k=1,i=0 -> W[1*2+0][0] = -1
    W2[0][0] = 1   # C00
    W2[3][0] = 1   # C11
    W2[1][0] = -1  # C01
    W2[2][0] = -1  # C10
    
    # M2: (A10+A11)*B00
    U2[2][1] = 1  # A10
    U2[3][1] = 1  # A11
    V2[0][1] = 1  # B00
    # C00: +1, C10: +1
    W2[0][1] = 1   # C00
    W2[2][1] = 1   # C10
    
    # M3: A00*(B01-B11)
    U2[0][2] = 1  # A00
    V2[1][2] = 1  # B01
    V2[3][2] = -1 # B11
    # C01: +1, C11: -1
    W2[1][2] = 1   # C01
    W2[3][2] = -1  # C11
    
    # M4: A11*(B10-B00)
    U2[3][3] = 1  # A11
    V2[2][3] = 1  # B10
    V2[0][3] = -1 # B00
    # C00: -1, C10: +1
    W2[0][3] = -1  # C00
    W2[2][3] = 1   # C10
    
    # M5: (A00+A01)*B11
    U2[0][4] = 1  # A00
    U2[1][4] = 1  # A01
    V2[3][4] = 1  # B11
    # C01: +1, C11: +1
    W2[1][4] = 1   # C01
    W2[3][4] = 1   # C11
    
    # M6: (A10-A00)*(B00+B01)
    U2[2][5] = 1  # A10
    U2[0][5] = -1 # A00
    V2[0][5] = 1  # B00
    V2[1][5] = 1  # B01
    # C11: -1
    W2[3][5] = -1  # C11
    
    # M7: (A01-A11)*(B10+B11)
    U2[1][6] = 1  # A01
    U2[3][6] = -1 # A11
    V2[2][6] = 1  # B10
    V2[3][6] = 1  # B11
    # C00: -1, C11: -1
    W2[0][6] = -1  # C00
    W2[3][6] = -1  # C11
    
    # Now build 4x4x4 by composing two levels of Strassen.
    # Block level: 7 block multiplications, each is a 2x2 matrix product.
    # For each block multiplication, we use the 2x2 Strassen decomposition.
    
    # Block-level Strassen formulas:
    # M1_block = (A00+A11)*(B00+B11)
    # M2_block = (A10+A11)*B00
    # M3_block = A00*(B01-B11)
    # M4_block = A11*(B10-B00)
    # M5_block = (A00+A01)*B11
    # M6_block = (A10-A00)*(B00+B01)
    # M7_block = (A01-A11)*(B10+B11)
    
    # C_block:
    # C00 = M1 - M4 + M2 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # For each block multiplication Mx_block = X * Y where X, Y are 2x2 matrices
    # (linear combinations of the 2x2 sub-blocks of A and B), we decompose using 
    # the 2x2 Strassen scheme.
    
    # Let me define the block-level linear combinations.
    # A blocks: A00, A01, A10, A11 (each 2x2, indexed by (i0, j0))
    # B blocks: B00, B01, B10, B11 (each 2x2, indexed by (j0, k0))
    
    # For block multiplication Mx = X * Y:
    # X is a linear combination of A-blocks
    # Y is a linear combination of B-blocks
    
    # Let me define for each block-level term b=0..6:
    #   X_blocks[b]: dict mapping A-block-index to coefficient
    #   Y_blocks[b]: dict mapping B-block-index to coefficient
    #   C_blocks[b]: dict mapping C-block-index to coefficient (with sign)
    
    # A-block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B-block indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C-block indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    X_blocks = [
        {0: 1, 3: 1},   # M1: A00 + A11
        {2: 1, 3: 1},   # M2: A10 + A11
        {0: 1},         # M3: A00
        {3: 1},         # M4: A11
        {0: 1, 1: 1},   # M5: A00 + A01
        {2: 1, 0: -1},  # M6: A10 - A00
        {1: 1, 3: -1},  # M7: A01 - A11
    ]
    
    Y_blocks = [
        {0: 1, 3: 1},   # M1: B00 + B11
        {0: 1},         # M2: B00
        {1: 1, 3: -1},  # M3: B01 - B11
        {2: 1, 0: -1},  # M4: B10 - B00
        {3: 1},         # M5: B11
        {0: 1, 1: 1},   # M6: B00 + B01
        {2: 1, 3: 1},   # M7: B10 + B11
    ]
    
    C_blocks = [
        {0: 1, 3: 1, 1: -1, 2: -1},  # M1: C00+C11-C01-C10
        {0: 1, 2: 1},                 # M2: C00+C10
        {1: 1, 3: -1},                # M3: C01-C11
        {0: -1, 2: 1},                # M4: -C00+C10
        {1: 1, 3: 1},                 # M5: C01+C11
        {3: -1},                      # M6: -C11
        {0: -1, 3: -1},               # M7: -C00-C11
    ]
    
    # Now for each block-level term b, and each 2x2-level term s (from U2,V2,W2),
    # we create a scalar-level term.
    
    # The 2x2 Strassen terms (s=0..6) have:
    # U2 contributions: for each (i0,j0), coefficient in the X matrix
    # V2 contributions: for each (j0,k0), coefficient in the Y matrix
    # W2 contributions: for each (k0,i0), coefficient in the P matrix
    
    # For block term b, X is sum over A-blocks a with coeff X_blocks[b][a]
    # Each A-block a corresponds to a 2x2 matrix with indices (i0, j0) where 
    # the block index a = a1*2 + a0 determines which 2x2 sub-block.
    
    # Wait, I need to be more careful. Let me redefine.
    
    # A is 4x4, split into 2x2 blocks:
    # A[i1*2+i0, j1*2+j0]
    # Block A[a1][a0] is the 2x2 submatrix with i1=a1, j1=a0.
    # So A00 has i1=0,j1=0; A01 has i1=0,j1=1; A10 has i1=1,j1=0; A11 has i1=1,j1=1.
    
    # Similarly for B: B[c1][c0] has j1=c1, k1=c0.
    # B00: j1=0,k1=0; B01: j1=0,k1=1; B10: j1=1,k1=0; B11: j1=1,k1=1.
    
    # C: C[d1][d0] has k1=d1, i1=d0.
    # C00: k1=0,i1=0; C01: k1=0,i1=1; C10: k1=1,i1=0; C11: k1=1,i1=1.
    
    # For block multiplication Mx = X * Y:
    # X is a 2x2 matrix (in block indices) formed from linear combinations of A-blocks.
    # Y is a 2x2 matrix formed from linear combinations of B-blocks.
    # The result P = X*Y is a 2x2 matrix of C-blocks.
    
    # When we apply 2x2 Strassen to P = X*Y, we get 7 scalar multiplications.
    # Each scalar multiplication involves:
    #   - A linear combination of entries of X (which are themselves linear combinations of A entries)
    #   - A linear combination of entries of Y (which are linear combinations of B entries)
    #   - And contributes to a linear combination of entries of P (which are C entries)
    
    # Let me work this out explicitly.
    
    # For 2x2 Strassen on matrices X, Y (each 2x2) producing P = X*Y:
    # S1 = (X00+X11)*(Y00+Y11)
    # S2 = (X10+X11)*Y00
    # S3 = X00*(Y01-Y11)
    # S4 = X11*(Y10-Y00)
    # S5 = (X00+X01)*Y11
    # S6 = (X10-X00)*(Y00+Y01)
    # S7 = (X01-X11)*(Y10+Y11)
    
    # P00 = S1 - S4 + S2 - S7
    # P01 = S3 + S5
    # P10 = S2 + S4
    # P11 = S1 - S3 + S5 - S6
    
    # For each scalar term s, the contribution to U (A-entries), V (B-entries), W (C-entries):
    
    # X00 corresponds to A-block X_blocks[b][0] (if present), which is a 2x2 sub-block of A.
    # X01 corresponds to A-block X_blocks[b][1]
    # X10 corresponds to A-block X_blocks[b][2]
    # X11 corresponds to A-block X_blocks[b][3]
    
    # Similarly for Y and C.
    
    # Let me define for each 2x2 Strassen term s, the coefficients on X-entries and Y-entries:
    
    # s=0 (S1): X: {00:1, 11:1}, Y: {00:1, 11:1}
    # s=1 (S2): X: {10:1, 11:1}, Y: {00:1}
    # s=2 (S3): X: {00:1}, Y: {01:1, 11:-1}
    # s=3 (S4): X: {11:1}, Y: {10:1, 00:-1}
    # s=4 (S5): X: {00:1, 01:1}, Y: {11:1}
    # s=5 (S6): X: {10:1, 00:-1}, Y: {00:1, 01:1}
    # s=6 (S7): X: {01:1, 11:-1}, Y: {10:1, 11:1}
    
    X_coeffs = [
        {0: 1, 3: 1},   # S1: X00, X11
        {2: 1, 3: 1},   # S2: X10, X11
        {0: 1},         # S3: X00
        {3: 1},         # S4: X11
        {0: 1, 1: 1},   # S5: X00, X01
        {2: 1, 0: -1},  # S6: X10, X00
        {1: 1, 3: -1},  # S7: X01, X11
    ]
    
    Y_coeffs = [
        {0: 1, 3: 1},   # S1: Y00, Y11
        {0: 1},         # S2: Y00
        {1: 1, 3: -1},  # S3: Y01, Y11
        {2: 1, 0: -1},  # S4: Y10, Y00
        {3: 1},         # S5: Y11
        {0: 1, 1: 1},   # S6: Y00, Y01
        {2: 1, 3: 1},   # S7: Y10, Y11
    ]
    
    # P-coefficients for each S term:
    # P00 = S1 - S4 + S2 - S7
    # P01 = S3 + S5
    # P10 = S2 + S4
    # P11 = S1 - S3 + S5 - S6
    
    P_coeffs = [
        {0: 1, 3: 1, 1: -1, 2: -1},  # S1
        {0: 1, 2: 1},                 # S2
        {1: 1, 3: -1},                # S3
        {0: -1, 2: 1},                # S4
        {1: 1, 3: 1},                 # S5
        {3: -1},                      # S6
        {0: -1, 3: -1},               # S7
    ]
    
    # Now for each block term b and 2x2 term s, the scalar multiplication is:
    # (sum over a: X_coeffs[s][a] * X_blocks[b][a] * A_block_a) * 
    # (sum over c: Y_coeffs[s][c] * Y_blocks[b][c] * B_block_c)
    
    # This contributes to C-blocks according to P_coeffs[s].
    
    # Let me now build the 4x4x4 U, V, W matrices.
    
    U4 = [[0]*49 for _ in range(16)]  # 16 = 4*4, rows indexed by (i,j)
    V4 = [[0]*49 for _ in range(16)]  # 16 = 4*4, rows indexed by (j,k)
    W4 = [[0]*49 for _ in range(16)]  # 16 = 4*4, rows indexed by (k,i)
    
    for b in range(7):
        for s in range(7):
            col = b * 7 + s
            
            # U contributions: for each A-block a with nonzero coeff in both X_coeffs[s] and X_blocks[b]
            for a, coeff_x in X_coeffs[s].items():
                if a in X_blocks[b]:
                    coeff_a = X_blocks[b][a]
                    # A-block a is at position (a1, a0) in the 2x2 block grid
                    a1 = a // 2
                    a0 = a % 2
                    # For each (i0, j0) in this 2x2 block:
                    for i0 in range(2):
                        for j0 in range(2):
                            i = a1 * 2 + i0
                            j = a0 * 2 + j0
                            U4[i * 4 + j][col] += coeff_x * coeff_a
            
            # V contributions: for each B-block c with nonzero coeff in both Y_coeffs[s] and Y_blocks[b]
            for c, coeff_y in Y_coeffs[s].items():
                if c in Y_blocks[b]:
                    coeff_c = Y_blocks[b][c]
                    c1 = c // 2
                    c0 = c % 2
                    for j0 in range(2):
                        for k0 in range(2):
                            j = c1 * 2 + j0
                            k = c0 * 2 + k0
                            V4[j * 4 + k][col] += coeff_y * coeff_c
            
            # W contributions: for each C-block d with nonzero coeff in P_coeffs[s]
            for d, coeff_p in P_coeffs[s].items():
                d1 = d // 2
                d0 = d % 2
                for k0 in range(2):
                    for i0 in range(2):
                        k = d1 * 2 + k0
                        i = d0 * 2 + i0
                        W4[k * 4 + i][col] += coeff_p
    
    # Now for 4x4x8: split B into B_left (4x4) and B_right (4x4).
    # C = A * [B_left | B_right] = [A*B_left | A*B_right]
    # 
    # U is the same for both halves (A is the same).
    # V is [V4_left; V4_right] where V4_left uses B_left entries and V4_right uses B_right entries.
    # W is [W4_left; W4_right] where W4_left uses C_left entries and W4_right uses C_right entries.
    
    # For the left half (columns 0-3 of B, columns 0-3 of C):
    # V_left[j*4+k, col] = V4[j*4+k, col] for k in 0..3
    # W_left[k*4+i, col] = W4[k*4+i, col] for k in 0..3
    
    # For the right half (columns 4-7 of B, columns 4-7 of C):
    # V_right[j*4+k, col] = V4[j*4+(k-4), col] for k in 4..7
    # W_right[k*4+i, col] = W4[(k-4)*4+i, col] for k in 4..7
    
    rank = 98
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]  # 4*8
    W = [[0]*rank for _ in range(32)]  # 8*4
    
    for col in range(49):
        # Left half
        for row in range(16):
            U[row][col] = U4[row][col]
        for row in range(16):
            V[row][col] = V4[row][col]
        for row in range(16):
            W[row][col] = W4[row][col]
        
        # Right half
        for row in range(16):
            U[row][col + 49] = U4[row][col]
        for row in range(16):
            V[row][col + 49] = V4[row][col]
        for row in range(16):
            W[row + 16][col + 49] = W4[row][col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
