"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_decomp(n, m, p):
    """
    Build a decomposition using block structure.
    For 2x3x8: split the p=8 dimension into two blocks of p=4.
    Use rank-16 decomposition for each 2x3x4 block (schoolbook is 2*3*4=24,
    but we can do better using the known structure).
    
    Actually, let's think more carefully.
    
    The matrix multiplication tensor T_{n,m,p} has entries T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    where (i,j) indexes A, (j,k) indexes B, (k,i) indexes C.
    
    For 2x3x8: A is 2x3, B is 3x8, C is 2x8.
    
    One approach: split B's columns into groups. B has 8 columns.
    We can group them into 4 groups of 2 columns each, or 2 groups of 4.
    
    Let's try: split the 8 columns of B (and corresponding 8 columns of C) into 
    4 blocks of 2 columns each.
    
    For each block b (0..3), B_b is 3x2 and C_b is 2x2.
    The sub-problem for each block is: multiply 2x3 matrix by 3x2 matrix to get 2x2.
    That's M(2,3,2). Schoolbook rank = 2*3*2 = 12.
    
    Can we do M(2,3,2) better? 
    M(2,3,2): A is 2x3 = [a1 a2 a3; b1 b2 b3], B is 3x2, C is 2x2.
    
    C[0,0] = a1*B[0,0] + a2*B[1,0] + a3*B[2,0]
    C[0,1] = a1*B[0,1] + a2*B[1,1] + a3*B[2,1]
    C[1,0] = b1*B[0,0] + b2*B[1,0] + b3*B[2,0]
    C[1,1] = b1*B[0,1] + b2*B[1,1] + b3*B[2,1]
    
    This is essentially 4 independent dot products of length 3, but with shared structure.
    The rank of M(2,3,2) - is it 12? 
    
    Actually, M(2,3,2) can be seen as: we need to compute 2*2=4 outputs, each being a dot product of length 3.
    Total = 12 multiplications in schoolbook.
    
    But we can use bilinear tricks. Consider:
    (a1+a2+a3)(B[0,0]+B[1,0]+B[2,0]) gives us cross terms.
    
    Actually, the rank of M(2,3,2) is known. Let me think...
    It's equivalent to computing the product of a 2x3 and 3x2 matrix.
    
    One known result: rank(M(2,3,2)) = 10 (from Bini, Capovani, Lotti, Romani 1981, or later work).
    
    Let me try to construct a rank-10 decomposition for M(2,3,2).
    
    A = [a1, a2, a3; b1, b2, b3], B = [c1, c2; c3, c4; c5, c6], C = [C11, C12; C21, C22]
    
    C11 = a1*c1 + a2*c3 + a3*c5
    C12 = a1*c2 + a2*c4 + a3*c6
    C21 = b1*c1 + b2*c3 + b3*c5
    C22 = b1*c2 + b2*c4 + b3*c6
    
    Let's define intermediate values:
    Let x1 = a1+b1, x2 = a2+b2, x3 = a3+b3 (sums of A rows)
    Let y1 = c1+c2, y2 = c3+c4, y3 = c5+c6 (sums of B columns... wait, B is 3x2)
    
    Actually B is 3x2: rows are (c1,c2), (c3,c4), (c5,c6).
    
    Let me try a different approach. Use the fact that M(2,3,2) can be embedded in larger problems.
    
    Alternatively, let's just use schoolbook for the sub-blocks but try to share computations.
    
    Actually, let me reconsider. The total schoolbook rank is 48. If I can reduce each M(2,3,2) block from 12 to, say, 10, I get 4*10=40.
    
    But I'm not 100% sure of the rank-10 construction. Let me try another approach.
    
    Alternative: Use the structure that A is 2x3. We can write:
    A = [A_left | A_right] where A_left is 2x2 (cols 0,1) and A_right is 2x1 (col 2).
    Similarly B can be split.
    
    Actually, let me try a simpler optimization. 
    
    Key insight: The 2x3x8 problem can be viewed as 4 independent 2x3x2 problems (one per pair of columns).
    If rank(M(2,3,2)) = 10, total rank = 40.
    
    Let me try to find a rank-10 decomposition for M(2,3,2).
    
    M(2,3,2): indices i∈{0,1}, j∈{0,1,2}, k∈{0,1}
    Tensor T[i,j,k] contributes to C[i,k] += A[i,j]*B[j,k]
    
    Schoolbook: 12 terms, one for each (i,j,k).
    
    To reduce, we need to combine terms. For example:
    A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] = C[0,0]
    A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0] = C[1,0]
    
    We can compute (A[0,0]+A[1,0])*B[0,0] = A[0,0]*B[0,0] + A[1,0]*B[0,0]
    Then C[0,0] needs A[0,0]*B[0,0] = (A[0,0]+A[1,0])*B[0,0] - A[1,0]*B[0,0]
    
    This doesn't immediately help.
    
    Let me try a known construction. For M(2,3,2), one approach:
    
    Define:
    p1 = A[0,0]*B[0,0]
    p2 = A[0,1]*B[1,0]
    p3 = A[0,2]*B[2,0]
    p4 = A[1,0]*B[0,0]
    p5 = A[1,1]*B[1,0]
    p6 = A[1,2]*B[2,0]
    p7 = A[0,0]*B[0,1]
    p8 = A[0,1]*B[1,1]
    p9 = A[0,2]*B[2,1]
    p10 = A[1,0]*B[0,1]
    p11 = A[1,1]*B[1,1]
    p12 = A[1,2]*B[2,1]
    
    That's schoolbook 12.
    
    To get 10, we need to eliminate 2 terms. 
    
    Try:
    Let s1 = A[0,0]+A[1,0], d1 = A[0,0]-A[1,0]
    Then A[0,0]*B[0,0] + A[1,0]*B[0,0] = s1*B[0,0]
    And A[0,0]*B[0,0] - A[1,0]*B[0,0] = d1*B[0,0]
    
    So: A[0,0]*B[0,0] = (s1*B[0,0] + d1*B[0,0])/2
         A[1,0]*B[0,0] = (s1*B[0,0] - d1*B[0,0])/2
    
    But this uses 2 terms (s1*B[0,0] and d1*B[0,0]) to replace 2 terms, no savings.
    
    Let me try combining across j:
    A[0,0]*B[0,0] + A[0,1]*B[1,0] = ?
    
    This is getting complex. Let me just try to see if we can do better than schoolbook in a simple way.
    
    Actually, for the purpose of this task, let me try a different strategy:
    Use the 4x4x4 Strassen-like decomposition if applicable, or just try to find any improvement.
    
    For 2x3x8, another approach: note that 2x3x8 = 2x(3x8). 
    The 3x8 part: B is 3x8. We can think of this as computing C = A*B where A is 2x3.
    C has 2*8=16 entries, each a sum of 3 products. Total 48 scalar multiplications.
    
    Can we reduce? The minimum rank of M(2,3,8) is what we're looking for.
    
    Known bounds: rank(M(2,3,8)) >= some lower bound.
    Upper bound: schoolbook = 48.
    
    Let me try to use a block decomposition:
    Split A into [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Split B into [B1; B2] where B1 is 2x8, B2 is 1x8.
    Then C = A1*B1 + A2*B2.
    
    A1*B1 is 2x2 * 2x8 = 2x8, rank at most 2*2*8=32 (schoolbook) but can be reduced.
    A2*B2 is 2x1 * 1x8 = 2x8, rank at most 2*1*8=16.
    
    But A1*B1 is M(2,2,8). If we can do M(2,2,8) with rank r1 and M(2,1,8) with rank r2, total is r1+r2.
    
    M(2,1,8): A is 2x1 = [a; b], B is 1x8 = [b1...b8], C is 2x8.
    C[0,k] = a*bk, C[1,k] = b*bk.
    Rank = 2*8 = 16 (each output needs one multiplication, and a and b are shared but can't be factored further since bk varies).
    Actually: C[0,k] = a*bk for k=0..7, that's 8 products. C[1,k] = b*bk for k=0..7, that's 8 products. Total 16.
    Can we do better? a*bk + b*bk = (a+b)*bk, but we need them separately. No savings. Rank = 16.
    
    M(2,2,8): A is 2x2, B is 2x8, C is 2x8.
    Schoolbook: 2*2*8 = 32.
    Using Strassen for each 2x2x2 block (4 blocks): each has rank 7, total 28.
    
    So total: 28 + 16 = 44.
    
    Let me construct this:
    
    M(2,2,8): Split B's 8 columns into 4 groups of 2.
    For each group (columns 2g, 2g+1):
    A = [a11 a12; a21 a22], B_g = [b11 b12; b21 b22], C_g = [c11 c12; c21 c22]
    
    Strassen:
    p1 = (a11+a22)(b11+b22)
    p2 = (a21+a22)b11
    p3 = a11(b22-b12)  -- wait, standard Strassen:
    p1 = (a11+a22)(b11+b22)
    p2 = (a21+a22)b11
    p3 = a11(b22-b12)  -- hmm, let me use the standard form
    p3 = a11(b22-b12)  -- no
    
    Standard Strassen for C = A*B where A,B are 2x2:
    C = [a11 a12; a21 a22] * [b11 b12; b21 b22]
    
    m1 = (a11 + a22)(b11 + b22)
    m2 = (a21 + a22)b11
    m3 = a11(b22 - b12)
    m4 = a22(b11 - b21)
    m5 = (a11 + a12)b22
    m6 = (a21 - a11)(b11 + b12)
    m7 = (a12 - a22)(b21 + b22)
    
    C11 = m1 - m4 + m6
    C12 = m3 + m5
    C21 = m2 + m4
    C22 = m1 - m3 + m7
    
    Wait, let me double-check. The standard Strassen formulas:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b22-b12)
    M4 = a22(b11-b21)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C11 = M1 - M4 + M6
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M7
    
    Yes, that's correct.
    
    So for each 2x2x2 block, rank 7.
    
    Now let me code this up for the full 2x3x8 problem.
    
    Plan:
    1. Split A (2x3) into A1 (2x2, cols 0,1) and A2 (2x1, col 2).
    2. Split B (3x8) into B1 (2x8, rows 0,1) and B2 (1x8, row 2).
    3. C = A1*B1 + A2*B2.
    4. A2*B2: rank 16. For k=0..7, col k of C:
       C[0,k] += A[0,2]*B[2,k]
       C[1,k] += A[1,2]*B[2,k]
       Each is one product, so 16 terms.
    5. A1*B1: 2x2 * 2x8. Split into 4 Strassen blocks of 2x2x2. Rank 7*4=28.
    
    Total rank: 16 + 28 = 44.
    
    Let me implement this.
    """
    n, m, p = 2, 3, 8
    
    # We'll build U, V, W with rank 44
    # Part 1: A2*B2 (rank 16)
    # Part 2: A1*B1 via Strassen (rank 28)
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2
    # A2 is column 2 of A: A[0,2] at index 0*3+2=2, A[1,2] at index 1*3+2=5
    # B2 is row 2 of B: B[2,k] at index 2*8+k = 16+k
    # C output: C[i,k] at index k*2+i
    for k in range(8):
        # C[0,k] += A[0,2] * B[2,k]
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # C[1,k] += A[1,2] * B[2,k]
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen
    # A1 is 2x2: A[0,0]@0, A[0,1]@1, A[1,0]@3, A[1,1]@4
    # B1 is 2x8: rows 0,1. B[0,k]@k, B[1,k]@8+k
    # C is 2x8: C[i,k] at k*2+i
    #
    # Split into 4 blocks of 2 columns each.
    # Block g (g=0..3) handles columns 2g, 2g+1 of B1.
    # B_g = [B[0,2g], B[0,2g+1]; B[1,2g], B[1,2g+1]]
    
    for g in range(4):
        b11 = 0 + 2 * g      # B[0, 2g]
        b12 = 0 + 2 * g + 1  # B[0, 2g+1]
        b21 = 8 + 2 * g      # B[1, 2g]
        b22 = 8 + 2 * g + 1  # B[1, 2g+1]
        
        a11 = 0  # A[0,0]
        a12 = 1  # A[0,1]
        a21 = 3  # A[1,0]
        a22 = 4  # A[1,1]
        
        # Strassen products:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b22-b12)
        # M4 = a22(b11-b21)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # M1
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        # C11 += M1, C22 += M1
        W[(2 * g) * 2 + 0][col] = 1      # C[0, 2g]
        W[(2 * g + 1) * 2 + 1][col] = 1  # C[1, 2g+1]
        col += 1
        
        # M2
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        # C21 += M2
        W[(2 * g) * 2 + 1][col] = 1  # C[1, 2g]
        col += 1
        
        # M3
        U[a11][col] = 1
        V[b22][col] = 1
        V[b12][col] = -1
        # C12 += M3, C22 -= M3
        W[(2 * g + 1) * 2 + 0][col] = 1   # C[0, 2g+1]
        W[(2 * g + 1) * 2 + 1][col] = -1  # C[1, 2g+1]
        col += 1
        
        # M4
        U[a22][col] = 1
        V[b11][col] = 1
        V[b21][col] = -1
        # C11 -= M4, C21 += M4
        W[(2 * g) * 2 + 0][col] = -1  # C[0, 2g]
        W[(2 * g) * 2 + 1][col] = 1   # C[1, 2g]
        col += 1
        
        # M5
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        # C12 += M5
        W[(2 * g + 1) * 2 + 0][col] = 1  # C[0, 2g+1]
        col += 1
        
        # M6
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        # C11 += M6
        W[(2 * g) * 2 + 0][col] = 1  # C[0, 2g]
        col += 1
        
        # M7
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        # C22 += M7
        W[(2 * g + 1) * 2 + 1][col] = 1  # C[1, 2g+1]
        col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # The decomposition should satisfy:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (the tensor entry)
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 if the triple corresponds to a valid multiplication term
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal
    # the (i,j,k)-entry of the matmul tensor.
    
    # For standard matmul: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # Wait, no. The matmul tensor is:
    # T[i, j, k] where i indexes row of A, j indexes col of A / row of B, k indexes col of B
    # T[i,j,k] = 1, and the contraction is: sum_{i,j,k} A[i,j] * B[j,k] * T[i,j,k] = (AB)[i,k]
    
    # So the tensor entry T[i,j,k] = 1 for all i,j,k in range.
    # The decomposition must satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 44)
    try:
        U, V, W = _build_decomp(n, m, p)
        # Quick verification
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
