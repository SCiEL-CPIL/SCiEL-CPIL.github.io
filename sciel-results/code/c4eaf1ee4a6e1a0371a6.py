"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-33 decomposition for 2x3 x 3x6 matrix multiplication.
    
    Strategy: Split the 3x3 inner dimension into a 2x2 block and a 1x1 block.
    - The 2x2 block uses Strassen's algorithm (rank 7) to multiply the 2x2 submatrices.
    - The 1x1 block uses schoolbook multiplication (rank 1) for the remaining column.
    - This yields a total rank of 7 + 1 = 8 for the core 2x2x2 structure? No.
    
    Wait, the dimensions are 2x3 and 3x6.
    A is 2x3, B is 3x6.
    Let's partition A into A_L (2x2) and A_R (2x1).
    Let's partition B into B_L (3x6? No, B is 3x6, so B_L is 2x6 and B_R is 1x6? No, B is 3x6, so rows correspond to j.
    A is 2x3 (rows i, cols j).
    B is 3x6 (rows j, cols k).
    
    Partition j into j1, j2 (size 2) and j3 (size 1).
    A = [A_1 | A_2] where A_1 is 2x2, A_2 is 2x1.
    B = [B_1; B_2] where B_1 is 2x6, B_2 is 1x6.
    
    C = A B = A_1 B_1 + A_2 B_2.
    
    Term 1: A_1 B_1. This is a 2x2 times 2x6 multiplication.
    We can view this as 3 separate 2x2 multiplications if we split the 6 columns into three pairs?
    Let B_1 = [B_{1,1} B_{1,2} B_{1,3}] where each B_{1,l} is 2x2.
    Then C_{1} = A_1 B_1 = [A_1 B_{1,1}, A_1 B_{1,2}, A_1 B_{1,3}].
    Each A_1 B_{1,l} is a 2x2 multiplication.
    Using Strassen (rank 7), we need 7 scalar multiplications for each 2x2 block.
    Since A_1 is shared, we can precompute the 7 Strassen inputs for A_1 once, and then multiply by the 7 Strassen inputs of each B_{1,l}.
    Wait, the rank of the tensor is the number of scalar multiplications.
    If we compute A_1 B_{1,1}, we need 7 mults.
    A_1 B_{1,2}, we need 7 mults.
    A_1 B_{1,3}, we need 7 mults.
    Total for this part: 21.
    
    Term 2: A_2 B_2. A_2 is 2x1, B_2 is 1x6.
    C_2 = A_2 B_2.
    (A_2 B_2)_{i,k} = A_{i,2} * B_{2,k}.
    This is 2x6 entries, each is a product of one A entry and one B entry.
    However, we can factor this.
    C_2 = A_2 B_2.
    This is an outer product of a 2x1 vector and a 1x6 vector.
    Rank of this tensor part?
    The tensor for 2x1 x 1x6 is rank 2?
    Let's check.
    C_{i,k} = a_i b_k.
    Tensor T = sum_{i,k} a_i b_k e_{ik}.
    This is a rank-1 tensor? No, it's a vector of products.
    The bilinear map is (a, b) -> a b^T.
    The dimension of the output is 12.
    Can we do it with fewer than 12 multiplications?
    The rank of the matrix multiplication tensor for m=2, n=1, p=6 is 2*6=12?
    Actually, for a 2x1 matrix and 1x6 matrix, the product is 2x6.
    Each entry c_{i,k} = a_i b_k.
    There is no sharing between different i or k.
    So we need 12 multiplications?
    Wait, if a = [a1, a2]^T and b = [b1, ..., b6].
    c11 = a1 b1
    c12 = a1 b2
    ...
    c21 = a2 b1
    ...
    Is there a way to share?
    No, because a1 and a2 are independent, and b1..b6 are independent.
    The tensor is a direct sum of 12 rank-1 tensors?
    Actually, the rank of the 2x1x6 multiplication tensor is 12.
    
    So total rank = 21 (Strassen for 3 blocks of 2x2) + 12 (Schoolbook for 2x1x6) = 33.
    
    This matches the "Reduces rank from 36 to 33" observation.
    
    Let's construct the matrices U, V, W.
    
    Dimensions:
    n=2, m=3, p=6.
    U is (n*m) x rank = 6 x 33.
    V is (m*p) x rank = 18 x 33.
    W is (p*n) x rank = 12 x 33.
    
    Indices:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    
    We have 33 columns in U, V, W.
    
    Part 1: Strassen for A_1 B_1.
    A_1 is 2x2 (i=0,1; j=0,1).
    B_1 is 2x6 (j=0,1; k=0..5).
    We split B_1 into three 2x2 blocks: B_{1,0} (k=0,1), B_{1,1} (k=2,3), B_{1,2} (k=4,5).
    
    For each block l in 0,1,2:
    We use 7 Strassen multiplications.
    
    Strassen 2x2:
    Let A_sub = [[a00, a01], [a10, a11]] (here aij are from A_1, so a_{i, j} with j in 0,1)
    Let B_sub = [[b00, b01], [b10, b11]] (here b_{j, k} with j in 0,1, k in 2l, 2l+1)
    
    The 7 products are:
    m1 = (a00 + a11) * (b00 + b11)
    m2 = (a10 + a11) * b00
    m3 = a00 * (b10 - b11)
    m4 = a11 * (b01 + b11)
    m5 = (a00 + a01) * b11
    m6 = (a10 - a00) * (b00 + b01)
    m7 = (a01 - a11) * (b10 + b11)
    
    We need to define U, V, W for each of these 7 terms for each of the 3 blocks.
    That's 3 * 7 = 21 columns.
    
    Let's define the indices for U, V, W for the Strassen part.
    For block l (0,1,2), the k indices are 2l and 2l+1.
    
    For term s (0..6) in block l:
    Column index c = l * 7 + s.
    
    We need to extract the linear forms for A and B.
    
    A linear forms (for U):
    The A part involves only j=0,1.
    U[i*m + j, c] is non-zero only for j=0,1.
    
    Let's list the coefficients for a00, a01, a10, a11 for each Strassen term.
    a00 = A[0,0], a01 = A[0,1], a10 = A[1,0], a11 = A[1,1].
    
    Term 0 (m1): A-part: a00 + a11.
    U[0*3+0, c] = 1
    U[1*3+1, c] = 1
    
    Term 1 (m2): A-part: a10 + a11.
    U[1*3+0, c] = 1
    U[1*3+1, c] = 1
    
    Term 2 (m3): A-part: a00.
    U[0*3+0, c] = 1
    
    Term 3 (m4): A-part: a11.
    U[1*3+1, c] = 1
    
    Term 4 (m5): A-part: a00 + a01.
    U[0*3+0, c] = 1
    U[0*3+1, c] = 1
    
    Term 5 (m6): A-part: a10 - a00.
    U[1*3+0, c] = 1
    U[0*3+0, c] = -1
    
    Term 6 (m7): A-part: a01 - a11.
    U[0*3+1, c] = 1
    U[1*3+1, c] = -1
    
    Now B linear forms (for V):
    V[j*p + k, c] is non-zero only for j=0,1.
    b00 = B[0, 2l], b01 = B[0, 2l+1], b10 = B[1, 2l], b11 = B[1, 2l+1].
    
    Term 0 (m1): B-part: b00 + b11.
    V[0*6 + 2l, c] = 1
    V[1*6 + 2l+1, c] = 1
    
    Term 1 (m2): B-part: b00.
    V[0*6 + 2l, c] = 1
    
    Term 2 (m3): B-part: b10 - b11.
    V[1*6 + 2l, c] = 1
    V[1*6 + 2l+1, c] = -1
    
    Term 3 (m4): B-part: b01 + b11.
    V[0*6 + 2l+1, c] = 1
    V[1*6 + 2l+1, c] = 1
    
    Term 4 (m5): B-part: b11.
    V[1*6 + 2l+1, c] = 1
    
    Term 5 (m6): B-part: b00 + b01.
    V[0*6 + 2l, c] = 1
    V[0*6 + 2l+1, c] = 1
    
    Term 6 (m7): B-part: b10 + b11.
    V[1*6 + 2l, c] = 1
    V[1*6 + 2l+1, c] = 1
    
    Now W linear forms (for W):
    W[k*n + i, c].
    We need to reconstruct C[i, k] for i=0,1 and k=2l, 2l+1.
    
    The Strassen output for 2x2 is:
    c00 = m1 + m4 - m5 - m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    
    So for block l:
    k0 = 2l, k1 = 2l+1.
    
    c[0, k0] = m1 + m4 - m5 - m7
    c[0, k1] = m3 + m5
    c[1, k0] = m2 + m4
    c[1, k1] = m1 - m2 + m3 + m6
    
    W entries for column c (which corresponds to term s in block l):
    We add the coefficient of m_s to the appropriate W[k*n + i, c].
    
    Term 0 (m1):
    W[k0*n + 0, c] += 1  (from c00)
    W[k1*n + 1, c] += 1  (from c11)
    
    Term 1 (m2):
    W[k0*n + 1, c] += 1  (from c10)
    W[k1*n + 1, c] += -1 (from c11)
    
    Term 2 (m3):
    W[k1*n + 0, c] += 1  (from c01)
    W[k1*n + 1, c] += 1  (from c11)
    
    Term 3 (m4):
    W[k0*n + 0, c] += 1  (from c00)
    W[k0*n + 1, c] += 1  (from c10)
    
    Term 4 (m5):
    W[k0*n + 0, c] += -1 (from c00)
    W[k1*n + 0, c] += 1  (from c01)
    
    Term 5 (m6):
    W[k1*n + 1, c] += 1  (from c11)
    
    Term 6 (m7):
    W[k0*n + 0, c] += -1 (from c00)
    
    Part 2: Schoolbook for A_2 B_2.
    A_2 is 2x1 (j=2).
    B_2 is 1x6 (j=2, k=0..5).
    C_2 = A_2 B_2.
    c[i, k] = A[i, 2] * B[2, k].
    
    This requires 12 multiplications.
    For each i in 0,1 and k in 0..5:
    Column c = 21 + i*6 + k.
    U[i*3 + 2, c] = 1
    V[2*6 + k, c] = 1
    W[k*2 + i, c] = 1
    
    Total rank = 21 + 12 = 33.
    
    Let's implement this.
    """
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for j=0,1
    for l in range(3):
        k0 = 2 * l
        k1 = 2 * l + 1
        for s in range(7):
            c = l * 7 + s
            # U
            if s == 0:
                U[0*3+0][c] = 1
                U[1*3+1][c] = 1
            elif s == 1:
                U[1*3+0][c] = 1
                U[1*3+1][c] = 1
            elif s == 2:
                U[0*3+0][c] = 1
            elif s == 3:
                U[1*3+1][c] = 1
            elif s == 4:
                U[0*3+0][c] = 1
                U[0*3+1][c] = 1
            elif s == 5:
                U[1*3+0][c] = 1
                U[0*3+0][c] = -1
            elif s == 6:
                U[0*3+1][c] = 1
                U[1*3+1][c] = -1
            
            # V
            if s == 0:
                V[0*6+k0][c] = 1
                V[1*6+k1][c] = 1
            elif s == 1:
                V[0*6+k0][c] = 1
            elif s == 2:
                V[1*6+k0][c] = 1
                V[1*6+k1][c] = -1
            elif s == 3:
                V[0*6+k1][c] = 1
                V[1*6+k1][c] = 1
            elif s == 4:
                V[1*6+k1][c] = 1
            elif s == 5:
                V[0*6+k0][c] = 1
                V[0*6+k1][c] = 1
            elif s == 6:
                V[1*6+k0][c] = 1
                V[1*6+k1][c] = 1
            
            # W
            if s == 0:
                W[k0*n+0][c] += 1
                W[k1*n+1][c] += 1
            elif s == 1:
                W[k0*n+1][c] += 1
                W[k1*n+1][c] += -1
            elif s == 2:
                W[k1*n+0][c] += 1
                W[k1*n+1][c] += 1
            elif s == 3:
                W[k0*n+0][c] += 1
                W[k0*n+1][c] += 1
            elif s == 4:
                W[k0*n+0][c] += -1
                W[k1*n+0][c] += 1
            elif s == 5:
                W[k1*n+1][c] += 1
            elif s == 6:
                W[k0*n+0][c] += -1
    
    # Part 2: Schoolbook for j=2
    base_c = 21
    for i in range(2):
        for k in range(6):
            c = base_c + i * 6 + k
            U[i*3+2][c] = 1
            V[2*6+k][c] = 1
            W[k*2+i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
