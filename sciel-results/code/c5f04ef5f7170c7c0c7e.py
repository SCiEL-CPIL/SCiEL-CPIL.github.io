"""Program for matrix multiplication 6x6x6 using Kronecker product of Strassen and schoolbook."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 decomposition for 6x6x6 matrix multiplication by taking
    the Kronecker product of the rank-7 Strassen decomposition for 2x2 matrices
    and the rank-27 schoolbook decomposition for 3x3 matrices.
    
    The tensor product structure means:
    - U has shape (n*m, rank) = (36, 189)
    - V has shape (m*p, rank) = (36, 189)  
    - W has shape (p*n, rank) = (36, 189)
    
    For the Kronecker product of two matrix multiplication tensors:
    If tensor A has dimensions (n1, m1, p1) with rank r1 and factors U1, V1, W1
    and tensor B has dimensions (n2, m2, p2) with rank r2 and factors U2, V2, W2
    then the Kronecker product tensor has dimensions (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    
    The factors are constructed as:
    U_combined[i1*n2 + i2, c1*r2 + c2] = U1[i1, c1] * U2[i2, c2]
    V_combined[j1*m2 + j2, c1*r2 + c2] = V1[j1, c2] * V2[j2, c2]  -- wait, need to be careful
    W_combined[k1*n2 + k2, c1*r2 + c2] = W1[k1, c1] * W2[k2, c2]
    
    Actually, let me think more carefully about the Kronecker product of tensors.
    
    For matrix multiplication, the tensor T satisfies:
    (A * B)[i, k] = sum_{j} A[i, j] * B[j, k]
    
    The decomposition is:
    T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, i]  (with the trace convention W[k*n+i, c])
    
    Wait, let me re-read the convention. The output factor follows W[k*n+i, r].
    So W has shape (p*n, rank), and W[k*n + i, c] is the entry.
    
    For the Kronecker product of two matrix multiplications:
    Let A be n1 x m1, B be m1 x p1, result C1 = A*B is n1 x p1.
    Let A' be n2 x m2, B' be m2 x p2, result C2 = A'*B' is n2 x p2.
    
    The block matrix A_block is (n1*n2) x (m1*m2) formed by Kronecker product structure.
    Actually, for the Kronecker product of matrix multiplication tensors, we're computing
    the multiplication of block matrices where each block is itself a matrix product.
    
    More precisely, if we have:
    - Tensor 1: multiplies (n1 x m1) by (m1 x p1) -> (n1 x p1), with factors U1 (n1*m1, r1), V1 (m1*p1, r1), W1 (p1*n1, r1)
    - Tensor 2: multiplies (n2 x m2) by (m2 x p2) -> (n2 x p2), with factors U2 (n2*m2, r2), V2 (m2*p2, r2), W2 (p2*n2, r2)
    
    Then the Kronecker product tensor multiplies (n1*n2 x m1*m2) by (m1*m2 x p1*p2) -> (n1*n2 x p1*p2)
    with rank r1*r2.
    
    The factors are:
    U_comb has shape (n1*n2 * m1*m2, r1*r2)
    V_comb has shape (m1*m2 * p1*p2, r1*r2)
    W_comb has shape (p1*p2 * n1*n2, r1*r2)
    
    For row index of U_comb: the first matrix is (n1*n2) x (m1*m2).
    Row index = i1 * (n2*m2) + i2 * m2 + j1 * m2 + j2 ... 
    
    Hmm, let me think about this differently. Let's use the standard Kronecker product indexing.
    
    For a matrix of size (n1*n2) x (m1*m2), we can index rows by (i1, i2) where i1 in [0,n1), i2 in [0,n2)
    and columns by (j1, j2) where j1 in [0,m1), j2 in [0,m2).
    
    The row index in the flattened matrix is i1 * (n2 * m2) + i2 * m2 + j2? No, that's not right.
    
    Actually, for a matrix A of size (n1*n2) x (m1*m2), if we think of it as a block matrix where each block
    is n2 x m2, then the row index is i1 * n2 + i2 and column index is j1 * m2 + j2.
    So the flattened row index for U (which has shape (n*m, rank)) would be:
    For U1: row index i1*m1 + j1
    For the combined: row index (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
    
    Wait, U has shape (n*m, rank). So for the combined tensor with n = n1*n2 and m = m1*m2:
    U_comb has shape (n1*n2 * m1*m2, r1*r2)
    
    Row index in U_comb: (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
    Column index in U_comb: c1 * r2 + c2
    
    U_comb[(i1*n2 + i2) * (m1*m2) + (j1*m2 + j2), c1 * r2 + c2] = U1[i1*m1 + j1, c1] * U2[i2*m2 + j2, c2]
    
    Similarly for V:
    V_comb has shape (m1*m2 * p1*p2, r1*r2)
    Row index: (j1*p2 + j2) * (p1*p2) + (k1*p2 + k2) ... wait no.
    
    V has shape (m*p, rank). For the combined: m = m1*m2, p = p1*p2.
    Row index: (j1*p2 + j2) * (p1*p2) + ... hmm, this is getting confusing.
    
    Let me restart with clearer indexing.
    
    For the combined tensor:
    n = n1 * n2, m = m1 * m2, p = p1 * p2
    
    U has shape (n*m, rank) = (n1*n2*m1*m2, r1*r2)
    V has shape (m*p, rank) = (m1*m2*p1*p2, r1*r2)
    W has shape (p*n, rank) = (p1*p2*n1*n2, r1*r2)
    
    For U: the row corresponds to a pair (i, j) where i is in [0, n) and j is in [0, m).
    i = i1 * n2 + i2, j = j1 * m2 + j2
    Row index = i * m + j = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
    Column index = c1 * r2 + c2
    U[row, col] = U1[i1*m1 + j1, c1] * U2[i2*m2 + j2, c2]
    
    For V: the row corresponds to a pair (j, k) where j is in [0, m) and k is in [0, p).
    j = j1 * m2 + j2, k = k1 * p2 + k2
    Row index = j * p + k = (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
    Column index = c1 * r2 + c2
    V[row, col] = V1[j1*p1 + k1, c1] * V2[j2*p2 + k2, c2]
    
    For W: the row corresponds to a pair (k, i) where k is in [0, p) and i is in [0, n).
    k = k1 * p2 + k2, i = i1 * n2 + i2
    Row index = k * n + i = (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
    Column index = c1 * r2 + c2
    W[row, col] = W1[k1*n1 + i1, c1] * W2[k2*n2 + i2, c2]
    
    Now let's define the Strassen decomposition for 2x2 matrices (rank 7).
    
    Strassen's 7 products for 2x2 matrix multiplication:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    p1 = (a+d)*(e+h)
    p2 = (c+d)*e
    p3 = a*(f-h)
    p4 = d*(g-e)
    p5 = (a+b)*h
    p6 = (c-a)*(e+f)
    p7 = (b-d)*(g+h)
    
    C = [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    So the decomposition has:
    U: shape (2*2, 7) = (4, 7), where U[i*2+j, c] corresponds to A[i,j]
    V: shape (2*2, 7) = (4, 7), where V[j*2+k, c] corresponds to B[j,k]
    W: shape (2*2, 7) = (4, 7), where W[k*2+i, c] corresponds to C[i,k]
    
    Let me write out the factors:
    
    For c=0 (p1 = (a+d)*(e+h)):
    U: a=1, d=1 -> U[0,0]=1, U[3,0]=1 (since i=0,j=0 for a; i=1,j=1 for d)
    V: e=1, h=1 -> V[0,0]=1, V[3,0]=1 (since j=0,k=0 for e; j=1,k=1 for h)
    W: C[0,0] gets +1, C[1,1] gets +1 -> W[0*2+0,0]=1, W[1*2+1,0]=1
    
    Wait, W[k*n+i, c] where n=2. So W[k*2+i, c].
    C[i,k] = sum_j A[i,j]*B[j,k]
    The contribution to C[i,k] from product c is W[k*2+i, c].
    
    For p1 = (a+d)*(e+h):
    This contributes to:
    C[0,0] = p1 + ... so W[0*2+0, 0] = 1
    C[1,1] = p1 + ... so W[1*2+1, 0] = 1
    
    For c=1 (p2 = (c+d)*e):
    U: c=1 (A[1,0]), d=1 (A[1,1]) -> U[1*2+0, 1]=1, U[1*2+1, 1]=1
    V: e=1 (B[0,0]) -> V[0*2+0, 1]=1
    W: C[1,0] gets +1 -> W[0*2+1, 1]=1
    
    For c=2 (p3 = a*(f-h)):
    U: a=1 (A[0,0]) -> U[0*2+0, 2]=1
    V: f=1 (B[0,1]), h=-1 (B[1,1]) -> V[0*2+1, 2]=1, V[1*2+1, 2]=-1
    W: C[0,1] gets +1 -> W[1*2+0, 2]=1
    
    For c=3 (p4 = d*(g-e)):
    U: d=1 (A[1,1]) -> U[1*2+1, 3]=1
    V: g=1 (B[1,0]), e=-1 (B[0,0]) -> V[1*2+0, 3]=1, V[0*2+0, 3]=-1
    W: C[0,0] gets +1, C[1,0] gets +1 -> W[0*2+0, 3]=1, W[0*2+1, 3]=1
    
    For c=4 (p5 = (a+b)*h):
    U: a=1 (A[0,0]), b=1 (A[0,1]) -> U[0*2+0, 4]=1, U[0*2+1, 4]=1
    V: h=1 (B[1,1]) -> V[1*2+1, 4]=1
    W: C[0,0] gets -1, C[0,1] gets +1 -> W[0*2+0, 4]=-1, W[1*2+0, 4]=1
    
    For c=5 (p6 = (c-a)*(e+f)):
    U: c=1 (A[1,0]), a=-1 (A[0,0]) -> U[1*2+0, 5]=1, U[0*2+0, 5]=-1
    V: e=1 (B[0,0]), f=1 (B[0,1]) -> V[0*2+0, 5]=1, V[0*2+1, 5]=1
    W: C[1,1] gets +1 -> W[1*2+1, 5]=1
    
    For c=6 (p7 = (b-d)*(g+h)):
    U: b=1 (A[0,1]), d=-1 (A[1,1]) -> U[0*2+1, 6]=1, U[1*2+1, 6]=-1
    V: g=1 (B[1,0]), h=1 (B[1,1]) -> V[1*2+0, 6]=1, V[1*2+1, 6]=1
    W: C[0,0] gets +1 -> W[0*2+0, 6]=1
    
    Let me verify this by checking C[0,0] = p1 + p4 - p5 + p7:
    W[0*2+0, 0]=1, W[0*2+0, 3]=1, W[0*2+0, 4]=-1, W[0*2+0, 6]=1. Sum = 1+1-1+1 = 2? That's not right.
    
    Hmm, I think I made an error. Let me re-derive.
    
    C[0,0] = a*e + b*g
    Using Strassen:
    p1 = (a+d)(e+h) = ae + ah + de + dh
    p4 = d(g-e) = dg - de
    p5 = (a+b)h = ah + bh
    p7 = (b-d)(g+h) = bg + bh - dg - dh
    
    p1 + p4 - p5 + p7 = (ae + ah + de + dh) + (dg - de) - (ah + bh) + (bg + bh - dg - dh)
    = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    = ae + bg
    Yes! That's correct.
    
    So W[0*2+0, c] for c=0,3,4,6 should be 1,1,-1,1. Let me re-check my W values.
    
    For c=4 (p5 = (a+b)*h):
    C[0,0] gets -p5, so W[0*2+0, 4] = -1. Correct.
    C[0,1] gets +p5, so W[1*2+0, 4] = 1. Correct.
    
    OK so my W values were correct. Let me re-verify C[0,0]:
    Contributions from c=0: W[0,0]=1
    Contributions from c=3: W[0,3]=1
    Contributions from c=4: W[0,4]=-1
    Contributions from c=6: W[0,6]=1
    Sum = 1 + 1 - 1 + 1 = 2? 
    
    But we need C[0,0] = ae + bg. The decomposition should give:
    C[0,0] = sum_c U[0, c] * V[?, c] * W[0, c] ... 
    
    Wait, I'm confusing myself. The decomposition is:
    C[i,k] = sum_j A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' V[j*p+k', c] * B[j,k']) ... 
    
    No, that's not right either. Let me think about this more carefully.
    
    The tensor decomposition says:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    where T[i, j, k] = 1 if this corresponds to a valid term in the matrix multiplication, i.e., always 1 for the standard tensor.
    
    Actually, the matrix multiplication tensor is:
    (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    
    So the tensor T has T[i, j, k] = 1 for all i, j, k (it's the all-ones tensor in the sense that each "slice" is the identity).
    
    Wait no. The tensor is defined such that:
    (A * B)[i, k] = sum_{j} A[i, j] * B[j, k] * T[i, j, k]
    
    And T[i, j, k] = 1 for all i, j, k. So the tensor is just the all-ones tensor.
    
    The decomposition is:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So for the Strassen case with n=m=p=2:
    T[i, j, k] = 1 = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    
    Let me verify for i=0, j=0, k=0:
    sum_c U[0, c] * V[0, c] * W[0, c]
    c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    c=1: U[0,1]=0, ... -> 0
    c=2: U[0,2]=1, V[0,2]=0, ... -> 0
    c=3: U[0,3]=0, ... -> 0
    c=4: U[0,4]=1, V[0,4]=0, ... -> 0
    c=5: U[0,5]=-1, V[0,5]=1, W[0,5]=0 -> 0
    c=6: U[0,6]=0, ... -> 0
    Sum = 1. Correct!
    
    For i=0, j=1, k=0:
    sum_c U[1, c] * V[2, c] * W[0, c]
    c=0: U[1,0]=0, ... -> 0
    c=1: U[1,1]=0, ... -> 0
    c=2: U[1,2]=0, ... -> 0
    c=3: U[1,3]=0, ... -> 0
    c=4: U[1,4]=1, V[2,4]=0, ... -> 0
    c=5: U[1,5]=0, ... -> 0
    c=6: U[1,6]=1, V[2,6]=1, W[0,6]=1 -> 1
    Sum = 1. Correct!
    
    OK so my Strassen factors look correct. Let me now code this up properly.
    
    """
    # Strassen decomposition for 2x2 matrices (rank 7)
    n1, m1, p1 = 2, 2, 2
    r1 = 7
    
    U1 = [[0]*r1 for _ in range(n1*m1)]
    V1 = [[0]*r1 for _ in range(m1*p1)]
    W1 = [[0]*r1 for _ in range(p1*n1)]
    
    # c=0: p1 = (a+d)*(e+h)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # c=1: p2 = (c+d)*e
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1
    
    # c=2: p3 = a*(f-h)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1
    
    # c=3: p4 = d*(g-e)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # c=4: p5 = (a+b)*h
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # c=5: p6 = (c-a)*(e+f)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # c=6: p7 = (b-d)*(g+h)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # Schoolbook decomposition for 3x3 matrices (rank 27)
    n2, m2, p2 = 3, 3, 3
    r2 = 27
    
    U2 = [[0]*r2 for _ in range(n2*m2)]
    V2 = [[0]*r2 for _ in range(m2*p2)]
    W2 = [[0]*r2 for _ in range(p2*n2)]
    
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i*m2 + j][col] = 1
                V2[j*p2 + k][col] = 1
                W2[k*n2 + i][col] = 1
                col += 1
    
    # Now construct the Kronecker product decomposition
    n, m, p = n1*n2, m1*m2, p1*p2  # 6, 6, 6
    rank = r1 * r2  # 189
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for c1 in range(r1):
                    u1_val = U1[i1*m1 + j1][c1]
                    v1_val = V1[j1*p1 + k1][c1]
                    w1_val = W1[k1*n1 + i1][c1]
                    
                    if u1_val == 0 and v1_val == 0 and w1_val == 0:
                        continue
                    
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                for c2 in range(r2):
                                    u2_val = U2[i2*m2 + j2][c2]
                                    v2_val = V2[j2*p2 + k2][c2]
                                    w2_val = W2[k2*n2 + i2][c2]
                                    
                                    if u2_val == 0 and v2_val == 0 and w2_val == 0:
                                        continue
                                    
                                    # Row indices
                                    u_row = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                                    v_row = (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
                                    w_row = (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
                                    col_idx = c1 * r2 + c2
                                    
                                    u_val = u1_val * u2_val
                                    v_val = v1_val * v2_val
                                    w_val = w1_val * w2_val
                                    
                                    if u_val != 0:
                                        U[u_row][col_idx] = u_val
                                    if v_val != 0:
                                        V[v_row][col_idx] = v_val
                                    if w_val != 0:
                                        W[w_row][col_idx] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
