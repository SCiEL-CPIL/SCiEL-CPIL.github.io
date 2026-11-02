"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Use Strassen's algorithm to decompose the 2x4x8 matrix multiplication.
    
    The 2x4x8 problem can be viewed as 8 independent 2x2x2 sub-problems
    (since 4 = 2*2 and 8 = 2*4, but actually we need to be more careful).
    
    Actually, let's think about this differently. A 2x4 matrix times a 4x8 matrix
    gives a 2x8 result. We can decompose this into 8 independent 2x2x2 problems
    by partitioning the 4x8 matrix into 2x2 blocks along the columns.
    
    Wait, that's not quite right either. Let me think more carefully.
    
    A (2x4) * (4x8) = (2x8) matrix. We can partition the input A (2x4) into two 
    2x2 blocks: A = [A1, A2] where A1, A2 are 2x2.
    We can partition the input B (4x8) into four 2x2 blocks: B = [B11, B12; B21, B22]
    where each is 2x2.
    
    Then C = A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    So we have 4 independent 2x2x2 matrix multiplications, each of which can be done
    with Strassen's algorithm (rank 7 instead of 8).
    
    Total rank: 4 * 7 = 28
    
    Wait, but the indices need to be handled correctly. Let me work through the indices.
    
    A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    B is 4x8: B[j,k] for j in {0,1,2,3}, k in {0,...,7}
    C is 2x8: C[i,k] for i in {0,1}, k in {0,...,7}
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can group j into pairs: j = 2*l + r where l in {0,1}, r in {0,1}
    
    C[i,k] = sum_{l=0}^{1} sum_{r=0}^{1} A[i, 2*l+r] * B[2*l+r, k]
    
    For a fixed k, let's partition k into groups of 4: k = 4*m + s where m in {0,1}, s in {0,1,2,3}
    
    Actually, let me just think of it as: we have 8 independent 2x2x2 problems.
    
    For each output column group, we have a 2x2 * 2x2 = 2x2 problem.
    
    Let me define:
    - A_left = A[:, 0:2] (2x2)
    - A_right = A[:, 2:4] (2x2)
    - B is 4x8, which we can view as 4 blocks of 2x2: B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4], etc.
    
    Actually, the cleanest way: C[i,k] depends on A[i,0..3] and B[0..3,k].
    
    For fixed i and k, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    This is a dot product of length 4. We can split it into two dot products of length 2:
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    The first term is a 2x2 matrix multiplication (for all i in {0,1} and for a fixed pair of k values).
    
    Let me define k pairs: for k in {0,1}, {2,3}, {4,5}, {6,7}.
    
    For each k pair (k0, k1), and for the two A columns (0,1) or (2,3):
    We have a 2x2 * 2x2 = 2x2 problem.
    
    Specifically:
    - For k pair (0,1): C[0:2, 0:2] = A[:,0:2] * B[0:2, 0:2]
    - For k pair (2,3): C[0:2, 2:4] = A[:,0:2] * B[0:2, 2:4]
    - For k pair (4,5): C[0:2, 4:6] = A[:,0:2] * B[0:2, 4:6]
    - For k pair (6,7): C[0:2, 6:8] = A[:,0:2] * B[0:2, 6:8]
    
    Wait, that's only using A[:,0:2]. What about A[:,2:4]?
    
    Ah, I see my error. Let me redo:
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    For fixed k, this is a sum over 4 terms. If I group by the B rows:
    - Rows 0,1 of B: contribute A[i,0:2] * B[0:2,k]
    - Rows 2,3 of B: contribute A[i,2:4] * B[2:4,k]
    
    So for each k, C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    Now, if I fix a pair of k values, say k in {k0, k1}, and consider all i in {0,1}:
    The first part is a 2x2 * 2x2 multiplication: A[:,0:2] * B[0:2, {k0,k1}]
    The second part is another 2x2 * 2x2 multiplication: A[:,2:4] * B[2:4, {k0,k1}]
    
    So for each of the 4 k-pairs, I have 2 independent 2x2x2 problems.
    Total: 4 * 2 = 8 independent 2x2x2 problems.
    
    Each 2x2x2 problem can be done with Strassen (rank 7).
    Total rank: 8 * 7 = 56.
    
    This matches the knowledge from the run! Let me implement this.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # We'll use Strassen for each 2x2x2 block.
    # Strassen's 7 products for 2x2x2:
    # Given A (2x2), B (2x2):
    #   A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    #   C = [[ae+cg, af+dh],[ce+dg, cf+gh]]
    # Strassen:
    #   p1 = (a+d)(e+h)
    #   p2 = (c+d)e
    #   p3 = a(f-h)
    #   p4 = d(g-e)
    #   p5 = (a+b)h
    #   p6 = (c-a)(e+f)
    #   p7 = (b-d)(g+h)
    #   C[0,0] = p1 + p4 - p5 + p7
    #   C[0,1] = p3 + p5
    #   C[1,0] = p2 + p4
    #   C[1,1] = p1 - p2 + p3 + p6
    
    # For each 2x2x2 block, we need to define the U, V, W contributions.
    # The tensor for 2x2x2 has indices (i, j, k) where i is row of A, j is col of A/row of B, k is col of B.
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen's 7 terms as linear combinations:
    # Each term is a product of a linear form in A and a linear form in B.
    # We need to express each linear form as a vector.
    
    # For a 2x2x2 block at position (i_off, j_off, k_off):
    # i_off: offset in the n dimension (0 or 1, but since n=2, it's just the row)
    # j_off: offset in the m dimension (0, 2, 4, or 6 for the 4 j-values)
    # k_off: offset in the p dimension (0, 2, 4, or 6 for the 8 k-values)
    
    # Wait, I need to be more careful. Let me re-index.
    
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x8: B[j, k] where j in {0,1,2,3}, k in {0,...,7}
    # C is 2x8: C[i, k] where i in {0,1}, k in {0,...,7}
    
    # The 8 independent 2x2x2 problems are indexed by (j_group, k_group):
    # j_group in {0, 1} (corresponding to j in {0,1} or j in {2,3})
    # k_group in {0, 1, 2, 3} (corresponding to k in {0,1}, {2,3}, {4,5}, {6,7})
    
    # For a given (j_group, k_group):
    #   A_sub = A[:, j_group*2 : j_group*2+2]  (2x2)
    #   B_sub = B[j_group*2 : j_group*2+2, k_group*2 : k_group*2+2]  (2x2)
    #   C_sub = C[:, k_group*2 : k_group*2+2]  (2x2)
    
    #   C_sub = A_sub * B_sub
    
    # For each such 2x2x2 problem, Strassen gives 7 rank-1 terms.
    
    # Let me define the Strassen terms more carefully.
    # For A_sub = [[a,b],[c,d]] and B_sub = [[e,f],[g,h]]:
    # The 7 products are:
    #   p1 = (a+d)(e+h)
    #   p2 = (c+d)e
    #   p3 = a(f-h)
    #   p4 = d(g-e)
    #   p5 = (a+b)h
    #   p6 = (c-a)(e+f)
    #   p7 = (b-d)(g+h)
    
    # Each p is a product of a linear form in A and a linear form in B.
    # We need to express these as vectors in the U and V spaces.
    
    # For a 2x2x2 block at (j_group, k_group):
    # The local indices are:
    #   i_local in {0,1} (corresponds to global i)
    #   j_local in {0,1} (corresponds to global j = j_group*2 + j_local)
    #   k_local in {0,1} (corresponds to global k = k_group*2 + k_local)
    
    # U index: i * m + j = i * 4 + (j_group*2 + j_local)
    # V index: j * p + k = (j_group*2 + j_local) * 8 + (k_group*2 + k_local)
    # W index: k * n + i = (k_group*2 + k_local) * 2 + i
    
    # Now, for each of the 7 Strassen terms, I need to define:
    #   u_vec: a vector in R^4 (for the 2x2 A_sub)
    #   v_vec: a vector in R^4 (for the 2x2 B_sub)
    #   w_vec: a vector in R^4 (for the 2x2 C_sub)
    
    # The term contributes to the tensor as:
    #   T[i,j,k] += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
    
    # Wait, that's not quite right. The tensor decomposition is:
    #   T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # So for each rank-1 term, I specify U[i,j,r], V[j,k,r], W[k,i,r] for all i,j,k.
    
    # For a 2x2x2 block, the local tensor is:
    #   T_local[i_local, j_local, k_local] = sum_r U_local[i_local, j_local, r] * V_local[j_local, k_local, r] * W_local[k_local, i_local, r]
    
    # And the global tensor is the sum over all blocks.
    
    # Let me define the 7 Strassen terms as (u, v, w) triples where each is a 2x2 matrix.
    
    # p1 = (a+d)(e+h):
    #   u1 = [[1,0],[0,1]]  (a+d)
    #   v1 = [[1,1],[0,0]]  (e+h) -- wait, that's not right.
    
    # Let me be more careful. a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    # e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
    
    # p1 = (a+d)(e+h)
    #   u1[i,j]: the coefficient of A[i,j] in (a+d)
    #     u1 = [[1,0],[0,1]]
    #   v1[j,k]: the coefficient of B[j,k] in (e+h)
    #     v1 = [[1,1],[0,0]]  -- e is B[0,0], h is B[1,1], so v1[0,0]=1, v1[1,1]=1
    #     v1 = [[1,0],[0,1]]  -- wait, that's the identity. e+h = B[0,0]+B[1,1], so v1[0,0]=1, v1[1,1]=1.
    
    # Hmm, I think I'm overcomplicating this. Let me just define the linear forms explicitly.
    
    # For p1 = (a+d)(e+h):
    #   u1: coefficient of each A element: a:1, b:0, c:0, d:1 -> u1 = [[1,0],[0,1]]
    #   v1: coefficient of each B element: e:1, f:0, g:0, h:1 -> v1 = [[1,0],[0,1]]
    #   w1: the contribution to C. p1 contributes to C[0,0] and C[1,1] with +1, and to C[0,1] and C[1,0] with 0? 
    #   No, that's not right. p1 is just a scalar product. It contributes to all of C via the formula:
    #   C[0,0] = p1 + p4 - p5 + p7
    #   C[0,1] = p3 + p5
    #   C[1,0] = p2 + p4
    #   C[1,1] = p1 - p2 + p3 + p6
    
    # So p1 contributes +1 to C[0,0] and +1 to C[1,1].
    #   w1 = [[1,0],[0,1]]
    
    # Let me just define all 7 terms this way.
    
    # Term 1: p1 = (a+d)(e+h)
    #   u = [[1,0],[0,1]], v = [[1,0],[0,1]], w = [[1,0],[0,1]]
    
    # Term 2: p2 = (c+d)e
    #   u = [[0,0],[1,1]], v = [[1,0],[0,0]], w = [[0,0],[1,0]]  (contributes +1 to C[1,0])
    
    # Term 3: p3 = a(f-h)
    #   u = [[1,0],[0,0]], v = [[0,1],[0,-1]], w = [[0,1],[0,1]]  (contributes +1 to C[0,1] and +1 to C[1,1])
    
    # Wait, p3 contributes to C[0,1] with +1 and to C[1,1] with +1.
    #   w = [[0,1],[0,1]]
    
    # Term 4: p4 = d(g-e)
    #   u = [[0,0],[0,1]], v = [[-1,0],[1,0]], w = [[1,0],[0,1]]  (contributes +1 to C[0,0] and +1 to C[1,0])
    
    # Term 5: p5 = (a+b)h
    #   u = [[1,1],[0,0]], v = [[0,0],[0,1]], w = [[-1,0],[0,0]]  (contributes -1 to C[0,0])
    
    # Wait, p5 contributes -1 to C[0,0] and +1 to C[0,1].
    #   w = [[-1,1],[0,0]]
    
    # Term 6: p6 = (c-a)(e+f)
    #   u = [[-1,0],[1,0]], v = [[1,1],[0,0]], w = [[0,0],[0,1]]  (contributes +1 to C[1,1])
    
    # Term 7: p7 = (b-d)(g+h)
    #   u = [[0,1],[0,-1]], v = [[0,0],[1,1]], w = [[1,0],[0,0]]  (contributes +1 to C[0,0])
    
    # Let me verify: 
    # C[0,0] = p1 + p4 - p5 + p7
    #   p1: w[0,0]=1, p4: w[0,0]=1, p5: w[0,0]=-1, p7: w[0,0]=1 -> 1+1-(-1)+1 = 4? That's not right.
    
    # I think I'm making an error. Let me look up Strassen's algorithm more carefully.
    
    # Strassen's 7 products:
    #   p1 = (a+d)(e+h)
    #   p2 = (c+d)e
    #   p3 = a(f-h)
    #   p4 = d(g-e)
    #   p5 = (a+b)h
    #   p6 = (c-a)(e+f)
    #   p7 = (b-d)(g+h)
    
    # And:
    #   C[0,0] = p1 - p5 + p7 + p4
    #   C[0,1] = p3 + p5
    #   C[1,0] = p2 + p4
    #   C[1,1] = p1 - p2 + p3 + p6
    
    # Let me verify with a simple example: A = I, B = I
    #   a=1, b=0, c=0, d=1, e=1, f=0, g=0, h=1
    #   p1 = (1+1)(1+1) = 4
    #   p2 = (0+1)*1 = 1
    #   p3 = 1*(0-1) = -1
    #   p4 = 1*(0-1) = -1
    #   p5 = (1+0)*1 = 1
    #   p6 = (0-1)*(1+0) = -1
    #   p7 = (0-1)*(0+1) = -1
    #   C[0,0] = 4 - 1 + (-1) + (-1) = 1 ✓
    #   C[0,1] = -1 + 1 = 0 ✓
    #   C[1,0] = 1 + (-1) = 0 ✓
    #   C[1,1] = 4 - 1 + (-1) + (-1) = 1 ✓
    
    # Good. Now let me define the w vectors:
    #   w1 (for p1): contributes +1 to C[0,0] and +1 to C[1,1] -> [[1,0],[0,1]]
    #   w2 (for p2): contributes +1 to C[1,0] -> [[0,0],[1,0]]
    #   w3 (for p3): contributes +1 to C[0,1] and +1 to C[1,1] -> [[0,1],[0,1]]
    #   w4 (for p4): contributes +1 to C[0,0] and +1 to C[1,0] -> [[1,0],[1,0]]
    #   w5 (for p5): contributes -1 to C[0,0] and +1 to C[0,1] -> [[-1,1],[0,0]]
    #   w6 (for p6): contributes +1 to C[1,1] -> [[0,0],[0,1]]
    #   w7 (for p7): contributes +1 to C[0,0] -> [[1,0],[0,0]]
    
    # Now let me define the u and v vectors:
    #   u1 = [[1,0],[0,1]], v1 = [[1,0],[0,1]]
    #   u2 = [[0,0],[1,1]], v2 = [[1,0],[0,0]]
    #   u3 = [[1,0],[0,0]], v3 = [[0,1],[0,-1]]
    #   u4 = [[0,0],[0,1]], v4 = [[-1,0],[1,0]]
    #   u5 = [[1,1],[0,0]], v5 = [[0,0],[0,1]]
    #   u6 = [[-1,0],[1,0]], v6 = [[1,1],[0,0]]
    #   u7 = [[0,1],[0,-1]], v7 = [[0,0],[1,1]]
    
    # Let me verify p1: u1 * A = (a+d), v1 * B = (e+h). Product = (a+d)(e+h) ✓
    # p2: u2 * A = (c+d), v2 * B = e. Product = (c+d)e ✓
    # p3: u3 * A = a, v3 * B = f-h. Product = a(f-h) ✓
    # p4: u4 * A = d, v4 * B = g-e. Product = d(g-e) ✓
    # p5: u5 * A = a+b, v5 * B = h. Product = (a+b)h ✓
    # p6: u6 * A = c-a, v6 * B = e+f. Product = (c-a)(e+f) ✓
    # p7: u7 * A = b-d, v7 * B = g+h. Product = (b-d)(g+h) ✓
    
    # Great! Now I can build the full decomposition.
    
    # For each of the 8 blocks (j_group in {0,1}, k_group in {0,1,2,3}):
    #   For each of the 7 Strassen terms:
    #     Add the contribution to U, V, W.
    
    rank = 8 * 7  # 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Define the 7 Strassen terms
    terms = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),  # p1
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),  # p2
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,1]]),  # p3
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,0],[1,0]]),  # p4
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),  # p5
        ([[ -1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),  # p6
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),  # p7
    ]
    
    col = 0
    for j_group in range(2):  # j in {0,1} or {2,3}
        for k_group in range(4):  # k in {0,1}, {2,3}, {4,5}, {6,7}
            for u_mat, v_mat, w_mat in terms:
                for i_local in range(2):
                    for j_local in range(2):
                        for k_local in range(2):
                            # Global indices
                            i = i_local
                            j = j_group * 2 + j_local
                            k = k_group * 2 + k_local
                            
                            # U index: i * m + j
                            U[i * m + j][col] += u_mat[i_local][j_local]
                            
                            # V index: j * p + k
                            V[j * p + k][col] += v_mat[j_local][k_local]
                            
                            # W index: k * n + i
                            W[k * n + i][col] += w_mat[k_local][i_local]
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
