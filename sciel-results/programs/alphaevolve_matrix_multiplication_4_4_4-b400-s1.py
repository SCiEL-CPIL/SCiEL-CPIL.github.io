"""Matrix multiplication 4x4x4 tensor decomposition using Strassen's algorithm."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen decomposition for 4x4 matrix multiplication.
    
    Split 4x4 matrices into 2x2 blocks of 2x2 matrices.
    Use Strassen's 7-multiplication scheme at the 4x4 level,
    and schoolbook (4 mults) for each 2x2 sub-multiplication.
    
    Total rank: 7 * 4 = 28
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # 2x2 block indices for 4x4 matrix
    # A[i,j] where i,j in {0,1,2,3}
    # Block (bi, bj) covers rows 2*bi, 2*bi+1 and cols 2*bj, 2*bj+1
    # So A_block(bi,bj) has entries at (2*bi + r, 2*bj + c) for r,c in {0,1}
    
    # For 4x4 Strassen, we define:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B
    
    # Strassen 7 products for 2x2 block matrix multiplication:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2 matrix product, done with 4 scalar multiplications (schoolbook)
    # So total rank = 7 * 4 = 28
    
    # For each of the 7 Strassen products, we need to define:
    # - Which elements of A go into the left factor (with signs)
    # - Which elements of B go into the right factor (with signs)
    # - How the result contributes to C elements
    
    # Let's define the 7 Strassen terms. For each term t (0-6), 
    # the left matrix L_t and right matrix R_t are linear combinations of A blocks and B blocks.
    # Each L_t and R_t is a 2x2 matrix. The product L_t * R_t is computed with 4 mults.
    
    # For the k-th scalar multiplication in term t (k in 0..3):
    # L_t[r][c] * R_t[c][r2] contributes to (L_t * R_t)[r][r2]
    # where r, c, r2 are in {0,1}
    
    # The (L_t * R_t)[r][r2] then contributes to C with some coefficient.
    
    # Let me define everything explicitly.
    
    # Block element mapping:
    # A[bi][bj] block element (r, c) corresponds to global A[2*bi + r][2*bj + c]
    # So U index for A[2*bi+r][2*bj+c] is (2*bi+r)*4 + (2*bj+c)
    # V index for B[2*bi+r][2*bj+c] is (2*bi+r)*4 + (2*bj+c)  
    # W index for C[2*bi+r][2*bj+c] is (2*bj+c)*4 + (2*bi+r)  [trace convention: W[k*n+i, r] where k is row of B, i is col of A... wait]
    
    # Actually, the trace convention: W[k*n+i, r]
    # For C = A*B, C[i][j] = sum_k A[i][k] * B[k][j]
    # The tensor is T[i,j,k] = A[i,k] * B[k,j] * C[j,i]? No...
    
    # Standard: (A*B)[i,j] = sum_k A[i,k] B[k,j]
    # Tensor T with T[i,k,j] = 1 means A[i,k]*B[k,j] contributes to C[i,j]
    # Factorization: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Wait, the convention says W[k*n+i, r]. Let me re-read.
    
    # "W[k*n+i,r]" - this is the index. So W is indexed by (k*n+i) and r.
    # In the factorization: C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[?, r]
    # The W index should correspond to the output. If W[k*n+i, r], that's odd.
    
    # Let me just use the schoolbook convention from the seed:
    # U[i*m+j][column], V[j*p+k][column], W[k*n+i][column]
    # So for term with A[i,j], B[j,k], C[k,i]:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For 4x4: n=m=p=4
    # U[i*4 + j], V[j*4 + k], W[k*4 + i]
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So the "i" in C is the row, "j" is the col. But W[k*n+i] suggests W index is k*4+i where k is the B-row and i is the A-row... 
    
    # Wait, let me re-examine. The output C has elements C[row, col].
    # C[row, col] = sum_k A[row, k] * B[k, col]
    # In the seed: W[k*n+i] where i is A's col (=k in sum) and... 
    
    # Actually in the seed: for i in range(n), j in range(m), k in range(p):
    # U[i*m+j], V[j*p+k], W[k*n+i]
    # This represents: A[i,j] * B[j,k] -> C[k,i]? That doesn't seem right for standard matmul.
    
    # Hmm, let me think about this differently. The tensor for matrix multiplication C = A*B:
    # C[a,b] = sum_c A[a,c] * B[c,b]
    # The tensor T[a,c,b] = 1 (one-hot for the multiplication)
    # Factorization: T[a,c,b] = sum_r U[a*m+c, r] * V[c*p+b, r] * W[b*n+a, r]
    # So W index is b*n + a where b is B's col and a is A's row.
    
    # But the task says W[k*n+i, r]. If we map: k corresponds to b (B's column = C's column), 
    # and i corresponds to a (A's row = C's row). So W[b*4 + a].
    
    # Let me just follow the seed's convention exactly:
    # U[i*m + j, r] where i = A row, j = A col
    # V[j*p + k, r] where j = B row, k = B col  
    # W[k*n + i, r] where k = B col, i = A row
    # And the product gives C[i, k] = sum_j A[i,j] * B[j,k]
    
    # Yes! So C[row, col] = sum_j A[row, j] * B[j, col]
    # U index: row*4 + j
    # V index: j*4 + col
    # W index: col*4 + row
    
    # Now for Strassen on 4x4:
    # Split into 2x2 blocks. Block (bi, bj) of A covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # A block element A[bi][bj][r][c] = A_global[2*bi+r][2*bj+c]
    # U index for this: (2*bi+r)*4 + (2*bj+c)
    
    # Similarly for B and C.
    
    # Strassen 7 products (for 2x2 block matrices):
    # Let A = [[A11, A12], [A21, A22]] where Aij are 2x2 blocks
    # Let B = [[B11, B12], [B21, B22]] where Bij are 2x2 blocks
    # C = [[C11, C12], [C21, C22]]
    
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2 matrix = product of two 2x2 matrices, computed with 4 scalar mults.
    # For Pi = Li * Ri, the (r,c) element of Pi is sum_d Li[r,d] * Ri[d,c]
    # Each such scalar product Li[r,d] * Ri[d,c] is one rank-1 term.
    
    # For each of the 7 terms, and each of the 4 scalar products within:
    # We need to express Li[r,d] as a linear combination of A elements
    # and Ri[d,c] as a linear combination of B elements
    # and the resulting Pi[r,c] contributes to C blocks with certain coefficients.
    
    # This is getting complex. Let me build it step by step.
    
    # Define helper: for a 2x2 block matrix M with elements M[r][c] (r,c in 0,1),
    # the global index in U/V/W is:
    # For A block (bi, bj): element (r,c) -> global (2*bi+r, 2*bj+c) -> U index (2*bi+r)*4 + (2*bj+c)
    # For B block (bi, bj): element (r,c) -> global (2*bi+r, 2*bj+c) -> V index (2*bi+r)*4 + (2*bj+c)
    # For C block (bi, bj): element (r,c) -> global (2*bi+r, 2*bj+c) -> W index (2*bj+c)*4 + (2*bi+r)
    
    # For each Strassen term t (0-6), define:
    # L_t: a 2x2 matrix whose elements are linear combos of A block elements
    # R_t: a 2x2 matrix whose elements are linear combos of B block elements
    # contrib_t: how P_t = L_t * R_t contributes to C blocks
    
    # Let me define the A and B block elements as abstract variables.
    # A block (bi,bj) element (r,c): let's call it A[bi][bj][r][c]
    # Its U index: (2*bi+r)*4 + (2*bj+c)
    
    # For each term, L_t[r][c] is a sum of coeff * A[bi][bj][r'][c']
    # Similarly for R_t.
    
    # Term 0: P1 = (A11 + A22)(B11 + B22)
    # L_0 = A11 + A22, so L_0[r][c] = A[0][0][r][c] + A[1][1][r][c]
    # R_0 = B11 + B22, so R_0[r][c] = B[0][0][r][c] + B[1][1][r][c]
    # P1 contributes: C11 += P1, C22 += P1
    
    # Term 1: P2 = (A21 + A22)B11
    # L_1 = A21 + A22, L_1[r][c] = A[1][0][r][c] + A[1][1][r][c]
    # R_1 = B11, R_1[r][c] = B[0][0][r][c]
    # P2 contributes: C21 += P2, C22 -= P2
    
    # Term 2: P3 = A11(B12 - B22)
    # L_2 = A11, L_2[r][c] = A[0][0][r][c]
    # R_2 = B12 - B22, R_2[r][c] = B[0][1][r][c] - B[1][1][r][c]
    # P3 contributes: C12 += P3, C22 += P3
    
    # Term 3: P4 = A22(B21 - B11)
    # L_3 = A22, L_3[r][c] = A[1][1][r][c]
    # R_3 = B21 - B11, R_3[r][c] = B[1][0][r][c] - B[0][0][r][c]
    # P4 contributes: C11 += P4, C21 += P4
    
    # Term 4: P5 = (A11 + A12)B22
    # L_4 = A11 + A12, L_4[r][c] = A[0][0][r][c] + A[0][1][r][c]
    # R_4 = B22, R_4[r][c] = B[1][1][r][c]
    # P5 contributes: C11 -= P5, C12 += P5
    
    # Term 5: P6 = (A21 - A11)(B11 + B12)
    # L_5 = A21 - A11, L_5[r][c] = A[1][0][r][c] - A[0][0][r][c]
    # R_5 = B11 + B12, R_5[r][c] = B[0][0][r][c] + B[0][1][r][c]
    # P6 contributes: C22 += P6
    
    # Term 6: P7 = (A12 - A22)(B21 + B22)
    # L_6 = A12 - A22, L_6[r][c] = A[0][1][r][c] - A[1][1][r][c]
    # R_6 = B21 + B22, R_6[r][c] = B[1][0][r][c] + B[1][1][r][c]
    # P7 contributes: C11 += P7
    
    # Now, for each term t and each scalar product (r, d, c) where P_t[r][c] = sum_d L_t[r][d] * R_t[d][c]:
    # The scalar product L_t[r][d] * R_t[d][c] contributes to P_t[r][c].
    # P_t[r][c] then contributes to C blocks.
    
    # For a given term t, scalar product index (r, d, c) in 0..1:
    # L_t[r][d] is a linear combo of A elements
    # R_t[d][c] is a linear combo of B elements
    # The product contributes to C with the coefficient from the C block mapping.
    
    # Let me now build the U, V, W matrices.
    
    # For each term t (0-6) and each (r, d, c) in 0..1 (4 scalar products):
    # rank index = t * 4 + (r * 4 + d * 2 + c)  # or some ordering
    
    # Actually, let's use rank index = t * 4 + (r * 2 + d) * 2 + c or just t*4 + r*4+d*2+c... 
    # Let me use: rank_idx = t * 4 + (r * 2 + d) * 2 + c  -> no, let's just use t*4 + r*4 + d*2 + c... 
    # Simpler: for term t, the 4 scalar products are indexed by (r, d, c) where r,d,c in {0,1}.
    # Let me map (r, d, c) to sub-index = r*4 + d*2 + c (range 0-7, but only 4 valid? No, all 8 combos? No, r,d,c each 0 or 1, so 8 combos, but we only have 4 scalar products per 2x2 matmul).
    
    # Wait, for a 2x2 matrix product P = L * R:
    # P[0][0] = L[0][0]*R[0][0] + L[0][1]*R[1][0]
    # P[0][1] = L[0][0]*R[0][1] + L[0][1]*R[1][1]
    # P[1][0] = L[1][0]*R[0][0] + L[1][1]*R[1][0]
    # P[1][1] = L[1][0]*R[0][1] + L[1][1]*R[1][1]
    
    # That's 8 scalar products, not 4! I made an error. Schoolbook 2x2 matmul is 4 multiplications of 2x2? No, schoolbook 2x2 matmul is 8 scalar multiplications (or 4 if you think of it as 2x2 dot products... no).
    
    # Schoolbook 2x2: 8 scalar mults. So 7 terms * 8 = 56? That's worse than 64? No, 56 < 64.
    
    # Wait, I think I confused myself. Let me reconsider.
    
    # For 4x4 matmul using Strassen at the 4x4 level (splitting into 2x2 blocks):
    # We do 7 multiplications of 2x2 matrices.
    # Each 2x2 matrix multiplication takes 8 scalar multiplications (schoolbook) or 7 (Strassen).
    # If schoolbook: 7 * 8 = 56 rank.
    # If Strassen on the 2x2: 7 * 7 = 49 rank.
    
    # So rank 49 is achievable. Let me implement Strassen recursively.
    
    # Actually, let me just do the 7 terms with schoolbook 2x2 (8 mults each) = rank 56 first, then consider 49.
    
    # Let me restart with rank 56 (7 Strassen terms, each with 8 scalar mults for schoolbook 2x2).
    
    rank = 56
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Define the 7 terms
    # Each term has:
    # L: 2x2 matrix, each element is list of (a_block_i, a_block_j, r, c, coeff)
    # R: 2x2 matrix, each element is list of (b_block_i, b_block_j, r, c, coeff)
    # contrib: for each (r, c) in P, list of (c_block_i, c_block_j, cr, cc, coeff)
    
    def get_U_index(bi, bj, r, c):
        return (2*bi + r) * 4 + (2*bj + c)
    
    def get_V_index(bi, bj, r, c):
        return (2*bi + r) * 4 + (2*bj + c)
    
    def get_W_index(bi, bj, r, c):
        # C block (bi, bj) element (r, c) -> global C[2*bi+r][2*bj+c]
        # W index = col*4 + row = (2*bj+c)*4 + (2*bi+r)
        return (2*bj + c) * 4 + (2*bi + r)
    
    # Term definitions
    # For each term, L[r][c] and R[r][c] are dicts: {(bi,bj,er,ec): coeff}
    # where (er,ec) is the element position within the block
    
    terms = []
    
    # Term 0: P1 = (A11 + A22)(B11 + B22)
    L0 = {}
    R0 = {}
    for r in range(2):
        for c in range(2):
            L0[(r,c)] = {(0,0,r,c): 1, (1,1,r,c): 1}
            R0[(r,c)] = {(0,0,r,c): 1, (1,1,r,c): 1}
    # P1 contributes to C11 (+1) and C22 (+1)
    contrib0 = {}
    for r in range(2):
        for c in range(2):
            contrib0[(r,c)] = [(0,0,r,c,1), (1,1,r,c,1)]
    terms.append((L0, R0, contrib0))
    
    # Term 1: P2 = (A21 + A22)B11
    L1 = {}
    R1 = {}
    for r in range(2):
        for c in range(2):
            L1[(r,c)] = {(1,0,r,c): 1, (1,1,r,c): 1}
            R1[(r,c)] = {(0,0,r,c): 1}
    contrib1 = {}
    for r in range(2):
        for c in range(2):
            contrib1[(r,c)] = [(1,0,r,c,1), (1,1,r,c,-1)]
    terms.append((L1, R1, contrib1))
    
    # Term 2: P3 = A11(B12 - B22)
    L2 = {}
    R2 = {}
    for r in range(2):
        for c in range(2):
            L2[(r,c)] = {(0,0,r,c): 1}
            R2[(r,c)] = {(0,1,r,c): 1, (1,1,r,c): -1}
    contrib2 = {}
    for r in range(2):
        for c in range(2):
            contrib2[(r,c)] = [(0,1,r,c,1), (1,1,r,c,1)]
    terms.append((L2, R2, contrib2))
    
    # Term 3: P4 = A22(B21 - B11)
    L3 = {}
    R3 = {}
    for r in range(2):
        for c in range(2):
            L3[(r,c)] = {(1,1,r,c): 1}
            R3[(r,c)] = {(1,0,r,c): 1, (0,0,r,c): -1}
    contrib3 = {}
    for r in range(2):
        for c in range(2):
            contrib3[(r,c)] = [(0,0,r,c,1), (1,0,r,c,1)]
    terms.append((L3, R3, contrib3))
    
    # Term 4: P5 = (A11 + A12)B22
    L4 = {}
    R4 = {}
    for r in range(2):
        for c in range(2):
            L4[(r,c)] = {(0,0,r,c): 1, (0,1,r,c): 1}
            R4[(r,c)] = {(1,1,r,c): 1}
    contrib4 = {}
    for r in range(2):
        for c in range(2):
            contrib4[(r,c)] = [(0,0,r,c,-1), (0,1,r,c,1)]
    terms.append((L4, R4, contrib4))
    
    # Term 5: P6 = (A21 - A11)(B11 + B12)
    L5 = {}
    R5 = {}
    for r in range(2):
        for c in range(2):
            L5[(r,c)] = {(1,0,r,c): 1, (0,0,r,c): -1}
            R5[(r,c)] = {(0,0,r,c): 1, (0,1,r,c): 1}
    contrib5 = {}
    for r in range(2):
        for c in range(2):
            contrib5[(r,c)] = [(1,1,r,c,1)]
    terms.append((L5, R5, contrib5))
    
    # Term 6: P7 = (A12 - A22)(B21 + B22)
    L6 = {}
    R6 = {}
    for r in range(2):
        for c in range(2):
            L6[(r,c)] = {(0,1,r,c): 1, (1,1,r,c): -1}
            R6[(r,c)] = {(1,0,r,c): 1, (1,1,r,c): 1}
    contrib6 = {}
    for r in range(2):
        for c in range(2):
            contrib6[(r,c)] = [(0,0,r,c,1)]
    terms.append((L6, R6, contrib6))
    
    # Now build U, V, W
    for t, (L, R, contrib) in enumerate(terms):
        for r in range(2):
            for d in range(2):
                for c in range(2):
                    rank_idx = t * 8 + r * 4 + d * 2 + c
                    # L[r][d] contributes to U
                    for (bi, bj, er, ec), coeff in L[(r, d)].items():
                        u_idx = get_U_index(bi, bj, er, ec)
                        U[u_idx][rank_idx] += coeff
                    # R[d][c] contributes to V
                    for (bi, bj, er, ec), coeff in R[(d, c)].items():
                        v_idx = get_V_index(bi, bj, er, ec)
                        V[v_idx][rank_idx] += coeff
                    # P[r][c] contributes to W
                    for (bi, bj, er, ec, coeff) in contrib[(r, c)]:
                        w_idx = get_W_index(bi, bj, er, ec)
                        W[w_idx][rank_idx] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
