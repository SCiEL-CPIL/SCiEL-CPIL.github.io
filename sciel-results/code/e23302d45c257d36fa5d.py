import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen.
    
    Uses the 7-product Strassen algorithm applied to 2x2 blocks of the 4x4 matrices,
    where each 2x2 block multiplication is done via the trivial 4-product method.
    Total rank = 7 * 4 = 28? No wait, let me think again.
    
    Actually: 4x4 = 4 blocks of 2x2. Strassen gives 7 multiplications of 2x2 blocks.
    Each 2x2 block multiplication has rank 4 (trivial). So total rank = 7 * 4 = 28.
    
    Wait, that's not right either. The trivial 2x2 multiplication has rank 4, not 2.
    So 7 * 4 = 28. But we know the trivial 4x4 is rank 64.
    
    Let me reconsider: The tensor for 4x4 multiplication has 4*4*4 = 64 terms.
    Strassen reduces the 7 multiplications at the top level, each of which is a 2x2
    multiplication (rank 4). So 7 * 4 = 28? That seems too good.
    
    Actually no. The standard analysis: M(4) <= 7 * M(2) = 7 * 4 = 28? 
    But M(2) = 4 is the trivial bound, and M(2) <= 7 (Strassen for 2x2).
    So M(4) <= 7 * 4 = 28 using trivial 2x2, or M(4) <= 7 * 7 = 49 using Strassen 2x2.
    
    Wait, I'm confusing myself. Let me be precise:
    - The multiplication tensor for n x n matrices has rank at most n^3 (trivial).
    - For 2x2: rank <= 4 (trivial) or rank <= 7 (Strassen).
    - For 4x4: split into 2x2 blocks. C = A*B where A,B are 4x4 with 2x2 blocks.
      Strassen: 7 products of 2x2 blocks. Each 2x2 product has rank 4 (trivial).
      Total rank = 7 * 4 = 28.
    
    But wait, that's not how it works. The rank of the whole tensor is not simply
    7 times the rank of the sub-tensor. Let me think more carefully.
    
    Actually, the correct way: if we have a rank-r decomposition of the n x n
    multiplication tensor, and we apply it to compute block multiplications, we get
    a rank-(r * k^2) decomposition for (nk) x (nk) where k is the block size... no.
    
    Let me just use the standard result: M(4) <= 49 via recursive Strassen
    (7 multiplications at top, each 2x2 done with Strassen's 7).
    Or M(4) <= 28 if each 2x2 is done trivially (rank 4).
    
    Hmm, let me verify: for 4x4, split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    Strassen computes:
      P1 = (A11+A22)(B11+B22)
      P2 = (A21+A22)B11
      P3 = A11(B12-B22)
      P4 = A22(B21-B11)
      P5 = (A11+A12)B22
      P6 = (A21-A11)(B11+B12)
      P7 = (A12-A22)(B21+B22)
    
    Each Pi is a product of two 2x2 matrices. If we compute each 2x2 product
    trivially (4 scalar multiplications), total = 7 * 4 = 28 scalar multiplications.
    
    So the rank should be 28! Let me implement this.
    
    The tensor decomposition: for each Strassen term t (0..6) and each trivial
    2x2 product term s (0..3), we get one rank-1 term. Total rank = 7 * 4 = 28.
    """
    start_time = time.time()
    DEADLINE = 140
    
    n = 4
    rank = 28
    
    # U: shape (n*n, rank) = (16, 28)
    # V: shape (n*n, rank) = (16, 28)  
    # W: shape (n*n, rank) = (16, 28)
    # Convention: U[i*n+j, r] for A[i,j]
    #              V[j*n+k, r] for B[j,k]
    #              W[k*n+i, r] for C[i,j] (trace convention)
    
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    def u_idx(i, j):
        return i * n + j
    
    def v_idx(j, k):
        return j * n + k
    
    def w_idx(k, i, j):
        # Trace convention: W[k*n+i, r] corresponds to C[i,j]
        # Wait, the convention says W[k*n+i, r]. Let me re-read.
        # "The output factor follows the official trace convention W[k*n+i,r]"
        # So W has shape (p*n, rank) = (16, rank)
        # W[k*n+i, r] means the (k,i) entry of W (flattened as k*n+i)
        # This corresponds to C[i,j] where the k index is the summation index.
        # 
        # Actually, for matrix multiplication C[i,j] = sum_k A[i,k] * B[k,j]
        # The tensor T[i,j,k] = 1 if this is a valid term.
        # U[i*n+k, r] = coefficient of A[i,k] in term r
        # V[k*n+j, r] = coefficient of B[k,j] in term r
        # W[k*n+i, r]... hmm, this is confusing.
        #
        # Let me re-read: "W[k*n+i,r]" with shape (p*n, rank)
        # p = 4 (number of columns of B / rows of C)
        # n = 4 (number of rows of A / columns of C)
        # So W has 16 entries indexed by (k, i) where k in 0..p-1, i in 0..n-1
        # W[k*n+i, r] is the coefficient for C[i, ?] in term r
        #
        # I think the convention is:
        # C[i,j] = sum_r W[?] * (sum_k U[i*n+k, r] * A[i,k]) * (sum_k V[k*n+j, r] * B[k,j])
        # But W is a matrix, not a tensor. So W must encode which C[i,j] each term contributes to.
        #
        # Actually, I think the correct interpretation is:
        # C[i,j] = sum_r (sum_k U[i*n+k, r] * A[i,k]) * (sum_k V[k*n+j, r] * B[k,j]) * W_factor
        # But W is (p*n, rank) = (16, rank), so it has one entry per (k,i) pair.
        #
        # I think the convention means:
        # The decomposition is: C = sum_r (U[:,r] outer V[:,r]) where
        # U[:,r] is a vector of length 16 (flattened 4x4), V[:,r] similarly.
        # But that doesn't make sense for matrix multiplication.
        #
        # Let me think differently. The standard tensor decomposition for matrix mult:
        # T[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[i,j,r]
        # But here we have U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
        # with n=m=p=4.
        #
        # I think: U[i*4+k, r] for A[i,k], V[k*4+j, r] for B[k,j], W[j*4+i, r] for C[i,j]
        # Wait no, it says W[k*n+i, r].
        #
        # Let me just use: W index = j * n + i for C[i,j], where j is the column of C.
        # But the problem says W[k*n+i, r]. So the first index of W is k*n+i.
        # In C[i,j] = sum_k A[i,k] B[k,j], the free indices of C are (i,j).
        # So W should be indexed by (i,j) or (j,i).
        #
        # Given W[k*n+i, r] with k in 0..3, i in 0..3:
        # I'll interpret k as the column j of C, and i as the row i of C.
        # So W[j*4+i, r] is the coefficient for C[i,j] in term r.
        
        return j * n + i
    
    # Strassen's 7 terms for 2x2 block multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each 2x2 block is at positions:
    # Block (br, bc) of a 4x4 matrix has elements at rows 2*br..2*br+1, cols 2*bc..2*bc+1
    
    def block_elem(br, bc, r, c):
        """Return (row, col) in 4x4 for element (r,c) of block (br,bc)"""
        return (2*br + r, 2*bc + c)
    
    # For each Strassen term, define the A and B coefficients for each 2x2 block
    # and the C contribution
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict {(br,bc): [(r,c,coeff), ...]}
        # B_coeffs: dict {(br,bc): [(r,c,coeff), ...]}
        # C_coeffs: dict {(br,bc): [(r,c,coeff), ...]}
        
        # P1 = (A11+A22)(B11+B22)
        ({(0,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(0,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(0,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]}),
        
        # P2 = (A21+A22)B11
        ({(1,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(0,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(1,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,1): [(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1)]}),
        
        # P3 = A11(B12-B22)
        ({(0,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(0,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,1): [(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1)]},
         {(0,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]}),
        
        # P4 = A22(B21-B11)
        ({(1,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(1,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (0,0): [(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1)]},
         {(0,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]}),
        
        # P5 = (A11+A12)B22
        ({(0,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (0,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(1,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(0,0): [(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1)],
          (0,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]}),
        
        # P6 = (A21-A11)(B11+B12)
        ({(1,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (0,0): [(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1)]},
         {(0,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (0,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(1,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]}),
        
        # P7 = (A12-A22)(B21+B22)
        ({(0,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,1): [(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1)]},
         {(1,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)],
          (1,1): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]},
         {(0,0): [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]}),
    ]
    
    # C contributions from Strassen:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    c_contrib = [
        # For C11: P1(+1), P4(+1), P5(-1), P7(+1)
        {0: 1, 3: 1, 4: -1, 6: 1},
        # For C12: P3(+1), P5(+1)
        {2: 1, 4: 1},
        # For C21: P2(+1), P4(+1)
        {1: 1, 3: 1},
        # For C22: P1(+1), P2(-1), P3(+1), P6(+1)
        {0: 1, 1: -1, 2: 1, 5: 1},
    ]
    
    # Now build the decomposition
    # For each Strassen term t (0..6) and each 2x2 trivial product term s (0..3):
    # The 2x2 trivial product: (A_block)(B_block) = sum_{r,c} A[r,c] * B[c,r'] ... 
    # Actually, for 2x2: C2x2[i,j] = sum_k A2x2[i,k] * B2x2[k,j]
    # Trivial: 4 terms, one for each (i,k,j) with i,j in {0,1}
    # Term (i,k,j): A[i,k] * B[k,j] -> C[i,j]
    
    # For the 2x2 trivial decomposition, the 4 terms are:
    # s=0: A[0,0]*B[0,0] -> C[0,0]
    # s=1: A[0,1]*B[1,0] -> C[0,0]  ... wait no
    # 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # So 8 terms total for 2x2! Not 4.
    # 
    # Wait, the trivial rank for 2x2 is 4? No, it's 8.
    # M(2) <= 8 (trivial), M(2) <= 7 (Strassen).
    #
    # So for 4x4: 7 (Strassen) * 8 (trivial 2x2) = 56? That's worse than 64.
    # Or 7 * 7 = 49 if we use Strassen for 2x2 too.
    #
    # Let me use the trivial 8-term decomposition for 2x2:
    # Total rank = 7 * 8 = 56. Still better than 64.
    
    # Actually, let me just use the trivial 64-term decomposition for 4x4 directly.
    # That's guaranteed correct and gives rank 64.
    
    # But wait, the current program scores 1.3061 which corresponds to rank 49.
    # So I should aim for 49 or better.
    
    # Let me use recursive Strassen: 7 * 7 = 49.
    # Each 2x2 block multiplication uses Strassen's 7 terms.
    
    # For 2x2 Strassen:
    # A2x2 = [[a, b], [c, d]], B2x2 = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # So for 4x4, I have 7 outer Strassen terms, each involving products of 2x2 blocks.
    # Each 2x2 block product is done with 7 inner Strassen terms.
    # Total: 7 * 7 = 49 rank-1 terms.
    
    # Let me rebuild with this approach.
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    def u_idx(i, k):
        return i * 4 + k
    
    def v_idx(k, j):
        return k * 4 + j
    
    def w_idx(i, j):
        # Trace convention: W[k*n+i, r]
        # I'll use W[j*4+i, r] for C[i,j]
        return j * 4 + i
    
    # Outer Strassen terms (for 4x4 -> 2x2 blocks)
    # Each term specifies which 2x2 blocks of A and B are involved
    # and their coefficients
    
    outer_terms = [
        # (A_block_coeffs, B_block_coeffs, C_block_contrib)
        # A_block_coeffs: dict {block_idx: coeff} where block_idx is (br,bc)
        # Similarly for B and C
        
        # P1 = (A11+A22)(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),  # C11 += P1, C22 += P1
        
        # P2 = (A21+A22)B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(1,0): 1, (1,1): -1}),  # C21 += P2, C22 -= P2
        
        # P3 = A11(B12-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),  # C12 += P3, C22 += P3
        
        # P4 = A22(B21-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),  # C11 += P4, C21 += P4
        
        # P5 = (A11+A12)B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (0,1): 1}),  # C11 -= P5, C12 += P5
        
        # P6 = (A21-A11)(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(1,1): 1}),  # C22 += P6
        
        # P7 = (A12-A22)(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): 1}),  # C11 += P7
    ]
    
    # Inner Strassen terms (for 2x2 -> scalar)
    # For 2x2 matrices A2x2 = [[a,b],[c,d]], B2x2 = [[e,f],[g,h]]
    # The 7 terms and their contributions:
    
    inner_terms = [
        # (A_elem_coeffs, B_elem_coeffs, C_elem_contrib)
        # A_elem_coeffs: dict {(r,c): coeff} for 2x2 A
        # B_elem_coeffs: dict {(r,c): coeff} for 2x2 B
        # C_elem_contrib: dict {(r,c): coeff} for 2x2 C
        
        # P1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        
        # P2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        
        # P3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        
        # P4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        
        # P5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        
        # P6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        
        # P7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    
    # Now combine: for each outer term t (0..6) and inner term s (0..6):
    # rank_idx = t * 7 + s
    
    for t in range(7):
        a_blocks, b_blocks, c_blocks = outer_terms[t]
        
        for s in range(7):
            a_elems, b_elems, c_elems = inner_terms[s]
            rank_idx = t * 7 + s
            
            # For each A block in a_blocks:
            for (br, bc), block_coeff in a_blocks.items():
                for (er, ec), elem_coeff in a_elems.items():
                    # Element position in 4x4:
                    row = 2 * br + er
                    col = 2 * bc + ec
                    u = u_idx(row, col)
                    U[u][rank_idx] += block_coeff * elem_coeff
            
            # For each B block in b_blocks:
            for (br, bc), block_coeff in b_blocks.items():
                for (er, ec), elem_coeff in b_elems.items():
                    # B block (br,bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
                    # In B, the element is B[k,j] where k is row, j is col
                    row = 2 * br + er
                    col = 2 * bc + ec
                    v = v_idx(row, col)
                    V[v][rank_idx] += block_coeff * elem_coeff
            
            # For each C block in c_blocks:
            for (br, bc), block_coeff in c_blocks.items():
                for (er, ec), elem_coeff in c_elems.items():
                    # C block (br,bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
                    row = 2 * br + er
                    col = 2 * bc + ec
                    w = w_idx(row, col)
                    W[w][rank_idx] += block_coeff * elem_coeff
    
    # Verify the decomposition using fast NumPy vectorized computation
    U_np = np.array(U, dtype=float)  # (16, 49)
    V_np = np.array(V, dtype=float)  # (16, 49)
    W_np = np.array(W, dtype=float)  # (16, 49)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    # Vectorized computation:
    # U_np reshaped to (4,4,49): U_np[i*4+k, r] -> U4[i,k,r]
    U4 = U_np.reshape(4, 4, 49)   # [i, k, r]
    V4 = V_np.reshape(4, 4, 49)   # [k, j, r]
    # W_np[j*4+i, r] -> W4[j, i, r]
    W4 = W_np.reshape(4, 4, 49)   # [j, i, r]
    
    # u_val[i, r] = sum_k U4[i,k,r] * A[i,k]
    u_vals = np.einsum('ikr,ik->ir', U4, A_test)  # (4, 49)
    # v_val[j, r] = sum_k V4[k,j,r] * B[k,j]
    v_vals = np.einsum('kj r,kj->jr', V4, B_test)  # (4, 49)
    
    # C[i,j] = sum_r W4[j,i,r] * u_vals[i,r] * v_vals[j,r]
    C_computed = np.einsum('jir,ir,jr->ij', W4, u_vals, v_vals)
    
    max_err = np.max(np.abs(C_computed - C_expected))
    
    if max_err > 1e-9:
        # Fallback: trivial rank-64 decomposition
        rank_fallback = 64
        U_f = [[0] * rank_fallback for _ in range(16)]
        V_f = [[0] * rank_fallback for _ in range(16)]
        W_f = [[0] * rank_fallback for _ in range(16)]
        
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U_f[i*4+k][r] = 1
                    V_f[k*4+j][r] = 1
                    W_f[j*4+i][r] = 1
        
        return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
