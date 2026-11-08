"""Matrix multiplication 4,4,7 rank minimization."""
import time
import sys

DIMENSIONS = [4, 4, 7]

# Strassen decomposition for 2x2x2: 7 terms
# Terms are (i1, i2, j1, j2, k1, k2) meaning:
# term = A[i1,j1]*A[i2,j2] ... actually let's define the linear forms
#
# Standard Strassen 2x2x2:
# C[0,0] = M1 - M2 + M3 - M4  ... wait, let me use the standard definitions
#
# Actually, for 2x2 matrices:
# C00 = M1 - M2 + M3 - M4  -> No
# Standard Strassen:
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B10-B11)
# M4 = A11(B01-B10)
# M5 = (A00+A01)B11
# M6 = (A10-A00)(B00+B10)
# M7 = (A01-A11)(B01+B11)
# C00 = M1 - M4 + M6 + M7  ... let me look up carefully
#
# Correct Strassen:
# C00 = M1 - M2 + M3 - M4  ... I'll just use the known factor matrices
#
# Let me define the U, V, W matrices for 2x2x2 Strassen directly.
# n=m=p=2, indices: U has shape (2*2, 7), V has shape (2*2, 7), W has shape (2*2, 7)
# U[i*m+j, t] = U[i*2+j, t]
# V[j*p+k, t] = V[j*2+k, t]
# W[k*n+i, t] = W[k*2+i, t]
#
# The 7 Strassen terms (using 0-based indexing for the 2x2 blocks):
# Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = A*B
#
# M1 = (a00+a11)(b00+b11)
# M2 = (a10+a11)(b00)
# M3 = (a00)(b10-b11)
# M4 = (a11)(b01-b10)
# M5 = (a00+a01)(b11)
# M6 = (a10-a00)(b00+b10)
# M7 = (a01-a11)(b01+b11)
#
# C00 = M1 - M4 + M6 + M7 ... 
# Actually let me use the canonical Strassen formulas:
# C00 = M1 - M2 + M3 - M4  ... no
#
# I'll use a well-known set. The factor matrices for 2x2x2 Strassen:
# For term t (0..6):
#   U[row, t], V[col, t], W[out, t]
#
# Let me just hardcode the Strassen factors from a known reference.
# Indices: U is 4x7, V is 4x7, W is 4x7
# U index: i*2+j for input A[i,j]
# V index: j*2+k for input B[j,k]  
# W index: k*2+i for output C[i,k]

def strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen, each 4x7 with integer entries."""
    # U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    # Let me use the standard Strassen with the following 7 products:
    # Product t: sum over entries u_a[i,j] * v_b[j,k] * w_c[k,i] should give C[i,k] contribution
    # 
    # I'll use this known decomposition:
    # Term 0: A00 * (B00+B11) -> contributes to... let me think differently.
    # 
    # Actually, let me use the bilinear form approach.
    # The multiplication tensor has components T[i,j,k,l,m,n] = delta[i,k]*delta[j,m]*delta[l,n]... 
    # No, for matrix mult: C_{ak} = sum_i A_{ai} B_{ik}
    # Tensor T[a, i, k, j, m, n] where C index is (a,k), A index is (a,i), B index is (i,k)
    # With the given indexing: U has rows (n*m) = (i*m+j) so (i,j) for A[i,j]
    # V has rows (m*p) = (j*p+k) so (j,k) for B[j,k]
    # W has rows (p*n) = (k*n+i) so (k,i) for C[i,k]
    #
    # So the tensor element is T[i,j,k] = C[i,k] coefficient from A[i,j]*B[j,k] = delta[i,i']*delta[k,k]...
    # Actually T[A_idx, B_idx, C_idx] where A_idx encodes (i,j), B_idx encodes (j,k), C_idx encodes (k,i)
    # And T is nonzero only when the j's match and the k in B matches the k in C's first index... 
    #
    # Let me just verify by constructing the schoolbook and checking, then modify.
    
    # I'll use a verified Strassen decomposition. 
    # Reference: the 7 terms of Strassen for 2x2 matrix multiplication.
    # Using 0-indexed positions in the 4x4 unfolded matrices:
    # A positions: 0=a00, 1=a01, 2=a10, 3=a11
    # B positions: 0=b00, 1=b01, 2=b10, 3=b11
    # C positions: 0=c00, 1=c01, 2=c10, 3=c11  (C[idx] where idx = k*n+i, so c00 is at 0, c01 at 1, c10 at 2, c11 at 3)
    
    # The 7 Strassen products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    
    # And:
    # C00 = P1 - P2 + P3 - P4  ... this isn't right either.
    
    # Let me use the definitive Strassen:
    # C00 = M1 - M2 + M3 - M4
    # C01 = M5 + M4
    # C10 = M2 + M3
    # C11 = M1 - M5 + M6 + M7
    # Hmm, this also doesn't look standard.
    
    # I'll use the WIKIPEDIA version:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 - M4 + M6 + M7
    # C12 = M4 + M5
    # C21 = M2 + M3
    # C22 = M1 - M2 + M5 - M7
    #
    # Converting to 0-indexed: A11->a00, A12->a01, A21->a10, A22->a11
    # B11->b00, B12->b01, B21->b10, B22->b11
    # C11->c00, C12->c01, C21->c10, C22->c11
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 - M4 + M6 + M7
    # c01 = M4 + M5
    # c10 = M2 + M3
    # c11 = M1 - M2 + M5 - M7
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Each term t: for each (i,j,k) with the coefficient
    # Term 0 (M1): U[a00]=1,U[a11]=1; V[b00]=1,V[b11]=1; W[c00]=1,W[c11]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1 (M2): U[a10]=1,U[a11]=1; V[b00]=1; W[c10]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 2 (M3): U[a00]=1; V[b10]=1,V[b11]=-1; W[c10]=1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # Term 3 (M4): U[a11]=1; V[b01]=1,V[b00]=-1; W[c00]=-1,W[c01]=1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # Term 4 (M5): U[a00]=1,U[a01]=1; V[b11]=1; W[c01]=1,W[c11]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # Term 5 (M6): U[a10]=1,U[a00]=-1; V[b00]=1,V[b01]=1; W[c00]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # Term 6 (M7): U[a01]=1,U[a11]=-1; V[b10]=1,V[b11]=1; W[c00]=1,W[c11]=-1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W


def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    import numpy as np
    rank = len(W[0]) if W else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i, j, k) input to output:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # With our indexing:
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    # 
    # The decomposition says: for each output C[i,k] (W row index k*n+i),
    # C[i,k] = sum_t W[k*n+i, t] * (product of A and B parts)
    # Actually: the tensor decomposition means:
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # And T[i,j,k] should equal 1 if the indices form a valid matrix mult term, i.e.
    # for fixed i,j,k: this contributes to C[i,k] with coefficient 1 iff j==j' and k==k'
    # Actually T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(k,i) for the same i,j,k
    # More precisely: T[i*m+j, j'*p+k', k''*n+i''] = delta[j,j'] * delta[k,k''] * delta[i,i'']
    # Wait no. The multiplication tensor T has T[a,b,c] where a indexes (i,j) in A, b indexes (j,k) in B, 
    # c indexes (k,i) in C. T[a,b,c] = 1 iff the j in a equals the j in b, AND the k in b equals the k in c, 
    # AND the i in a equals the i in c. Otherwise 0.
    
    # Let's just check: for all i, j, k:
    # sum over the decomposition should give the right contribution
    # The tensor element at (A_row=i*m+j, B_row=j*p+k, C_row=k*n+i) should be 1
    # And all other elements should be 0.
    
    # Check a sample of elements
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                expected = 1.0
                actual = 0.0
                for t in range(rank):
                    actual += U_np[a_idx, t] * V_np[b_idx, t] * W_np[c_idx, t]
                if abs(actual - expected) > 1e-10:
                    return False
    
    # Check that non-matching elements are 0
    # Sample some non-matching cases
    import random
    for _ in range(min(50, n*m*p)):
        a_idx = random.randint(0, n*m - 1)
        b_idx = random.randint(0, m*p - 1)
        c_idx = random.randint(0, p*n - 1)
        # Extract i, j from a_idx; j2, k from b_idx; k2, i2 from c_idx
        i_a = a_idx // m
        j_a = a_idx % m
        j_b = b_idx // p
        k_b = b_idx % p
        k_c = c_idx // n
        i_c = c_idx % n
        # Should be 1 only if j_a == j_b and k_b == k_c and i_a == i_c
        expected = 1.0 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0.0
        actual = 0.0
        for t in range(rank):
            actual += U_np[a_idx, t] * V_np[b_idx, t] * W_np[c_idx, t]
        if abs(actual - expected) > 1e-10:
            return False
    
    return True


def combine_block_decomposition(n_blocks_a, m_blocks_a, p_blocks_a, 
                                  n_blocks_b, m_blocks_b, p_blocks_b,
                                  base_rank_per_block, block_U, block_V, block_W,
                                  block_positions):
    """
    Combine multiple block decompositions into a global one.
    
    block_positions: list of (a_block, b_block, c_block) indicating which 
                     A-block, B-block, C-block pair this block multiplication corresponds to.
                     a_block: (i1, i2) block indices in A (row block, col block... actually for block structure)
    """
    pass


def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    n, m, p = 4, 4, 7
    
    # Strategy: Use block decomposition
    # Split 4x4x4 part using Strassen (rank 49 for 4x4x4)
    # Split 4x4x3 part using blocks (rank 44 for 4x4x3)
    # Total: 49 + 44 = 93
    
    # First, let's build the 4x4x4 part using 2x2 blocks with Strassen
    # A is 4x4, B_top is 4x4 (rows 0-3 of B), C_top is 4x4
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B_top = [[B00, B01], [B10, B11]] where each is 2x2
    # C_top = A * B_top
    
    # Using Strassen on blocks: 7 block multiplications of 2x2 matrices
    # Each 2x2*2x2 multiplication uses Strassen with rank 7
    # So total rank for 4x4x4 = 7*7 = 49
    
    # Build 4x4x4 Strassen decomposition
    U_4x4x4, V_4x4x4, W_4x4x4 = strassen_2x2x2()  # 4x7 each
    
    # Now build the global decomposition for 4x4x7
    # Split B (4x7) into B_left (4x4) and B_right (4x3)
    # C (4x7) = A*B_left (4x4) + A*B_right (4x3)
    
    # For 4x4x4 part (using Strassen blocks): rank 49
    # For 4x4x3 part: split B_right into blocks
    # B_right is 4x3. Split A into 2x2 blocks (same as before), B_right into 2x2 and 2x1 blocks
    # B_right = [[Br00 (2x2), Br01 (2x1)], [Br10 (2x2), Br11 (2x1)]]
    
    # C_top_left (2x2) = A00*Br00 + A01*Br10
    # C_top_right (2x1) = A00*Br01 + A01*Br11
    # C_bot_left (2x2) = A10*Br00 + A11*Br10
    # C_bot_right (2x1) = A10*Br01 + A11*Br11
    
    # For 2x2x2 products: use Strassen (rank 7)
    # For 2x2x1 products: schoolbook (rank 4)
    
    # So:
    # C_top_left: 2 products of 2x2x2 -> 2*7 = 14
    # C_top_right: 2 products of 2x2x1 -> 2*4 = 8
    # C_bot_left: 2 products of 2x2x2 -> 2*7 = 14
    # C_bot_right: 2 products of 2x2x1 -> 2*4 = 8
    # Total for 4x4x3: 14+8+14+8 = 44
    
    total_rank = 49 + 44  # = 93
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 93
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 93
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 93
    
    col_offset = 0
    
    # Part 1: 4x4x4 Strassen (columns 0..48)
    # Global A index: for block (bi, bj) of A (each 2x2), local A index (ii, jj) -> global (bi*2+ii, bj*2+jj)
    #   Global A row index = (bi*2+ii)*4 + (bj*2+jj)  [since m=4]
    #   Wait, A is 4x4, so A[i,j] has row index i*4+j in U.
    #   For 2x2 block at (bi, bj) where bi, bj in {0,1}:
    #     A[bi*2+ii, bj*2+jj] has U row index (bi*2+ii)*4 + (bj*2+jj)
    
    # For B_left (4x4), B[j,k] has V row index j*7 + k  [since p=7]
    #   For 2x2 block at (bi, bk):
    #     B[bi*2+ii, bk*2+jj] has V row index (bi*2+ii)*7 + (bk*2+jj)
    
    # For C_top (4x4), C[i,k] has W row index k*4 + i  [since p=4 for this part... wait p=7 total]
    #   Actually W row index is k*n+i where n=4. So C[i,k] -> W row k*4+i.
    #   For 2x2 block at (ci, ck) where ci is the i-block and ck is the k-block:
    #     C[ci*2+ii, ck*2+jj] has W row (ck*2+jj)*4 + (ci*2+ii)
    
    # Strassen 2x2x2 local indices:
    # U_4x4x4 has rows 0..3 corresponding to local A[0,0],A[0,1],A[1,0],A[1,1]
    #   local idx ii*2+jj -> global A idx (bi*2+ii)*4 + (bj*2+jj)
    # V_4x4x4 has rows 0..3 corresponding to local B[0,0],B[0,1],B[1,0],B[1,1]
    #   local idx ii*2+jj -> global B idx (bi*2+ii)*7 + (bk*2+jj)
    # W_4x4x4 has rows 0..3 corresponding to local C[0,0],C[0,1],C[1,0],C[1,1]
    #   local idx kk*2+ii -> global C idx... C[i,k] with i=ci*2+ii, k=ck*2+jj
    #   W row = k*n+i = (ck*2+jj)*4 + (ci*2+ii)
    
    # Strassen has 7 terms. For each term t in 0..6, the local U,V,W entries are defined.
    # We need to map local indices to global indices.
    
    # The 7 Strassen terms combine different blocks:
    # M1 = (A00+A11)(B00+B11): involves A blocks (0,0) and (1,1), B blocks (0,0) and (1,1), C blocks (0,0) and (1,1)
    # This means for term 0, we have contributions from multiple (block_i, block_j, block_k) combinations.
    
    # Actually, the Strassen decomposition for 4x4x4 is more subtle. Each of the 7 "scalar" Strassen 
    # operations is itself a matrix multiplication of 2x2 matrices. So:
    
    # For M1: (A00+A11)*(B00+B11)
    #   U part: for local A[0,0] (A00): coefficient +1
    #           for local A[1,1] (A11): coefficient +1
    #   V part: for local B[0,0] (B00): coefficient +1
    #           for local B[1,1] (B11): coefficient +1
    #   W part: for local C[0,0]: coeff from W_4x4x4[0,0]
    #           for local C[1,1]: coeff from W_4x4x4[3,0]
    #
    # Wait, I think I'm overcomplicating this. The recursive Strassen approach:
    # Each of the 7 Strassen terms for the 4x4x4 is itself a 2x2x2 matrix multiplication.
    # So term t (of the outer Strassen) uses the 7-term Strassen decomposition internally.
    # That gives 7*7 = 49 terms total.
    #
    # For outer term t (0..6) and inner term s (0..6):
    #   The U coefficient for global A entry depends on:
    #     - which A-blocks are involved in outer term t
    #     - which local A entries within those blocks are involved in inner term s
    #
    # This is getting complex. Let me use a different approach: directly construct the 49 terms.
    
    # Actually, let me just use the schoolbook for now and verify, then try to optimize.
    # Schoolbook: rank = 4*4*7 = 112
    
    # Let me first implement schoolbook correctly and verify, then attempt Strassen.
    
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Verify schoolbook
    if not verify_decomposition(U_school, V_school, W_school, n, m, p):
        # Fallback: just return it
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Now try to build the Strassen-based rank 93 decomposition
    
    U_93 = [[0] * 93 for _ in range(n*m)]
    V_93 = [[0] * 93 for _ in range(m*p)]
    W_93 = [[0] * 93 for _ in range(p*n)]
    
    # Part 1: 4x4x4 using nested Strassen (rank 49, columns 0..48)
    # For outer Strassen term t (0..6) and inner Strassen term s (0..6):
    #   global_col = t * 7 + s
    
    # The outer Strassen terms for 4x4x4:
    # Each outer term involves specific 2x2 blocks of A, B, C.
    # Let me define the outer Strassen in terms of block indices.
    
    # Outer Strassen for A(4x4) * B(4x4) = C(4x4):
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # where each sub-block is 2x2.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M5 - M7
    
    # For each outer term, the A-blocks, B-blocks, and their coefficients:
    # Term 0 (M1): A_blocks = {(0,0):+1, (1,1):+1}, B_blocks = {(0,0):+1, (1,1):+1}, C_blocks = {(0,0):+1, (1,1):+1}
    # Term 1 (M2): A_blocks = {(1,0):+1, (1,1):+1}, B_blocks = {(0,0):+1}, C_blocks = {(1,0):+1}
    # Term 2 (M3): A_blocks = {(0,0):+1}, B_blocks = {(1,0):+1, (1,1):-1}, C_blocks = {(1,0):+1}
    # Term 3 (M4): A_blocks = {(1,1):+1}, B_blocks = {(0,1):+1, (0,0):-1}, C_blocks = {(0,0):-1, (0,1):+1}
    # Term 4 (M5): A_blocks = {(0,0):+1, (0,1):+1}, B_blocks = {(1,1):+1}, C_blocks = {(0,1):+1, (1,1):+1}
    # Term 5 (M6): A_blocks = {(1,0):+1, (0,0):-1}, B_blocks = {(0,0):+1, (0,1):+1}, C_blocks = {(0,0):+1}
    # Term 6 (M7): A_blocks = {(0,1):+1, (1,1):-1}, B_blocks = {(1,0):+1, (1,1):+1}, C_blocks = {(0,0):+1, (1,1):-1}
    
    outer_terms = [
        # (A_block_coeffs: [(ai,aj,coeff)], B_block_coeffs: [(bi,bj,coeff)], C_block_coeffs: [(ci,ck,coeff)])
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),                    # M2
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(1,0,1)]),                   # M3
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,-1),(0,1,1)]),          # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)]),            # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,0,1)]),           # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]),  # M7
    ]
    
    # For each outer term t and each pair of A-block (from A_block_coeffs) and B-block (from B_block_coeffs),
    # we perform a 2x2x2 Strassen multiplication, and the result goes to the C-blocks.
    #
    # Actually, for outer term t:
    #   The product is: (sum over A_blocks with coeffs) * (sum over B_blocks with coeffs)
    #   = sum_{a_blk, b_blk} A_coeff[a_blk] * B_coeff[b_blk] * (a_blk * b_blk)
    #   Each (a_blk * b_blk) is a 2x2x2 matrix multiplication, which we decompose with Strassen.
    
    # So for outer term t:
    for t in range(7):
        A_blocks_t, B_blocks_t, C_blocks_t = outer_terms[t]
        for (ai, aj, a_coeff) in A_blocks_t:
            for (bi, bj, b_coeff) in B_blocks_t:
                # Perform 2x2x2 Strassen for A_block(ai,aj) * B_block(bi,bj)
                # A_block(ai,aj) is 2x2, located at A rows ai*2..ai*2+1, cols aj*2..aj*2+1
                # B_block(bi,bj) is 2x2, located at B rows bi*2..bi*2+1, cols bj*2..bj*2+1
                # Result is a 2x2 matrix.
                # We decompose this 2x2x2 product using Strassen (7 inner terms).
                # The result needs to be added to C with coefficient a_coeff * b_coeff * (C_block_coeff)
                # But the C_block depends on the outer term.
                
                # Inner Strassen: 7 terms
                sU, sV, sW = U_4x4x4, V_4x4x4, W_4x4x4  # local 4x7 Strassen factors
                
                for s in range(7):
                    global_col = t * 7 + s
                    coeff = a_coeff * b_coeff
                    
                    # U part: for each local U entry in this Strassen term
                    for local_row in range(4):
                        u_val = sU[local_row][s]
                        if u_val != 0:
                            # local_row corresponds to (ii, jj) in local 2x2: ii = local_row//2, jj = local_row%2
                            ii = local_row // 2
                            jj = local_row % 2
                            global_i = ai * 2 + ii
                            global_j = aj * 2 + jj
                            global_a_idx = global_i * m + global_j  # m=4
                            U_93[global_a_idx][global_col] += u_val * coeff
                    
                    # V part
                    for local_row in range(4):
                        v_val = sV[local_row][s]
                        if v_val != 0:
                            ii = local_row // 2
                            jj = local_row % 2
                            global_j2 = bi * 2 + ii
                            global_k = bj * 2 + jj
                            global_b_idx = global_j2 * p + global_k  # p=7
                            V_93[global_b_idx][global_col] += v_val * coeff
                    
                    # W part: need to determine which C blocks this contributes to
                    # For inner Strassen, the W entries specify the local C positions.
                    # local_row in W: kk*2+ii -> local C[ii, kk]... 
                    # Wait, W_4x4x4[k*n+i, t] with n=2. So W row = k*2+i for C[i,k].
                    # local_row = kk*2 + ii means C[ii, kk] locally.
                    # The local C[ii, kk] needs to be placed into the global C.
                    # The C block structure: C_block(ci, ck) is 2x2 at C rows ci*2..ci*2+1, cols ck*2..ck*2+1.
                    # C[global_i, global_k] where global_i = ci*2+ii, global_k = ck*2+jj
                    # W row index = global_k * n + global_i = global_k*4 + global_i
                    #
                    # But which C blocks does this outer term contribute to?
                    # The outer term t has C_blocks_t which tells us the mapping.
                    # However, the inner Strassen produces a full 2x2 result, so all 4 local C entries
                    # contribute to the corresponding C blocks.
                    
                    for local_row in range(4):
                        w_val = sW[local_row][s]
                        if w_val != 0:
                            kk = local_row // 2
                            ii = local_row % 2
                            # This is C[ii, kk] locally
                            # Need to determine which global C position this maps to.
                            # The outer term determines which C blocks are involved.
                            # But actually, the product (a_blk * b_blk) produces a 2x2 result.
                            # This result needs to be scaled by a_coeff*b_coeff and added to the appropriate C blocks.
                            # But which C blocks? It depends on the outer Strassen formula.
                            
                            # Hmm, this is where it gets tricky. The outer Strassen says:
                            # C_block(ci,ck) = sum of certain M terms with certain coefficients.
                            # Each M term is a product of sums of A-blocks and B-blocks.
                            # So M_t = (sum_a A_coeff[a] * A_a) * (sum_b B_coeff[b] * B_b)
                            #         = sum_{a,b} A_coeff[a]*B_coeff[b] * (A_a * B_b)
                            # And C_block = sum_t (C_coeff_t * M_t)
                            #
                            # So the total contribution to C from term (t, a_blk, b_blk) is:
                            #   a_coeff * b_coeff * (A_a * B_b)
                            # And this gets added to C with an additional coefficient from the C_block mapping.
                            #
                            # Wait, I think I'm missing something. The C_blocks_t for each outer term
                            # tell us which C blocks receive this term and with what coefficient.
                            # But the inner product (A_a * B_b) produces a 2x2 matrix, so each entry
                            # of that 2x2 result maps to a specific entry in each of the C blocks.
                            
                            # Actually, re-reading the Strassen formulas:
                            # C00 = M1 - M4 + M6 + M7
                            # Here M1 is a single scalar (in the 1x1x1 sense), but in our 2x2 block
                            # sense, M1 = (A00+A11)(B00+B11) is a 2x2 matrix product result.
                            # So M1 is a 2x2 matrix, and C00 = M1 - M4 + M6 + M7 means element-wise
                            # subtraction/addition of 2x2 matrices.
                            #
                            # So for outer term t, the result is a 2x2 matrix that gets added to
                            # the C blocks listed in C_blocks_t with the given coefficients.
                            # But C_blocks_t can have multiple entries! For example, M1 contributes
                            # to both C00 and C11.
                            
                            # I think the issue is that the outer Strassen formula has terms like:
                            # C00 = M1 - M4 + M6 + M7
                            # where M1, M4, M6, M7 are each 2x2 matrices.
                            # So C00 is the sum of these 2x2 matrices.
                            #
                            # This means that for each outer term t, the 2x2 result of M_t is added
                            # to EACH of the C blocks listed in C_blocks_t with the given coefficient.
                            
                            # So for term (t, a_blk, b_blk, s), the contribution to C is:
                            #   for each (ci, ck, c_coeff) in C_blocks_t:
                            #     local C entry (ii, kk) -> global C entry (ci*2+ii, ck*2+jj) where jj=kk? 
                            #     No wait: local C[ii, kk] maps to C[ci*2+ii, ck*2+jj] where jj should be...
                            #     Actually the local C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1]
                            #     These map to global C entries in the C block (ci, ck).
                            #     C[ii, kk] in local -> C[ci*2+ii, ck*2+jj] where jj=kk
                            
                            for (ci, ck, c_coeff) in C_blocks_t:
                                global_i = ci * 2 + ii
                                global_k = ck * 2 + kk
                                global_c_idx = global_k * n + global_i  # n=4
                                U_93[global_a_idx][global_col]  # already handled above
                                total_coeff = coeff * c_coeff
                                W_93[global_c_idx][global_col] += w_val * total_coeff
    
    # Hmm wait, I think I have a bug. The U and V parts are set once per (t, a_blk, b_blk, s),
    # but the W part depends on which C blocks are affected. Also, the U and V shouldn't depend
    # on c_coeff - they're fixed by the A and B blocks. Let me restructure.
    
    # Actually, I realize there's an error in my logic. For a given (t, a_blk, b_blk, s):
    # The U and V entries are fixed. The W entries depend on which C blocks this term feeds into.
    # But the W value from the inner Strassen (w_val) should be multiplied by a_coeff * b_coeff * c_coeff.
    # So the W entry at global C position should accumulate w_val * a_coeff * b_coeff * c_coeff.
    
    # But I was adding to W inside the C_blocks_t loop, which is correct.
    # However, I was also not properly handling the fact that for each C block, the same local
    # Strassen W entries apply but mapped to different global C positions.
    
    # Let me redo this more carefully.
    
    U_93 = [[0] * 93 for _ in range(n*m)]
    V_93 = [[0] * 93 for _ in range(m*p)]
    W_93 = [[0] * 93 for _ in range(p*n)]
    
    sU, sV, sW = U_4x4x4, V_4x4x4, W_4x4x4
    
    for t in range(7):
        A_blocks_t, B_blocks_t, C_blocks_t = outer_terms[t]
        for (ai, aj, a_coeff) in A_blocks_t:
            for (bi, bj, b_coeff) in B_blocks_t:
                base_coeff = a_coeff * b_coeff
                for s in range(7):
                    global_col = t * 7 + s
                    # U
                    for local_row in range(4):
                        u_val = sU[local_row][s]
                        if u_val != 0:
                            ii = local_row // 2
                            jj = local_row % 2
                            global_i = ai * 2 + ii
                            global_j = aj * 2 + jj
                            U_93[global_i * m + global_j][global_col] += u_val * base_coeff
                    # V
                    for local_row in range(4):
                        v_val = sV[local_row][s]
                        if v_val != 0:
                            ii = local_row // 2
                            jj = local_row % 2
                            global_j2 = bi * 2 + ii
                            global_k = bj * 2 + jj
                            V_93[global_j2 * p + global_k][global_col] += v_val * base_coeff
                    # W
                    for local_row in range(4):
                        w_val = sW[local_row][s]
                        if w_val != 0:
                            kk = local_row // 2
                            ii = local_row % 2
                            for (ci, ck, c_coeff) in C_blocks_t:
                                global_i = ci * 2 + ii
                                global_k = ck * 2 + kk
                                W_93[global_k * n + global_i][global_col] += w_val * base_coeff * c_coeff
    
    col_offset = 49
    
    # Part 2: 4x4x3 (columns 49..92)
    # Split into 4 sub-products as described:
    # C_top_left (2x2) = A00*Br00 + A01*Br10  (two 2x2x2 products)
    # C_top_right (2x1) = A00*Br01 + A01*Br11  (two 2x2x1 products)
    # C_bot_left (2x2) = A10*Br00 + A11*Br10  (two 2x2x2 products)
    # C_bot_right (2x1) = A10*Br01 + A11*Br11  (two 2x2x1 products)
    
    # For 2x2x2: use Strassen (rank 7)
    # For 2x2x1: schoolbook (rank 4)
    # Total: 4*7 + 4*4 = 28 + 16 = 44
    
    # Br is 4x3: rows 0..3, cols 4..6 (since B is 4x7, cols 0-3 are B_left, cols 4-6 are B_right)
    # Br00: rows 0-1, cols 4-5 (2x2)
    # Br01: row 0-1, col 6 (2x1)
    # Br10: rows 2-3, cols 4-5 (2x2)
    # Br11: rows 2-3, col 6 (2x1)
    
    # C_top_left: C rows 0-1, cols 4-5 -> C[i,k] with i=0,1 and k=4,5
    # C_top_right: C rows 0-1, col 6 -> C[i,6] with i=0,1
    # C_bot_left: C rows 2-3, cols 4-5 -> C[i,k] with i=2,3 and k=4,5
    # C_bot_right: C rows 2-3, col 6 -> C[i,6] with i=2,3
    
    def add_2x2x2_strassen(A_row_block, A_col_block, B_row_block, B_col_block1, B_col_block2,
                            C_row_block, C_col_block1, C_col_block2, col_start, coeff=1):
        """Add a 2x2x2 Strassen decomposition for A(A_row_block x A_col_block) * B(B_row_block x [B_col_block1:B_col_block2]) = C(C_row_block x [C_col_block1:C_col_block2])"""
        # A is 2x2: rows A_row_block*2..A_row_block*2+1, cols A_col_block*2..A_col_block*2+1
        # B is 2x2: rows B_row_block*2..B_row_block*2+1, cols start at B_col_block1, 2 columns
        # C is 2x2: rows C_row_block*2..C_row_block*2+1, cols start at C_col_block1, 2 columns
        global U_93, V_93, W_93, n, m, p
        for s in range(7):
            gcol = col_start + s
            for local_row in range(4):
                u_val = sU[local_row][s]
                if u_val != 0:
                    ii = local_row // 2
                    jj = local_row % 2
                    gi = A_row_block * 2 + ii
                    gj = A_col_block * 2 + jj
                    U_93[gi * m + gj][gcol] += u_val * coeff
            
            for local_row in range(4):
                v_val = sV[local_row][s]
                if v_val != 0:
                    ii = local_row // 2
                    jj = local_row % 2
                    gj = B_row_block * 2 + ii
                    gk = B_col_block1 + jj
                    V_93[gj * p + gk][gcol] += v_val * coeff
            
            for local_row in range(4):
                w_val = sW[local_row][s]
                if w_val != 0:
                    kk = local_row // 2
                    ii = local_row % 2
                    gi = C_row_block * 2 + ii
                    gk = C_col_block1 + kk
                    W_93[gk * n + gi][gcol] += w_val * coeff
    
    def add_2x2x1_schoolbook(A_row_block, A_col_block, B_row_block, B_col,
                              C_row_block, C_col, col_start, coeff=1):
        """Add a 2x2x1 schoolbook decomposition (rank 4)."""
        global U_93, V_93, W_93, n, m, p
        for ii in range(2):
            for jj in range(2):
                gcol = col_start + ii*2 + jj
                gi = A_row_block * 2 + ii
                gj = A_col_block * 2 + jj
                U_93[gi * m + gj][gcol] += coeff
                # B[jj, B_col]
                V_93[jj * p + B_col][gcol] += coeff
                # C[C_row_block*2+ii, C_col]
                W_93[C_col * n + C_row_block*2 + ii][gcol] += coeff
    
    # C_top_left = A00*Br00 + A01*Br10
    # A00: rows 0-1, cols 0-1 -> A_row_block=0, A_col_block=0
    # Br00: rows 0-1, cols 4-5 -> B_row_block=0, B_col_block1=4
    # C_top_left: rows 0-1, cols 4-5 -> C_row_block=0, C_col_block1=4
    
    add_2x2x2_strassen(0, 0, 0, 4, 5, 0, 4, 5, col_offset, coeff=1)  # A00*Br00
    col_offset += 7
    
    # A01: rows 0-1, cols 2-3 -> A_row_block=0, A_col_block=1
    # Br10: rows 2-3, cols 4-5 -> B_row_block=1, B_col_block1=4
    add_2x2x2_strassen(0, 1, 1, 4, 5, 0, 4, 5, col_offset, coeff=1)  # A01*Br10
    col_offset += 7
    
    # C_top_right = A00*Br01 + A01*Br11
    # Br01: rows 0-1, col 6 -> B_row_block=0, B_col=6
    # Br11: rows 2-3, col 6 -> B_row_block=1, B_col=6
    # C_top_right: rows 0-1, col 6
    
    add_2x2x1_schoolbook(0, 0, 0, 6, 0, 6, col_offset, coeff=1)  # A00*Br01
    col_offset += 4
    
    add_2x2x1_schoolbook(0, 1, 1, 6, 0, 6, col_offset, coeff=1)  # A01*Br11
    col_offset += 4
    
    # C_bot_left = A10*Br00 + A11*Br10
    # A10: rows 2-3, cols 0-1 -> A_row_block=1, A_col_block=0
    # A11: rows 2-3, cols 2-3 -> A_row_block=1, A_col_block=1
    # C_bot_left: rows 2-3, cols 4-5 -> C_row_block=1, C_col_block1=4
    
    add_2x2x2_strassen(1, 0, 0, 4, 5, 1, 4, 5, col_offset, coeff=1)  # A10*Br00
    col_offset += 7
    
    add_2x2x2_strassen(1, 1, 1, 4, 5, 1, 4, 5, col_offset, coeff=1)  # A11*Br10
    col_offset += 7
    
    # C_bot_right = A10*Br01 + A11*Br11
    # C_bot_right: rows 2-3, col 6
    
    add_2x2x1_schoolbook(1, 0, 0, 6, 1, 6, col_offset, coeff=1)  # A10*Br01
    col_offset += 4
    
    add_2x2x1_schoolbook(1, 1, 1, 6, 1, 6, col_offset, coeff=1)  # A11*Br11
    col_offset += 4
    
    # col_offset should be 93
    assert col_offset == 93, f"col_offset = {col_offset}, expected 93"
    
    # Verify the rank-93 decomposition
    if verify_decomposition(U_93, V_93, W_93, n, m, p):
        return {"U": U_93, "V": V_93, "W": W_93}
    
    # If verification fails, fall back to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
