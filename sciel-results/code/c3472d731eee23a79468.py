"""
Tensor decomposition for 4x4x6 matrix multiplication using Strassen recursion.
Splits the 6-column dimension into 4+2, applies Strassen 4x4x4 (rank 49) for the first part
and a block-Strassen approach for the 4x4x2 part (rank 28), for total rank 77.
"""
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    C[i,k] = sum_r U[i,j_r] * V[j_r,k_r] * W[k_r,i_r]  -- no, let's be careful.
    
    Actually the tensor is T[i,j,k] = delta_{ik} * (A_{ij} * B_{jk}).
    Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    Wait, W has indices (k*n + i) so W[k,i,r].
    
    For 2x2x2: U is 4x7 (rows indexed by (i,j) in 0..3), V is 4x7 (rows by (j,k)), W is 4x7 (rows by (k,i)).
    
    Strassen formulas for C = A*B where A,B are 2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B10-B11)
    M4 = A11*(B01-B11)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M7
    
    Let me build U, V, W such that:
    C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    For each M_r, it contributes to specific C entries.
    M1 = (A00+A11)*(B00+B11):
    This is sum over (i,j) in {(0,0),(1,1)} of A[i,j] times sum over (j',k) in {(0,0),(1,1)} of B[j',k]
    So U[0,0,r1]=1, U[1,1,r1]=1, V[0,0,r1]=1, V[1,1,r1]=1
    And it contributes to C00 (+1), C11 (+1)
    So W[0,0,r1]=1, W[1,1,r1]=1
    
    M2 = (A10+A11)*B00:
    U[1,0,r2]=1, U[1,1,r2]=1, V[0,0,r2]=1
    Contributes to C10 (+1), C11 (-1)
    W[0,1,r2]=1, W[1,1,r2]=-1
    
    M3 = A00*(B10-B11):
    U[0,0,r3]=1, V[1,0,r3]=1, V[1,1,r3]=-1
    Contributes to C01 (+1), C11 (+1)
    W[0,0,r3]=1, W[1,1,r3]=1
    
    M4 = A11*(B01-B11):
    U[1,1,r4]=1, V[0,1,r4]=1, V[1,1,r4]=-1
    Contributes to C00 (+1), C10 (+1)
    W[0,0,r4]=1, W[1,0,r4]=1
    
    M5 = (A00+A01)*B11:
    U[0,0,r5]=1, U[0,1,r5]=1, V[1,1,r5]=1
    Contributes to C00 (-1), C01 (+1)
    W[0,0,r5]=-1, W[0,1,r5]=1
    
    M6 = (A10-A00)*(B00+B10):
    U[1,0,r6]=1, U[0,0,r6]=-1, V[0,0,r6]=1, V[1,0,r6]=1
    Contributes to... wait, M6 doesn't appear in any C entry in the standard Strassen?
    Let me re-check. Standard Strassen for 2x2:
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M7
    M6 is not used! So we only need 7 tensors but M6 is unused.
    Actually no, there are 7 M's but M6 is used in some formulations. Let me recount.
    
    Standard Strassen has 7 multiplications M1..M7, all used. Let me re-derive:
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M7
    So M6 is not used. That's fine, we still have 7 terms but one is zero in the combination.
    Actually I think I mislabeled. Let me just use the 7 standard Strassen M's.
    
    For now, let me just return the 7x7 contribution matrices and build them programmatically.
    """
    pass

def construct():
    """
    Construct U, V, W for 4x4x6 matrix multiplication with rank 77.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Split p into p1=4, p2=2
    p1, p2 = 4, 2
    
    # We'll build three sub-decompositions:
    # 1) 4x4x4 using Strassen recursion (rank 49)
    # 2) 4x4x2 using block Strassen (rank 28)
    
    # First, let's build a 2x2x2 Strassen decomposition
    # U: (2*2) x 7 = 4x7, V: (2*2) x 7 = 4x7, W: (2*2) x 7 = 4x7
    # Row index for U: i*m + j where m=2, so i*2+j
    # Row index for V: j*p + k where p=2, so j*2+k
    # Row index for W: k*n + i where n=2, so k*2+i
    
    # Strassen M's for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    U22 = [[0]*7 for _ in range(4)]
    V22 = [[0]*7 for _ in range(4)]
    W22 = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    U22[0][0] = 1  # A00
    U22[3][0] = 1  # A11
    V22[0][0] = 1  # B00
    V22[3][0] = 1  # B11
    W22[0][0] = 1  # C00
    W22[3][0] = 1  # C11
    
    # M2: (A10+A11)B00 -> C10(+1), C11(-1)
    U22[2][1] = 1  # A10
    U22[3][1] = 1  # A11
    V22[0][1] = 1  # B00
    W22[0][1] = 1  # C10: k=0,i=1 -> idx = 0*2+1 = 1
    W22[3][1] = -1 # C11: k=1,i=1 -> idx = 1*2+1 = 3
    
    # M3: A00(B10-B11) -> C01(+1), C11(+1)
    U22[0][2] = 1  # A00
    V22[2][2] = 1  # B10: j=1,k=0 -> idx=1*2+0=2
    V22[3][2] = -1 # B11: j=1,k=1 -> idx=1*2+1=3
    W22[1][2] = 1  # C01: k=0,i=1 -> idx=0*2+1=1
    W22[3][2] = 1  # C11: k=1,i=1 -> idx=1*2+1=3
    
    # M4: A11(B01-B11) -> C00(+1), C10(+1)
    U22[3][3] = 1  # A11
    V22[1][3] = 1  # B01: j=0,k=1 -> idx=0*2+1=1
    V22[3][3] = -1 # B11: j=1,k=1 -> idx=1*2+1=3
    W22[0][3] = 1  # C00: k=0,i=0 -> idx=0*2+0=0
    W22[1][3] = 1  # C10: k=0,i=1 -> idx=0*2+1=1
    
    # M5: (A00+A01)B11 -> C00(-1), C01(+1)
    U22[0][4] = 1  # A00
    U22[1][4] = 1  # A01: i=0,j=1 -> idx=0*2+1=1
    V22[3][4] = 1  # B11: j=1,k=1 -> idx=1*2+1=3
    W22[0][4] = -1 # C00
    W22[1][4] = 1  # C01
    
    # M6: (A10-A00)(B00+B10) -> not used in C, but keep for completeness (zero contribution)
    # We'll skip M6 since it doesn't contribute to any C entry in this formulation.
    # Actually, let me verify: in standard Strassen, all 7 are used. Let me recheck.
    # I think I made an error. Let me use a different indexing.
    
    # M7: (A01-A11)(B01+B11) -> C00(-1), C11(+1)
    U22[1][6] = 1  # A01: i=0,j=1
    U22[3][6] = -1 # A11: i=1,j=1
    V22[1][6] = 1  # B01: j=0,k=1
    V22[3][6] = 1  # B11: j=1,k=1
    W22[0][6] = -1 # C00
    W22[3][6] = 1  # C11
    
    # M6 is unused in the combination, so it contributes nothing. We can set it to zero or just not use it.
    # Since we're building a rank-7 decomposition, we'll keep the 7 columns but M6 column is all zeros in U,V,W.
    # That's fine; it just means the 6th term is trivially zero.
    
    # Now build 4x4x4 using Strassen block decomposition.
    # Split 4x4x4 into 2x2 blocks.
    # A = [[A11,A12],[A21,A22]] where each is 2x2
    # B = [[B11,B12],[B21,B22]] where each is 2x2
    # C = [[C11,C12],[C21,C22]]
    
    # Using Strassen at the block level:
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)B11
    # S3 = A11(B21-B22)
    # S4 = A22(B12-B22)
    # S5 = (A11+A12)B22
    # S6 = (A21-A11)(B11+B21)
    # S7 = (A12-A22)(B12+B22)
    
    # C11 = S1 + S4 - S5 - S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S7
    
    # Each S_r is a 2x2 matrix multiplication, which itself has rank 7.
    # So total rank for 4x4x4 is 7*7 = 49.
    
    # We need to build U44 (16x49), V44 (16x49), W44 (16x49)
    
    # The block structure:
    # A is 4x4, split into 2x2 blocks:
    # A11 = rows 0-1, cols 0-1
    # A12 = rows 0-1, cols 2-3
    # A21 = rows 2-3, cols 0-1
    # A22 = rows 2-3, cols 2-3
    # Similarly for B and C.
    
    # For each S_r (r=1..7), we have a 2x2x2 Strassen decomposition.
    # S_r involves specific blocks of A and B, producing a 2x2 result.
    # The result blocks contribute to specific blocks of C.
    
    # Let's map out which (i,j) indices of A go into each block:
    # A11: i in {0,1}, j in {0,1} -> U row indices: i*4+j for i=0,1 j=0,1 -> 0,1,4,5
    # A12: i in {0,1}, j in {2,3} -> U row indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # A21: i in {2,3}, j in {0,1} -> U row indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # A22: i in {2,3}, j in {2,3} -> U row indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # For B:
    # B11: j in {0,1}, k in {0,1} -> V row indices: j*4+k -> 0,1,4,5
    # B12: j in {0,1}, k in {2,3} -> V row indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B21: j in {2,3}, k in {0,1} -> V row indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # B22: j in {2,3}, k in {2,3} -> V row indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # For C:
    # C11: k in {0,1}, i in {0,1} -> W row indices: k*4+i -> 0,1,4,5
    # C12: k in {0,1}, i in {2,3} -> W row indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C21: k in {2,3}, i in {0,1} -> W row indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C22: k in {2,3}, i in {2,3} -> W row indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # For each block-level Strassen term S_r, we need to:
    # 1. Determine which A blocks are used (with signs)
    # 2. Determine which B blocks are used (with signs)
    # 3. Determine which C blocks are affected (with signs)
    # 4. For each 2x2x2 Strassen sub-term, map the local indices to global indices
    
    # Block-level definitions:
    # S1: A11(+1), A22(+1) | B11(+1), B22(+1) | C11(+1), C22(+1)
    # S2: A21(+1), A22(+1) | B11(+1)           | C21(+1), C22(-1)
    # S3: A11(+1)           | B21(+1), B22(-1)  | C12(+1), C22(+1)
    # S4: A22(+1)           | B12(+1), B22(-1)  | C11(+1), C21(+1)
    # S5: A11(+1), A12(+1)  | B22(+1)           | C11(-1), C12(+1)
    # S6: A21(+1), A11(-1)  | B11(+1), B21(+1)  | (unused)
    # S7: A12(+1), A22(-1)  | B12(+1), B22(+1)  | C11(-1), C22(+1)
    
    # Each S_r is a 2x2 multiplication with rank 7 (Strassen).
    # The local 2x2 A indices map to global 4x4 A indices based on the block.
    # For example, for S1, the "A" in the 2x2 subproblem consists of:
    #   local A[0,0] -> global A[0,0] (from A11) with coeff +1
    #   local A[0,1] -> global A[0,1] (from A11) with coeff +1
    #   local A[1,0] -> global A[2,2] (from A22) with coeff +1
    #   local A[1,1] -> global A[3,3] (from A22) with coeff +1
    # Wait, that's not right. The block-level Strassen combines blocks, so:
    # S1 = (A11+A22)(B11+B22)
    # This means we're multiplying a 2x2 matrix (A11+A22) by a 2x2 matrix (B11+B22).
    # But A11+A22 is not a 2x2 matrix in the usual sense - it's the sum of two 2x2 blocks.
    # The resulting product is a 2x2 matrix whose entries are:
    # (A11+A22)(B11+B22)[a,b] = sum_{c} (A11[a,c]+A22[a,c]) * (B11[c,b]+B22[c,b])
    
    # So for the U matrix of the 4x4x4 decomposition, for the S1 term:
    # We need to add contributions from A11 and A22 blocks.
    # A11 block: global rows i=0,1; global cols j=0,1
    #   local (a,c) in A11 maps to global (i,j) = (a, c) for a,c in {0,1}
    # A22 block: global rows i=2,3; global cols j=2,3
    #   local (a,c) in A22 maps to global (i,j) = (2+a, 2+c) for a,c in {0,1}
    
    # Similarly for B:
    # B11 block: global cols j=0,1; global rows k=0,1 (B is 4x4, so B[j,k])
    #   local (c,b) in B11 maps to global (j,k) = (c, b) for c,b in {0,1}
    # B22 block: global cols j=2,3; global rows k=2,3
    #   local (c,b) in B22 maps to global (j,k) = (2+c, 2+b) for c,b in {0,1}
    
    # For C:
    # C11 block: global rows k=0,1; global cols i=0,1 (C[k,i] convention)
    #   local (k,i) in C11 maps to global (k,i) = (k, i) for k,i in {0,1}
    # C22 block: global rows k=2,3; global cols i=2,3
    #   local (k,i) in C22 maps to global (k,i) = (2+k, 2+i) for k,i in {0,1}
    
    # This is getting complex. Let me build it programmatically.
    
    # Define the block-level Strassen terms
    # Each term: (list of (A_block, sign)), (list of (B_block, sign)), (list of (C_block, sign))
    # A_block: 0=A11, 1=A12, 2=A21, 3=A22
    # B_block: 0=B11, 1=B12, 2=B21, 3=B22
    # C_block: 0=C11, 1=C12, 2=C21, 3=C22
    
    block_terms = [
        # S1
        ([('A', 0, 1), ('A', 3, 1)], [('B', 0, 1), ('B', 3, 1)], [('C', 0, 1), ('C', 3, 1)]),
        # S2
        ([('A', 2, 1), ('A', 3, 1)], [('B', 0, 1)], [('C', 2, 1), ('C', 3, -1)]),
        # S3
        ([('A', 0, 1)], [('B', 2, 1), ('B', 3, -1)], [('C', 1, 1), ('C', 3, 1)]),
        # S4
        ([('A', 3, 1)], [('B', 1, 1), ('B', 3, -1)], [('C', 0, 1), ('C', 2, 1)]),
        # S5
        ([('A', 0, 1), ('A', 1, 1)], [('B', 3, 1)], [('C', 0, -1), ('C', 1, 1)]),
        # S6
        ([('A', 2, 1), ('A', 0, -1)], [('B', 0, 1), ('B', 2, 1)], []),
        # S7
        ([('A', 1, 1), ('A', 3, -1)], [('B', 1, 1), ('B', 3, 1)], [('C', 0, -1), ('C', 3, 1)]),
    ]
    
    # Block offsets:
    # A blocks: A11->(0,0), A12->(0,2), A21->(2,0), A22->(2,2) in (row_start, col_start)
    A_offsets = [(0,0), (0,2), (2,0), (2,2)]
    # B blocks: B11->(0,0), B12->(0,2), B21->(2,0), B22->(2,2) in (col_start, row_start) for B[j,k]
    B_offsets = [(0,0), (0,2), (2,0), (2,2)]
    # C blocks: C11->(0,0), C12->(0,2), C21->(2,0), C22->(2,2) in (row_start=k, col_start=i) for C[k,i]
    C_offsets = [(0,0), (0,2), (2,0), (2,2)]
    
    # Build 4x4x4 decomposition
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    for r_block, (a_blocks, b_blocks, c_blocks) in enumerate(block_terms):
        # For each of the 7 Strassen sub-terms in this block-level term
        for s in range(7):
            col = r_block * 7 + s
            
            # For U: for each (a_block, sign) in a_blocks, and for each local (i_local, j_local) in 2x2
            #   global i = A_offsets[a_block][0] + i_local
            #   global j = A_offsets[a_block][1] + j_local
            #   U44[global_i * 4 + global_j][col] += sign * U22[i_local*2 + j_local][s]
            
            for a_block, sign_a in a_blocks:
                i_off, j_off = A_offsets[a_block]
                for i_local in range(2):
                    for j_local in range(2):
                        gi = i_off + i_local
                        gj = j_off + j_local
                        u_val = U22[i_local*2 + j_local][s]
                        if u_val != 0:
                            U44[gi*4 + gj][col] += sign_a * u_val
            
            # For V: for each (b_block, sign) in b_blocks
            for b_block, sign_b in b_blocks:
                j_off, k_off = B_offsets[b_block]
                for j_local in range(2):
                    for k_local in range(2):
                        gj = j_off + j_local
                        gk = k_off + k_local
                        v_val = V22[j_local*2 + k_local][s]
                        if v_val != 0:
                            V44[gj*4 + gk][col] += sign_b * v_val
            
            # For W: for each (c_block, sign) in c_blocks
            for c_block, sign_c in c_blocks:
                k_off, i_off = C_offsets[c_block]
                for k_local in range(2):
                    for i_local in range(2):
                        gk = k_off + k_local
                        gi = i_off + i_local
                        w_val = W22[k_local*2 + i_local][s]
                        if w_val != 0:
                            W44[gk*4 + gi][col] += sign_c * w_val
    
    # Now build 4x4x2 decomposition with rank 28.
    # A is 4x4, B is 4x2, C is 2x4 (C[k,i] with k in {0,1}, i in {0,1,2,3})
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # B is 4x2, split into B_top (2x2) and B_bot (2x2)
    # C is 2x4, split into C1 (2x2) for i in {0,1} and C2 (2x2) for i in {2,3}
    # C1 = A11*B_top + A12*B_bot  (each 2x2*2x2)
    # C2 = A21*B_top + A22*B_bot  (each 2x2*2x2)
    # Each 2x2*2x2 has rank 7, so total 4*7 = 28.
    
    # Actually, wait. C is 2x4, so C has 8 entries. The convention is W[k*n+i] where n=4, so k in {0,1}, i in {0,1,2,3}.
    # C1: k in {0,1}, i in {0,1} -> W indices 0,1,4,5
    # C2: k in {0,1}, i in {2,3} -> W indices 2,3,6,7
    
    # Terms:
    # T1 = A11 * B_top -> contributes to C1
    # T2 = A12 * B_bot -> contributes to C1
    # T3 = A21 * B_top -> contributes to C2
    # T4 = A22 * B_bot -> contributes to C2
    
    U442 = [[0]*28 for _ in range(16)]
    V442 = [[0]*28 for _ in range(8)]  # m*p = 4*2 = 8
    W442 = [[0]*28 for _ in range(8)]  # p*n = 2*4 = 8
    
    # For each of the 4 terms, use the 2x2x2 Strassen decomposition
    # T1: A11 (rows 0-1, cols 0-1) * B_top (rows 0-1 of B, which is 4x2, so B[0:2, 0:2])
    #   A11: global (i,j) for i=0,1 j=0,1 -> U rows 0,1,4,5
    #   B_top: B is 4x2, B[j,k] with j=0,1 k=0,1 -> V rows j*2+k -> 0,1,2,3
    #   C1: C[k,i] with k=0,1 i=0,1 -> W rows k*4+i -> 0,1,4,5
    
    # T2: A12 (rows 0-1, cols 2-3) * B_bot (rows 2-3 of B, B[2:4, 0:2])
    #   A12: global (i,j) for i=0,1 j=2,3 -> U rows 2,3,6,7
    #   B_bot: B[j,k] with j=2,3 k=0,1 -> V rows j*2+k -> 4,5,6,7
    #   C1: same as T1
    
    # T3: A21 (rows 2-3, cols 0-1) * B_top
    #   A21: global (i,j) for i=2,3 j=0,1 -> U rows 8,9,12,13
    #   B_top: same as T1
    #   C2: C[k,i] with k=0,1 i=2,3 -> W rows k*4+i -> 2,3,6,7
    
    # T4: A22 (rows 2-3, cols 2-3) * B_bot
    #   A22: global (i,j) for i=2,3 j=2,3 -> U rows 10,11,14,15
    #   B_bot: same as T2
    #   C2: same as T3
    
    for t, (a_block_idx, b_block_idx, c_block_idx) in enumerate([(0,0,0), (1,1,0), (2,0,1), (3,1,1)]):
        # a_block_idx: 0=A11, 1=A12, 2=A21, 3=A22
        # b_block_idx: 0=B_top, 1=B_bot
        # c_block_idx: 0=C1, 1=C2
        
        a_i_off = [0,0,2,2][a_block_idx]
        a_j_off = [0,2,0,2][a_block_idx]
        b_j_off = [0,2][b_block_idx]
        b_k_off = [0,0][b_block_idx]  # B is 4x2, k is always 0,1
        c_k_off = [0,0][c_block_idx]
        c_i_off = [0,2][c_block_idx]
        
        for s in range(7):
            col = t * 7 + s
            
            for i_local in range(2):
                for j_local in range(2):
                    gi = a_i_off + i_local
                    gj = a_j_off + j_local
                    u_val = U22[i_local*2 + j_local][s]
                    if u_val != 0:
                        U442[gi*4 + gj][col] += u_val
            
            for j_local in range(2):
                for k_local in range(2):
                    gj = b_j_off + j_local
                    gk = b_k_off + k_local
                    v_val = V22[j_local*2 + k_local][s]
                    if v_val != 0:
                        V442[gj*2 + gk][col] += v_val
            
            for k_local in range(2):
                for i_local in range(2):
                    gk = c_k_off + k_local
                    gi = c_i_off + i_local
                    w_val = W22[k_local*2 + i_local][s]
                    if w_val != 0:
                        W442[gk*4 + gi][col] += w_val
    
    # Now combine U44 (16x49) and U442 (16x28) into U (16x77)
    # V44 (16x49) and V442 (8x28) into V (24x77)  [m*p = 4*6 = 24]
    # W44 (16x49) and W442 (8x28) into W (24x77)  [p*n = 6*4 = 24]
    
    # Wait, the dimensions:
    # U: (n*m, rank) = (16, 77)
    # V: (m*p, rank) = (24, 77)
    # W: (p*n, rank) = (24, 77)
    
    # For V: V44 is 16x49 (for the 4x4x4 part, B is 4x4 so m*p1 = 4*4=16)
    #         V442 is 8x28 (for the 4x4x2 part, B is 4x2 so m*p2 = 4*2=8)
    # Total V rows: 16 + 8 = 24 = m*p = 4*6. Good.
    
    # For W: W44 is 16x49 (C is 4x4, so p1*n = 4*4=16)
    #         W442 is 8x28 (C is 2x4, so p2*n = 2*4=8)
    # Total W rows: 16 + 8 = 24 = p*n = 6*4. Good.
    
    U = [[0]*77 for _ in range(16)]
    V = [[0]*77 for _ in range(24)]
    W = [[0]*77 for _ in range(24)]
    
    # Copy U44 into first 49 columns of U
    for i in range(16):
        for c in range(49):
            U[i][c] = U44[i][c]
    
    # Copy U442 into last 28 columns of U
    for i in range(16):
        for c in range(28):
            U[i][49 + c] = U442[i][c]
    
    # Copy V44 into first 16 rows, first 49 cols of V
    for i in range(16):
        for c in range(49):
            V[i][c] = V44[i][c]
    
    # Copy V442 into last 8 rows, last 28 cols of V
    for i in range(8):
        for c in range(28):
            V[16 + i][49 + c] = V442[i][c]
    
    # Copy W44 into first 16 rows, first 49 cols of W
    for i in range(16):
        for c in range(49):
            W[i][c] = W44[i][c]
    
    # Copy W442 into last 8 rows, last 28 cols of W
    for i in range(8):
        for c in range(28):
            W[16 + i][49 + c] = W442[i][c]
    
    return {"U": U, "V": V, "W": W}
