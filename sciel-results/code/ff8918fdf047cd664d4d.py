"""
Matrix multiplication 4x4x7 tensor decomposition.

Strategy:
1. Build the schoolbook decomposition (rank 112).
2. Use the Kronecker product structure: T_{4x4x7} = T_{2x2x2} ⊗ T_{2x2x1} (approximately).
   Actually, 4x4x7 = (2x2x2) ⊗ (2x2x1) + correction, or more precisely:
   4x4x7 can be viewed as multiplying a 4x4 by 4x7.
   
   The key insight: 4x4x7 = 2*(2x2x2) ⊗ 2*(2x2x2) for the 4x4x4 part (rank 4*4=16 for 2x2x2, so 16*16=256? No.)
   
   Let me think more carefully. The tensor for m×n×p multiplication has rank at most m*n*p (schoolbook).
   
   For 2x2x2: rank is 7 (Strassen).
   For 4x4x4: rank is at most 7^2 = 49 (using Strassen recursively).
   For 4x4x3: we need to handle the 3 columns.
   
   Actually, let me use a different decomposition:
   4x4x7 = 4x4x4 + 4x4x3
   
   For 4x4x4: use Strassen recursively: rank ≤ 7^2 = 49
   For 4x4x3: we can pad to 4x4x4 and use rank 49, but that's wasteful.
   
   Alternatively: 4x4x3 can be done by padding B to 4x4, computing 4x4x4, and discarding the last column.
   But that gives rank 49 for the 4x4x3 part too, total 98.
   
   Wait, that's not right. If we compute C = A*B where B is 4x7, we can split B into B1 (4x4) and B2 (4x3).
   C1 = A*B1 (4x4), C2 = A*B2 (4x3).
   
   For C1: rank 49 using Strassen.
   For C2: we can pad B2 to 4x4 (adding a zero column), compute A*B2_padded (4x4), discard last column.
   This also has rank 49.
   
   Total: 49 + 49 = 98.
   
   But wait, can we do better? The 4x4x3 tensor might have rank less than 49.
   
   Actually, let me think about this differently. The tensor for 4x4x7 multiplication:
   T[i,j,k] = 1 if the indices match appropriately.
   
   Using the Kronecker structure:
   4 = 2*2, so we can view A as a 2x2 block matrix with 2x2 blocks.
   B as a 2x2 block matrix with 2x4 blocks (since 7 = 2*2 + 2*1 + ... hmm, 7 is not a multiple of 2).
   
   Let me try: 7 = 2*2 + 2*2 + 2*1 = 4+4+1? No, 7 = 2+2+2+1.
   
   This is getting complicated. Let me just implement the Strassen-based approach:
   
   For 4x4x4: use 2x2x2 Strassen (rank 7) applied recursively.
   For 4x4x3: pad to 4x4x4, use Strassen, discard.
   
   Total rank: 49 + 49 = 98.
   
   But we can do better for 4x4x3. Let me think...
   
   Actually, 4x4x3: we can split A into two 2x4 matrices A1, A2, and B into a 4x3 matrix.
   C1 = A1*B (2x3), C2 = A2*B (2x3).
   Each 2x4x3 can be done with rank... let's see.
   
   2x4x3: schoolbook is 2*4*3 = 24. Can we do better?
   Pad to 2x4x4: rank 2*4*4 = 32 schoolbook, or use Strassen-like: 2x2x2 (rank 7) ⊗ 2x2x2 (rank 7) = 49? No, that's for 4x4x4.
   
   For 2x4x4: view as 2x2x2 ⊗ 2x2x2? No, 2x4x4 = (2x2x2) with some structure.
   
   This is getting too complex. Let me just go with the straightforward approach:
   
   4x4x4: Strassen rank 49
   4x4x3: pad to 4x4x4, Strassen rank 49, total 98
   
   But 98 is already better than 112. Let me implement this.
   
   Actually, I realize I need to be more careful. The Strassen decomposition for 2x2x2 gives rank 7.
   For 4x4x4, we apply it twice: rank 7*7 = 49.
   
   For the 4x4x3 part, if I pad B to 4x4, I get a 4x4x4 problem with rank 49, but the last column of the result is zero (since B's last column is zero), so I can discard it. The rank is still 49 for the computation, but the output is 4x3.
   
   So total rank: 49 + 49 = 98.
   
   Let me implement the Strassen decomposition for 2x2x2 and use it to build 4x4x4 and 4x4x3.
"""

import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    # Build the 2x2x2 Strassen decomposition (rank 7)
    # For C = A*B where A, B, C are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # The rank-1 terms for 2x2x2:
    # Each M corresponds to a rank-1 term in the tensor.
    # M1: (A00+A11) * (B00+B11) -> contributes to C00, C11
    # M2: (A10+A11) * B00 -> contributes to C10, C11
    # M3: A00 * (B10-B11) -> contributes to C01, C11
    # M4: A11 * (B01-B10) -> contributes to C00, C10
    # M5: (A00+A01) * B11 -> contributes to C00, C01
    # M6: (A10-A00) * (B00+B10) -> contributes to C11
    # M7: (A00+A01) * (B01+B11) -> contributes to C00
    
    # Let me define the Strassen decomposition more carefully.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now, for 4x4x4, we view A, B, C as 2x2 block matrices with 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # We apply Strassen at the block level:
    # M1 = (A00+A11)(B00+B11)  -> 2x2 result
    # M2 = (A10+A11)B00         -> 2x2 result
    # M3 = A00(B10-B11)         -> 2x2 result
    # M4 = A11(B01-B10)         -> 2x2 result
    # M5 = (A00+A01)B11         -> 2x2 result
    # M6 = (A10-A00)(B00+B10)   -> 2x2 result
    # M7 = (A01-A11)(B01+B11)   -> 2x2 result
    
    # Each Mj is a 2x2 matrix product, which we compute using Strassen (rank 7).
    # So total rank: 7 * 7 = 49.
    
    # Let me build the 4x4x4 decomposition using this approach.
    
    # First, let's define the 2x2x2 Strassen decomposition as a list of rank-1 terms.
    # Each rank-1 term is (U_factor, V_factor, W_factor) where:
    # U_factor is a vector of length 4 (for A's entries)
    # V_factor is a vector of length 4 (for B's entries)
    # W_factor is a vector of length 4 (for C's entries)
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    def strassen_2x2x2():
        """Return the 7 rank-1 terms for 2x2x2 Strassen decomposition."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1, others 0
        # V: B00=1, B11=1, others 0
        # W: C00=1, C11=1, others 0
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))  # M1: C00, C11
        
        # M2 = (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1, C11=1
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,1]))  # M2: C10, C11
        
        # M3 = A00(B10-B11)
        # U: A00=1
        # V: B10=1, B11=-1
        # W: C01=1, C11=1
        terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))  # M3: C01, C11
        
        # M4 = A11(B01-B10)
        # U: A11=1
        # V: B01=1, B10=-1
        # W: C00=1, C10=1
        terms.append(([0,0,0,1], [0,1,-1,0], [1,0,1,0]))  # M4: C00, C10
        
        # M5 = (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=1, C01=1
        terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))  # M5: C00, C01
        
        # M6 = (A10-A00)(B00+B10)
        # U: A10=1, A00=-1
        # V: B00=1, B10=1
        # W: C11=1
        terms.append(([-1,0,1,0], [1,0,1,0], [0,0,0,1]))  # M6: C11
        
        # M7 = (A01-A11)(B01+B11)
        # U: A01=1, A11=-1
        # V: B01=1, B11=1
        # W: C00=1
        terms.append(([0,1,0,-1], [0,1,0,1], [1,0,0,0]))  # M7: C00
        
        return terms
    
    # Now, for 4x4x4, we need to build the decomposition.
    # A is 4x4, viewed as 2x2 blocks of 2x2.
    # A_block[i,j] is a 2x2 matrix, with global indices:
    # A_block[0,0] = A[0:2, 0:2] -> indices 0,1,4,5
    # A_block[0,1] = A[0:2, 2:4] -> indices 2,3,6,7
    # A_block[1,0] = A[2:4, 0:2] -> indices 8,9,12,13
    # A_block[1,1] = A[2:4, 2:4] -> indices 10,11,14,15
    
    # B is 4x4, viewed as 2x2 blocks of 2x2.
    # B_block[i,j] is a 2x2 matrix, with global indices:
    # B_block[0,0] = B[0:2, 0:2] -> indices 0,1,4,5
    # B_block[0,1] = B[0:2, 2:4] -> indices 2,3,6,7
    # B_block[1,0] = B[2:4, 0:2] -> indices 8,9,12,13
    # B_block[1,1] = B[2:4, 2:4] -> indices 10,11,14,15
    
    # C is 4x4, viewed as 2x2 blocks of 2x2.
    # C_block[i,j] is a 2x2 matrix, with global indices:
    # C_block[0,0] = C[0:2, 0:2] -> indices 0,1,4,5
    # C_block[0,1] = C[0:2, 2:4] -> indices 2,3,6,7
    # C_block[1,0] = C[2:4, 0:2] -> indices 8,9,12,13
    # C_block[1,1] = C[2:4, 2:4] -> indices 10,11,14,15
    
    # The global index for A[i,j] is i*4+j.
    # The global index for B[i,j] is i*4+j.
    # The global index for C[i,j] is i*4+j.
    
    # For the 4x4x4 decomposition, we have 49 rank-1 terms.
    # Each term corresponds to a pair (block_strassen_term, inner_strassen_term).
    
    # Let's build the 4x4x4 decomposition.
    block_terms = strassen_2x2x2()  # 7 terms
    inner_terms = strassen_2x2x2()  # 7 terms
    
    U_4x4x4 = []
    V_4x4x4 = []
    W_4x4x4 = []
    
    for bt in block_terms:
        bu, bv, bw = bt
        for it in inner_terms:
            iu, iv, iw = it
            # bu is the block-level A factor: [bu00, bu01, bu10, bu11]
            # iu is the inner-level A factor: [iu00, iu01, iu10, iu11]
            # The combined A factor for the 4x4 matrix:
            # A_global[i,j] where i = block_row*2 + inner_row, j = block_col*2 + inner_col
            # The factor is bu[block_row, block_col] * iu[inner_row, inner_col]
            
            U_vec = [0]*16
            V_vec = [0]*16
            W_vec = [0]*16
            
            for br in range(2):
                for bc in range(2):
                    bu_val = bu[br*2+bc]
                    if bu_val == 0:
                        continue
                    for ir in range(2):
                        for ic in range(2):
                            iu_val = iu[ir*2+ic]
                            if iu_val == 0:
                                continue
                            global_i = br*2 + ir
                            global_j = bc*2 + ic
                            idx = global_i*4 + global_j
                            U_vec[idx] += bu_val * iu_val
                            V_vec[idx] += bv[br*2+bc] * iv[ir*2+ic]
                            W_vec[idx] += bw[br*2+bc] * iw[ir*2+ic]
            
            U_4x4x4.append(U_vec)
            V_4x4x4.append(V_vec)
            W_4x4x4.append(W_vec)
    
    # Now, for 4x4x3, we pad B to 4x4 (adding a zero column), compute 4x4x4, and discard the last column of C.
    # The 4x4x3 tensor: A is 4x4, B is 4x3, C is 4x3.
    # Global indices:
    # A[i,j]: i*4+j, i in [0,4), j in [0,4)
    # B[i,j]: i*4+j, i in [0,4), j in [0,3)
    # C[i,j]: i*4+j, i in [0,4), j in [0,3)
    
    # We pad B to 4x4 by adding a zero column (j=3).
    # Then compute C_padded = A * B_padded (4x4), and discard C_padded[:,3].
    
    # The 4x4x4 decomposition gives us 49 terms for the padded problem.
    # For the 4x4x3 problem, we use the same 49 terms, but:
    # - B's global index is i*4+j for j in [0,3), and B[i,3]=0.
    # - C's global index is i*4+j for j in [0,3), and we discard C[i,3].
    
    # So the V factor for B: V_vec[i*4+j] for j in [0,3), and V_vec[i*4+3]=0.
    # The W factor for C: W_vec[i*4+j] for j in [0,3), and we discard W_vec[i*4+3].
    
    # Let's build the 4x4x3 decomposition.
    U_4x4x3 = []
    V_4x4x3 = []
    W_4x4x3 = []
    
    for U_vec, V_vec, W_vec in zip(U_4x4x4, V_4x4x4, W_4x4x4):
        U_new = U_vec[:]  # A is still 4x4
        V_new = [0]*12  # B is 4x3
        W_new = [0]*12  # C is 4x3
        
        for i in range(4):
            for j in range(3):
                V_new[i*4+j] = V_vec[i*4+j]
                W_new[i*4+j] = W_vec[i*4+j]
        
        U_4x4x3.append(U_new)
        V_4x4x3.append(V_new)
        W_4x4x3.append(W_new)
    
    # Now, combine the 4x4x4 and 4x4x3 decompositions.
    # The 4x4x7 problem: A is 4x4, B is 4x7, C is 4x7.
    # Split B into B1 (4x4) and B2 (4x3).
    # C1 = A*B1 (4x4), C2 = A*B2 (4x3).
    # C = [C1 | C2] (4x7).
    
    # Global indices:
    # A[i,j]: i*4+j, i in [0,4), j in [0,4)
    # B[i,j]: i*7+j, i in [0,4), j in [0,7)
    # C[i,j]: i*7+j, i in [0,4), j in [0,7)
    
    # For the 4x4x4 part (B1, C1):
    # B1[i,j] = B[i,j] for j in [0,4), global B index: i*7+j
    # C1[i,j] = C[i,j] for j in [0,4), global C index: i*7+j
    
    # For the 4x4x3 part (B2, C2):
    # B2[i,j] = B[i,j+4] for j in [0,3), global B index: i*7+(j+4)
    # C2[i,j] = C[i,j+4] for j in [0,3), global C index: i*7+(j+4)
    
    # Let's build the combined decomposition.
    U_combined = []
    V_combined = []
    W_combined = []
    
    # Add 4x4x4 terms
    for U_vec, V_vec, W_vec in zip(U_4x4x4, V_4x4x4, W_4x4x4):
        U_new = [0]*16
        V_new = [0]*28
        W_new = [0]*28
        
        for i in range(4):
            for j in range(4):
                U_new[i*4+j] = U_vec[i*4+j]
                V_new[i*7+j] = V_vec[i*4+j]
                W_new[i*7+j] = W_vec[i*4+j]
        
        U_combined.append(U_new)
        V_combined.append(V_new)
        W_combined.append(W_new)
    
    # Add 4x4x3 terms
    for U_vec, V_vec, W_vec in zip(U_4x4x3, V_4x4x3, W_4x4x3):
        U_new = [0]*16
        V_new = [0]*28
        W_new = [0]*28
        
        for i in range(4):
            for j in range(4):
                U_new[i*4+j] = U_vec[i*4+j]
            for j in range(3):
                V_new[i*7+(j+4)] = V_vec[i*4+j]
                W_new[i*7+(j+4)] = W_vec[i*4+j]
        
        U_combined.append(U_new)
        V_combined.append(V_new)
        W_combined.append(W_new)
    
    # Convert to numpy arrays
    U = np.array(U_combined, dtype=object)
    V = np.array(V_combined, dtype=object)
    W = np.array(W_combined, dtype=object)
    
    rank = len(U_combined)
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    expected = 1 if (a % m == b // p and a // m == c % n and b % p == c // n) else 0
                    actual = 0
                    for l in range(rank):
                        actual += U_mat[a, l] * V_mat[b, l] * W_mat[c, l]
                    if actual != expected:
                        return False
        return True
    
    # Check if the decomposition is valid
    if not verify(U, V, W, rank):
        # Fall back to schoolbook
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb), dtype=object)
        V = np.zeros((mp, rank_sb), dtype=object)
        W = np.zeros((pn, rank_sb), dtype=object)
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, column] = 1
                    V[b, column] = 1
                    W[c, column] = 1
                    column += 1
        rank = rank_sb
    
    # Convert to list of lists with Fraction for exactness
    U_list = [[int(U[i, j]) if U[i, j] == int(U[i, j]) else Fraction(U[i, j]) for j in range(rank)] for i in range(nm)]
    V_list = [[int(V[i, j]) if V[i, j] == int(V[i, j]) else Fraction(V[i, j]) for j in range(rank)] for i in range(mp)]
    W_list = [[int(W[i, j]) if W[i, j] == int(W[i, j]) else Fraction(W[i, j]) for j in range(rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
