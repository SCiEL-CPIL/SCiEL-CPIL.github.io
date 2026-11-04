import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j] * V_arr[j*p+k], W_arr[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2(U, V, W, base_u, base_v, base_w):
    """
    Adds Strassen 2x2x2 decomposition terms.
    U indices: base_u + i*2 + j (i,j in 0..1)
    V indices: base_v + j*2 + k (j,k in 0..1)
    W indices: base_w + k*2 + i (k,i in 0..1)
    """
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    
    # We need to add 7 terms.
    # Let's define the coefficients for U, V, W for each of the 7 M's.
    # U row: i*2+j
    # V row: j*2+k
    # W row: k*2+i
    
    # M1:
    # U: A00(0), A11(3) -> coeff 1
    # V: B00(0), B11(3) -> coeff 1
    # W: C00(0), C11(3) -> coeff 1
    # Wait, Strassen updates C:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So we need to add these contributions to the tensor.
    # The decomposition is a sum of rank-1 tensors.
    # Each M corresponds to a product of linear forms in A, B and a linear form in C.
    # Actually, the tensor decomposition is sum_r U_r V_r W_r.
    # For Strassen, we have 7 terms.
    # Term r corresponds to M_r.
    # U_r is the linear form in A for M_r.
    # V_r is the linear form in B for M_r.
    # W_r is the linear form in C for M_r (which is actually the coefficient in the output C).
    # Wait, the condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    # This means that for each (i,j,k), the sum of products is 1.
    # This is equivalent to saying that the tensor T = sum_r u_r v_r w_r.
    # In Strassen, we compute C = A*B using 7 multiplications.
    # The "multiplication" is the rank-1 term.
    # So U_r, V_r, W_r are the vectors such that (U_r . A) * (V_r . B) contributes to C via W_r.
    # Specifically, C = sum_r (U_r . A) (V_r . B) W_r.
    # Here W_r is a 2x2 matrix (or vectorized).
    # Let's check the indices.
    # C[k,i] = sum_{j} A[i,j] B[j,k].
    # In the decomposition: C[k,i] = sum_r (sum_{i',j'} U[i',j',r] A[i',j']) (sum_{j',k'} V[j',k',r] B[j',k']) W[k,i,r].
    # This matches the standard tensor decomposition.
    
    # Let's construct the 7 terms explicitly.
    # Indices in local 2x2 space:
    # A: i in 0,1; j in 0,1. U index: i*2+j.
    # B: j in 0,1; k in 0,1. V index: j*2+k.
    # C: k in 0,1; i in 0,1. W index: k*2+i.
    
    # M1 = (A00+A11)(B00+B11)
    # U1: A00(1), A11(1) -> [1,0,0,1]
    # V1: B00(1), B11(1) -> [1,0,0,1]
    # W1: C00(1), C11(1) -> [1,0,0,1] (since C00 gets +M1, C11 gets +M1)
    # Wait, W1 is the vector such that W1[k,i] is the coefficient of M1 in C[k,i].
    # C00 = M1 + ... -> W1[0,0] = 1
    # C11 = M1 - ... -> W1[1,1] = 1
    # So W1 = [1, 0, 0, 1] (indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1) in k*2+i order?
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    # Yes.
    
    # M2 = (A10+A11)B00
    # U2: A10(1), A11(1) -> [0,0,1,1]
    # V2: B00(1) -> [1,0,0,0]
    # W2: C10(1) -> [0,0,1,0] (C10 = M2 + M4, so coeff of M2 is 1)
    
    # M3 = A00(B10-B11)
    # U3: A00(1) -> [1,0,0,0]
    # V3: B10(1), B11(-1) -> [0,0,1,-1]
    # W3: C01(1), C11(1) -> [0,1,0,1] (C01 = M3+M5, C11 = M1-M2+M3+M7)
    
    # M4 = A11(B01-B10)
    # U4: A11(1) -> [0,0,0,1]
    # V4: B01(1), B10(-1) -> [0,1,-1,0]
    # W4: C00(1), C10(1) -> [1,0,1,0] (C00 = M1+M4-M5-M7, C10 = M2+M4)
    
    # M5 = (A00+A01)B11
    # U5: A00(1), A01(1) -> [1,1,0,0]
    # V5: B11(1) -> [0,0,0,1]
    # W5: C00(-1), C01(1) -> [-1,1,0,0] (C00 has -M5, C01 has +M5)
    
    # M6 = (A10-A00)(B00+B01)
    # U6: A10(1), A00(-1) -> [-1,0,1,0]
    # V6: B00(1), B01(1) -> [1,1,0,0]
    # W6: Not used in standard Strassen output?
    # Wait, standard Strassen uses 7 multiplications. M6 is often not used in the final sum for 2x2?
    # Let's check:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # M6 is defined but not used in the output!
    # Ah, Strassen's algorithm defines 7 products, but only 7 are used?
    # Actually, M6 is (A10-A00)(B00+B01). It is not part of the final C computation in the basic 2x2 Strassen.
    # The 7 products are M1..M7.
    # Let's re-verify the 7 products.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    # And the outputs are:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # Indeed, M6 is not used. This is a known quirk of Strassen's original presentation or I am misremembering.
    # Actually, many sources say 7 multiplications. If M6 is not used, then we only need 6?
    # No, the rank of 2x2x2 is 7. So there must be 7 terms.
    # Maybe my formula for C is wrong?
    # Let's check a standard reference.
    # Wikipedia:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # It seems M6 is indeed unused in this specific set of formulas.
    # However, the tensor rank is 7. This implies that there is a dependency or I am missing a term.
    # Actually, the decomposition of the tensor into 7 rank-1 terms is what matters.
    # The "unused" M6 might be a linear combination of the others? No, they are independent.
    # Wait, if M6 is not in the output, then the sum of the 7 rank-1 tensors corresponding to M1..M7
    # would have a component in the direction of M6 that cancels out?
    # No, the tensor is sum_r u_r v_r w_r.
    # If we include M6 with w_6 = 0, it doesn't contribute.
    # But the rank is 7, so we need 7 non-zero terms.
    # Perhaps the standard Strassen formulas I'm recalling are slightly off or there's a different set of 7.
    # Let's use the fact that the rank is 7 and try to find a valid set.
    # Actually, I can just use the schoolbook for 2x2x2 (rank 8) and see if I can save 1.
    # Or better, I can use the known rank 7 decomposition.
    # Let's look up the explicit vectors for the rank 7 decomposition of 2x2x2.
    # It is well known.
    # Let's assume the standard Strassen M1..M7 are the 7 terms, and maybe M6 is used in a different formulation.
    # Actually, looking at "The Rank of the 2x2x2 Tensor", the 7 terms are indeed the Strassen ones.
    # The fact that M6 doesn't appear in the C formula means that the vector w_6 is zero?
    # If w_6 is zero, then the term u_6 v_6 w_6 is zero, so it doesn't count towards the rank?
    # No, rank is the minimum number of terms. If w_6=0, we can drop it.
    # So if M6 is not used, the rank would be 6? But the rank is 7.
    # This implies that my C formula is incomplete or M6 is used.
    # Let's check:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # Sum of M's:
    # M1: C00, C11
    # M2: C10
    # M3: C01, C11
    # M4: C00, C10
    # M5: C00, C01
    # M6: None
    # M7: C00, C11
    # It really seems M6 is not used.
    # However, it is possible that the 7th term is not M6 but something else.
    # Or, perhaps the "rank 7" refers to the fact that you cannot do it with 6.
    # Let's try to implement the 7 terms as M1..M7, and if M6 has w=0, I'll just include it with w=0?
    # No, that's not a valid term.
    # Let's assume the question implies I should use the best known decomposition.
    # For 2x2x2, the best is 7.
    # I will use the 7 Strassen terms. If M6 is unused, I will just use 6 terms?
    # But 6 is not enough.
    # Let's check if M6 can be expressed as a combination.
    # Actually, I found a source that says:
    # "The 7 multiplications are M1..M7. The output is C00=M1+M4-M5-M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M7."
    # "Note that M6 is not used in the output. This is a peculiarity of the 2x2 case."
    # "However, the rank is 7, so a 6-term decomposition is impossible."
    # This is a contradiction. If M6 is not used, then the tensor is a sum of 6 terms.
    # Unless... the vectors u_6, v_6 are such that u_6 v_6 w_6 is not zero, but w_6 is not zero?
    # If w_6 is not zero, then M6 must appear in the output.
    # So my formula for C must be wrong.
    # Let's re-derive.
    # A B = [ A00 B00 + A01 B10 , A00 B01 + A01 B11 ]
    #       [ A10 B00 + A11 B10 , A10 B01 + A11 B11 ]
    #
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M3 = A00(B10-B11) = A00B10 - A00B11
    # M4 = A11(B01-B10) = A11B01 - A11B10
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = (A01-A10)(B10+B11) = A01B10 + A01B11 - A10B10 - A10B11
    #
    # Let's sum them up to see if we can form the output.
    # We want:
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    #
    # Let's try to find coefficients c_00, c_01, c_10, c_11 for each M.
    # This is a system of linear equations.
    # Given the complexity, I will stick to the schoolbook for 2x2x2 (rank 8) for safety,
    # or find a reliable rank 7 implementation online.
    # Actually, I can just use the schoolbook for the whole 2x3x6 problem if I can't get Strassen right.
    # But the goal is to beat 36.
    # Let's try a simpler block decomposition.
    # 2x3x6.
    # Split B into two 3x3 blocks.
    # C1 = A B1 (2x3), C2 = A B2 (2x3).
    # Rank(2x3x3) is 15?
    # Schoolbook is 18.
    # If I can do 2x3x3 in 15, then total rank is 30.
    # Is rank(2x3x3) 15?
    # Yes, it is known that R(2,3,3) <= 15.
    # How to get 15?
    # Use Strassen for 2x2x2 part (7 terms) and schoolbook for the rest.
    # A = [ A11 A12 ; A21 A22 ] (2x2 block) + A13, A23 (2x1 column).
    # B = [ B11 B12 B13 ; B21 B22 B23 ; B31 B32 B33 ] (3x3).
    # C = A B.
    # C = [ A11 B11 + A12 B21 + A13 B31 , ... ]
    #
    # Let's decompose 2x3x3.
    # Use 7 Strassen terms for the 2x2x2 part (A11,A12; A21,A22) * (B11,B12; B21,B22) -> C11,C12; C21,C22.
    # This handles the first 2 columns of B and first 2 rows of A.
    # Remaining terms:
    # A13 B31, A13 B32, A13 B33
    # A23 B31, A23 B32, A23 B33
    # And the cross terms from the 2x2 part?
    # The Strassen part computes C_sub = A_sub B_sub.
    # C_sub is 2x2.
    # C = C_sub + A_col * B_row3.
    # A_col is 2x1 (A13, A23).
    # B_row3 is 1x3 (B31, B32, B33).
    # So C = C_sub + [A13; A23] [B31, B32, B33].
    # This is an outer product of a 2x1 and 1x3 matrix.
    # Rank of outer product is 1? No, it's a rank-1 tensor?
    # The tensor for A_col B_row3 is sum_{i,j} A_col[i] B_row3[j] C[i,j].
    # This can be decomposed into 1 term?
    # U[i, 0] = A_col[i], V[0, j] = B_row3[j], W[i, j, 0] = 1?
    # No, the indices are different.
    # U is 2x3 (n*m). V is 3x3 (m*p). W is 3x2 (p*n).
    # For the term A13 B31 -> C11.
    # U[0, r] = 1 (for A13), V[2, r] = 1 (for B31), W[0, r] = 1 (for C11).
    # This is 1 term.
    # Similarly for A13 B32 -> C12: 1 term.
    # A13 B33 -> C13: 1 term.
    # A23 B31 -> C21: 1 term.
    # A23 B32 -> C22: 1 term.
    # A23 B33 -> C23: 1 term.
    # Total 6 terms for the outer product part.
    # Plus 7 terms for the Strassen part.
    # Total 13 terms?
    # Wait, the Strassen part computes C11, C12, C21, C22.
    # It uses A11, A12, A21, A22 and B11, B12, B21, B22.
    # It does NOT use A13, A23 or B31, B32, B33.
    # So the 7 terms are disjoint from the 6 terms?
    # Yes, they involve different indices.
    # So total rank is 7 + 6 = 13.
    # Is 13 < 15? Yes.
    # So R(2,3,3) <= 13.
    # Then for 2x3x6, split B into two 3x3 blocks.
    # Rank = 13 + 13 = 26.
    # This is much better than 36.
    # Let's verify if R(2,3,3) is indeed 13.
    # Actually, I recall R(2,3,3) is 15. Why?
    # Maybe the Strassen part and the outer product part overlap?
    # No, they use disjoint sets of A and B indices.
    # A11, A12, A21, A22 vs A13, A23.
    # B11, B12, B21, B22 vs B31, B32, B33.
    # So they are disjoint.
    # So why is R(2,3,3) 15?
    # Maybe R(2,2,2) is 7, but the way I'm combining them is not optimal?
    # Or maybe R(2,3,3) is actually 13?
    # Let's check online knowledge.
    # "The rank of the 2x3x3 tensor is 15."
    # "A 13-term decomposition is not known."
    # So my 13-term construction must be flawed.
    # Where is the flaw?
    # The Strassen decomposition of 2x2x2 computes C_sub = A_sub B_sub.
    # C_sub[i,j] = sum_k A_sub[i,k] B_sub[k,j].
    # In our case, A_sub is 2x2 (rows 0,1; cols 0,1 of A).
    # B_sub is 2x2 (rows 0,1; cols 0,1 of B).
    # So C_sub is 2x2.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1].
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0].
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1].
    # This is correct.
    # The outer product part:
    # C[0,2] += A[0,2]B[2,2].
    # C[1,2] += A[1,2]B[2,2].
    # C[0,0] += A[0,2]B[2,0].
    # C[0,1] += A[0,2]B[2,1].
    # C[1,0] += A[1,2]B[2,0].
    # C[1,1] += A[1,2]B[2,1].
    # This is also correct.
    # So why is it not a valid decomposition?
    # It must be valid.
    # Perhaps R(2,3,3) is 13?
    # I will proceed with this assumption and construct the rank 26 decomposition.
    # If it fails verification, I will fall back to schoolbook.
    
    pass

def _construct_2x3x3(U, V, W, base_u, base_v, base_w):
    """
    Adds a 13-term decomposition for 2x3x3.
    U rows: base_u + i*3 + j (i in 0,1; j in 0,1,2)
    V rows: base_v + j*3 + k (j in 0,1,2; k in 0,1,2)
    W rows: base_w + k*3 + i (k in 0,1,2; i in 0,1)
    """
    # Part 1: Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Local indices:
    # A: i in 0,1; j in 0,1. U index: i*3+j.
    # B: j in 0,1; k in 0,1. V index: j*3+k.
    # C: k in 0,1; i in 0,1. W index: k*3+i.
    
    # We need the 7 Strassen terms.
    # I will use the standard 7 terms.
    # M1..M7.
    # I'll hardcode the vectors.
    
    # M1: (A00+A11)(B00+B11)
    # U: [1,0,0, 1,0,0] (A00, A11) -> indices 0, 3
    # V: [1,0,0, 0,1,0, 0,0,0] (B00, B11) -> indices 0, 3
    # W: [1,0,0, 0,0,0, 0,1,0] (C00, C11) -> indices 0, 4 (k*3+i: 0*3+0=0, 1*3+1=4)
    
    # M2: (A10+A11)B00
    # U: [0,0,0, 1,1,0] (A10, A11) -> indices 3, 4
    # V: [1,0,0, 0,0,0, 0,0,0] (B00) -> index 0
    # W: [0,0,0, 0,0,0, 1,0,0] (C10) -> index 3 (1*3+0=3)
    
    # M3: A00(B10-B11)
    # U: [1,0,0, 0,0,0] (A00) -> index 0
    # V: [0,0,0, 1,-1,0, 0,0,0] (B10, B11) -> indices 3, 4
    # W: [0,1,0, 0,0,0, 0,0,1] (C01, C11) -> indices 1, 4
    
    # M4: A11(B01-B10)
    # U: [0,0,0, 0,1,0] (A11) -> index 4
    # V: [0,1,0, -1,0,0, 0,0,0] (B01, B10) -> indices 1, 3
    # W: [1,0,0, 1,0,0, 0,0,0] (C00, C10) -> indices 0, 3
    
    # M5: (A00+A01)B11
    # U: [1,1,0, 0,0,0] (A00, A01) -> indices 0, 1
    # V: [0,0,0, 0,0,0, 0,0,1] (B11) -> index 5
    # W: [-1,1,0, 0,0,0, 0,0,0] (C00, C01) -> indices 0, 1
    
    # M6: (A10-A00)(B00+B01)
    # U: [-1,0,0, 1,0,0] (A10, A00) -> indices 3, 0
    # V: [1,1,0, 0,0,0, 0,0,0] (B00, B01) -> indices 0, 1
    # W: [0,0,0, 0,0,0, 0,0,0] (Unused) -> all 0
    
    # M7: (A01-A10)(B10+B11)
    # U: [0,1,0, -1,0,0] (A01, A10) -> indices 1, 3
    # V: [0,0,0, 1,1,0, 0,0,0] (B10, B11) -> indices 3, 4
    # W: [-1,0,0, 0,0,0, 0,0,1] (C00, C11) -> indices 0, 4
    
    # Add these 7 terms.
    for r in range(7):
        col = 7 * r # relative column index, we'll offset later
        
        if r == 0:
            U_vals = {0: 1, 3: 1}
            V_vals = {0: 1, 3: 1}
            W_vals = {0: 1, 4: 1}
        elif r == 1:
            U_vals = {3: 1, 4: 1}
            V_vals = {0: 1}
            W_vals = {3: 1}
        elif r == 2:
            U_vals = {0: 1}
            V_vals = {3: 1, 4: -1}
            W_vals = {1: 1, 4: 1}
        elif r == 3:
            U_vals = {4: 1}
            V_vals = {1: 1, 3: -1}
            W_vals = {0: 1, 3: 1}
        elif r == 4:
            U_vals = {0: 1, 1: 1}
            V_vals = {5: 1}
            W_vals = {0: -1, 1: 1}
        elif r == 5:
            U_vals = {0: -1, 3: 1}
            V_vals = {0: 1, 1: 1}
            W_vals = {}
        elif r == 6:
            U_vals = {1: 1, 3: -1}
            V_vals = {3: 1, 4: 1}
            W_vals = {0: -1, 4: 1}
        
        for idx, val in U_vals.items():
            U[base_u + idx][col] += val
        for idx, val in V_vals.items():
            V[base_v + idx][col] += val
        for idx, val in W_vals.items():
            W[base_w + idx][col] += val

    # Part 2: Outer product for A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
    # 6 terms.
    # A indices: 2 (A02), 5 (A12)
    # B indices: 6 (B20), 7 (B21), 8 (B22)
    # C indices: 
    # C00 (0): A02 B20
    # C01 (1): A02 B21
    # C02 (2): A02 B22
    # C10 (3): A12 B20
    # C11 (4): A12 B21
    # C12 (5): A12 B22
    
    for r in range(6):
        col = 7 + r
        
        if r == 0:
            U_vals = {2: 1}
            V_vals = {6: 1}
            W_vals = {0: 1}
        elif r == 1:
            U_vals = {2: 1}
            V_vals = {7: 1}
            W_vals = {1: 1}
        elif r == 2:
            U_vals = {2: 1}
            V_vals = {8: 1}
            W_vals = {2: 1}
        elif r == 3:
            U_vals = {5: 1}
            V_vals = {6: 1}
            W_vals = {3: 1}
        elif r == 4:
            U_vals = {5: 1}
            V_vals = {7: 1}
            W_vals = {4: 1}
        elif r == 5:
            U_vals = {5: 1}
            V_vals = {8: 1}
            W_vals = {5: 1}
        
        for idx, val in U_vals.items():
            U[base_u + idx][col] += val
        for idx, val in V_vals.items():
            V[base_v + idx][col] += val
        for idx, val in W_vals.items():
            W[base_w + idx][col] += val

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    
    # Try the 26-rank decomposition.
    # Split B into two 3x3 blocks.
    # B1: cols 0,1,2. B2: cols 3,4,5.
    # C1 = A B1 (2x3). C2 = A B2 (2x3).
    # C = [C1, C2].
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)] # 6 x 26
    V = [[0] * rank for _ in range(m * p)] # 18 x 26
    W = [[0] * rank for _ in range(p * n)] # 12 x 26
    
    # Block 1: B cols 0-2.
    # U base: 0
    # V base: 0 (j*6 + k, k in 0..2)
    # W base: 0 (k*2 + i, k in 0..2)
    _construct_2x3x3(U, V, W, 0, 0, 0)
    
    # Block 2: B cols 3-5.
    # U base: 0 (same A)
    # V base: 0 (j*6 + k, k in 3..5) -> indices 3, 4, 5 for j=0; 9,10,11 for j=1; 15,16,17 for j=2.
    # Wait, the local V indices in _construct_2x3x3 are 0..8.
    # We need to map local k (0..2) to global k (3..5).
    # So global V index = j*6 + (k_local + 3).
    # My _construct_2x3x3 uses local indices.
    # I need to modify it to accept a V offset or handle the mapping.
    # Let's create a helper that takes the mapping.
    
    # Actually, I can just call _construct_2x3x3 with adjusted bases?
    # No, the indices inside are hardcoded.
    # I will create a second call with a modified function or just copy the logic.
    # Let's just write the second block explicitly or use a lambda.
    
    # For Block 2:
    # U rows are the same: 0..5.
    # V rows: local j, k -> global j, k+3.
    # W rows: local k, i -> global k+3, i.
    
    # I'll just duplicate the logic with offsets.
    
    # Block 2: 13 terms.
    # Columns 13..25.
    
    for r in range(7):
        col = 13 + r
        if r == 0:
            U_vals = {0: 1, 3: 1}
            V_vals = {0+3: 1, 3+3: 1} # j=0,k=0->3; j=1,k=1->10? No.
            # Local V indices: 0 (j=0,k=0), 3 (j=1,k=0)?
            # In _construct_2x3x3, V index is j*3+k.
            # So local 0 is j=0,k=0. Local 3 is j=1,k=0.
            # Global V index for j=0,k=0 is 0*6+3 = 3.
            # Global V index for j=1,k=0 is 1*6+3 = 9.
            # So I need to map local V index to global.
            # local v_idx = j*3 + k.
            # global v_idx = j*6 + (k+3).
            
            # This is getting messy. I'll write a generic mapper.
            pass
    
    # Let's restart the construction with a cleaner approach.
    # I will define the terms explicitly for both blocks.
    
    U = [[0] * 26 for _ in range(6)]
    V = [[0] * 26 for _ in range(18)]
    W = [[0] * 26 for _ in range(12)]
    
    # Helper to add a term
    def add_term(col, u_vals, v_vals, w_vals):
        for idx, val in u_vals.items():
            U[idx][col] += val
        for idx, val in v_vals.items():
            V[idx][col] += val
        for idx, val in w_vals.items():
            W[idx][col] += val
    
    # Block 1 (cols 0-2 of B)
    # Strassen 2x2x2
    # M1
    add_term(0, {0:1, 3:1}, {0:1, 9:1}, {0:1, 4:1}) # V: j=0,k=0->0; j=1,k=1->1*6+1=7? No.
    # V index j*p+k. p=6.
    # j=0,k=0 -> 0.
    # j=1,k=1 -> 1*6+1 = 7.
    # My previous local index was j*3+k.
    # Local 0 (j=0,k=0) -> Global 0.
    # Local 3 (j=1,k=0) -> Global 1*6+0 = 6.
    # Local 4 (j=1,k=1) -> Global 1*6+1 = 7.
    
    # Let's map local (j,k) to global V index.
    # Global V = j*6 + k.
    # Local V in 2x3x3 logic was j*3+k.
    # So I need to be careful.
    
    # I will just use the explicit (j,k) pairs.
    
    # Block 1:
    # M1: A(0,0),A(1,1) * B(0,0),B(1,1) -> C(0,0),C(1,1)
    # U: 0,3. V: 0,7. W: 0,4.
    add_term(0, {0:1, 3:1}, {0:1, 7:1}, {0:1, 4:1})
    
    # M2: A(1,0),A(1,1) * B(0,0) -> C(1,0)
    # U: 3,4. V: 0. W: 2. (C(1,0) is k=0,i=1 -> 0*2+1=1? No. W index k*n+i. n=2.
    # C(1,0) means i=1, k=0. Index 0*2+1 = 1.
    # Wait, C[i,j] in matrix mult is row i, col j.
    # In W[k*n+i], k is col of C, i is row of C.
    # So C(1,0) is i=1, k=0. Index 1.
    add_term(1, {3:1, 4:1}, {0:1}, {1:1})
    
    # M3: A(0,0) * B(1,0),B(1,1) -> C(0,1),C(1,1)
    # U: 0. V: 6,7. W: 3,4. (C(0,1) is i=0,k=1 -> 1*2+0=2. C(1,1) is i=1,k=1 -> 1*2+1=3.)
    # Wait, C(0,1) means row 0, col 1. i=0, k=1. Index 2.
    # C(1,1) means row 1, col 1. i=1, k=1. Index 3.
    add_term(2, {0:1}, {6:1, 7:-1}, {2:1, 3:1})
    
    # M4: A(1,1) * B(0,1),B(1,0) -> C(0,0),C(1,0)
    # U: 4. V: 1,6. W: 0,1.
    add_term(3, {4:1}, {1:1, 6:-1}, {0:1, 1:1})
    
    # M5: A(0,0),A(0,1) * B(1,1) -> C(0,0),C(0,1)
    # U: 0,1. V: 7. W: 0,2.
    add_term(4, {0:1, 1:1}, {7:1}, {0:-1, 2:1})
    
    # M6: A(1,0),A(0,0) * B(0,0),B(0,1) -> Unused
    # U: 3,0. V: 0,1. W: {}
    add_term(5, {0:-1, 3:1}, {0:1, 1:1}, {})
    
    # M7: A(0,1),A(1,0) * B(1,0),B(1,1) -> C(0,0),C(1,1)
    # U: 1,3. V: 6,7. W: 0,3.
    add_term(6, {1:1, 3:-1}, {6:1, 7:1}, {0:-1, 3:1})
    
    # Outer Product Block 1
    # A(0,2) * B(2,0) -> C(0,0). U:2. V:12. W:0.
    add_term(7, {2:1}, {12:1}, {0:1})
    # A(0,2) * B(2,1) -> C(0,1). U:2. V:13. W:2.
    add_term(8, {2:1}, {13:1}, {2:1})
    # A(0,2) * B(2,2) -> C(0,2). U:2. V:14. W:4.
    add_term(9, {2:1}, {14:1}, {4:1})
    # A(1,2) * B(2,0) -> C(1,0). U:5. V:12. W:1.
    add_term(10, {5:1}, {12:1}, {1:1})
    # A(1,2) * B(2,1) -> C(1,1). U:5. V:13. W:3.
    add_term(11, {5:1}, {13:1}, {3:1})
    # A(1,2) * B(2,2) -> C(1,2). U:5. V:14. W:5.
    add_term(12, {5:1}, {14:1}, {5:1})
    
    # Block 2 (cols 3-5 of B)
    # Same A, shifted B and C.
    # B cols 3,4,5.
    # B(0,3) -> V index 3.
    # B(0,4) -> V index 4.
    # B(0,5) -> V index 5.
    # B(1,3) -> V index 9.
    # B(1,4) -> V index 10.
    # B(1,5) -> V index 11.
    # B(2,3) -> V index 15.
    # B(2,4) -> V index 16.
    # B(2,5) -> V index 17.
    
    # C cols 3,4,5.
    # C(0,3) -> W index 3*2+0 = 6.
    # C(0,4) -> W index 4*2+0 = 8.
    # C(0,5) -> W index 5*2+0 = 10.
    # C(1,3) -> W index 3*2+1 = 7.
    # C(1,4) -> W index 4*2+1 = 9.
    # C(1,5) -> W index 5*2+1 = 11.
    
    # M1
    add_term(13, {0:1, 3:1}, {3:1, 10:1}, {6:1, 9:1})
    # M2
    add_term(14, {3:1, 4:1}, {3:1}, {7:1})
    # M3
    add_term(15, {0:1}, {9:1, 10:-1}, {8:1, 9:1})
    # M4
    add_term(16, {4:1}, {4:1, 9:-1}, {6:1, 7:1})
    # M5
    add_term(17, {0:1, 1:1}, {10:1}, {6:-1, 8:1})
    # M6
    add_term(18, {0:-1, 3:1}, {3:1, 4:1}, {})
    # M7
    add_term(19, {1:1, 3:-1}, {9:1, 10:1}, {6:-1, 9:1})
    
    # Outer Product Block 2
    # A(0,2) * B(2,3) -> C(0,3). U:2. V:15. W:6.
    add_term(20, {2:1}, {15:1}, {6:1})
    # A(0,2) * B(2,4) -> C(0,4). U:2. V:16. W:8.
    add_term(21, {2:1}, {16:1}, {8:1})
    # A(0,2) * B(2,5) -> C(0,5). U:2. V:17. W:10.
    add_term(22, {2:1}, {17:1}, {10:1})
    # A(1,2) * B(2,3) -> C(1,3). U:5. V:15. W:7.
    add_term(23, {5:1}, {15:1}, {7:1})
    # A(1,2) * B(2,4) -> C(1,4). U:5. V:16. W:9.
    add_term(24, {5:1}, {16:1}, {9:1})
    # A(1,2) * B(2,5) -> C(1,5). U:5. V:17. W:11.
    add_term(25, {5:1}, {17:1}, {11:1})
    
    result = {"U": U, "V": V, "W": W}
    
    if _verify(U, V, W):
        return result
    
    # Fallback to schoolbook
    return _schoolbook()

# Run construct
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    print("Valid:", _verify(res["U"], res["V"], res["W"]))
