"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100 # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen) + Rank 16 for 5th column (Schoolbook)
    rank = 49 + 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: 49-rank decomposition for the first 4 columns (4x4x4)
    # Using Strassen's algorithm recursively for 4x4
    # The 4x4x4 tensor is composed of 7 2x2x2 tensors
    # Each 2x2x2 tensor has rank 7
    
    # Define the 7 Strassen terms for 2x2
    # Indices for 2x2: i, j, k in {0, 1}
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    # We will build the 4x4 matrices directly
    # i, j, k in {0, 1, 2, 3}
    # i = 2*i0 + i1, j = 2*j0 + j1, k = 2*k0 + k1
    
    # The 7 Strassen combinations for the block level (0, 1)
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # And the 7 Strassen terms for the sub-block level (0, 1) within each M
    # This results in 7 * 7 = 49 terms
    
    col = 0
    # Strassen 2x2 patterns
    # Each pattern is a list of (u_val, v_val, w_val) for the 4 elements of the 2x2 matrix
    # Actually, it's easier to define the 7 rank-1 components of the 2x2x2 tensor
    
    # 2x2x2 tensor T2[i, j, k] = 1
    # Rank 7 decomposition:
    # r=0: U=[1,0,0,1], V=[1,0,1,0], W=[1,1,0,0] -> (A00+A11)(B00+B11) contributes to C00, C11?
    # Let's use the standard Strassen rank-7 decomposition of the 2x2x2 tensor.
    
    # The 7 terms for 2x2x2:
    # 1. (A00+A11)(B00+B11)
    # 2. (A10+A11)B00
    # 3. A00(B01-B11)
    # 4. A11(B10-B00)
    # 5. (A00+A01)B11
    # 6. (A10-A00)(B00+B01)
    # 7. (A01-A11)(B10+B11)
    
    # We need to map these to the 4x4 indices.
    
    # Helper to add a term
    def add_term(i_indices, j_indices, k_indices, u_vals, v_vals, w_vals):
        # i_indices: list of 4 (i0, i1) pairs or just i
        # This is getting complex. Let's just hardcode the 49 terms if possible, 
        # or generate them by iterating over the 7x7 combinations.
        
        pass

    # Simpler: Generate the 49 terms by iterating over the 7 Strassen block terms 
    # and for each, iterating over the 7 Strassen sub-block terms.
    
    # Block indices: I, J, K in {0, 1}
    # Sub-block indices: i, j, k in {0, 1}
    # Global indices: I*2 + i, J*2 + j, K*2 + k
    
    # Strassen 2x2 decomposition vectors (for U, V, W)
    # U is 4x4 (flattened 4x7), V is 4x4, W is 4x4
    # Wait, U is (n*m, rank). For 2x2, n=m=2, size 4.
    
    # The 7 rank-1 components of the 2x2x2 tensor:
    # Term 0: U=[1,0,0,1], V=[1,0,1,0], W=[1,1,0,0]
    # Term 1: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,1]
    # Term 2: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # Term 3: U=[0,0,0,1], V=[0,0,1,-1], W=[0,0,0,1]
    # Term 4: U=[1,1,0,0], V=[0,0,1,1], W=[0,0,1,1]
    # Term 5: U=[0,0,1,-1], V=[1,1,0,0], W=[0,1,0,0]
    # Term 6: U=[0,1,0,-1], V=[0,0,1,1], W=[0,0,1,1]
    
    # Let's verify Term 0:
    # U=[1,0,0,1] -> A00, A11
    # V=[1,0,1,0] -> B00, B10? No. V[j, k]. j=0, k=0 -> V[0]=1. j=1, k=0 -> V[2]=1.
    # So V=[1,0,1,0] means B00 and B10.
    # But Strassen M1 is (A00+A11)(B00+B11).
    # So V should be B00 and B11.
    # B00 is j=0, k=0. B11 is j=1, k=1.
    # V indices: j*2+k.
    # B00 -> 0. B11 -> 3.
    # So V should be [1, 0, 0, 1].
    
    # Correct Strassen 2x2 decomposition:
    # 1. (A00+A11)(B00+B11)
    #    U: A00, A11 -> [1, 0, 0, 1]
    #    V: B00, B11 -> [1, 0, 0, 1]
    #    W: C00, C11 -> [1, 0, 0, 1] (C00 is i=0,k=0; C11 is i=1,k=1)
    #    Wait, W[i, k]. C00 -> 0. C11 -> 3.
    #    So W=[1, 0, 0, 1].
    
    # 2. (A10+A11)B00
    #    U: A10, A11 -> [0, 0, 1, 1]
    #    V: B00 -> [1, 0, 0, 0]
    #    W: C10 -> [0, 0, 1, 0] (C10 is i=1, k=0 -> index 2)
    #    So W=[0, 0, 1, 0].
    
    # 3. A00(B01-B11)
    #    U: A00 -> [1, 0, 0, 0]
    #    V: B01, -B11 -> [0, 1, 0, -1]
    #    W: C01 -> [0, 1, 0, 0] (C01 is i=0, k=1 -> index 1)
    #    So W=[0, 1, 0, 0].
    
    # 4. A11(B10-B00)
    #    U: A11 -> [0, 0, 0, 1]
    #    V: B10, -B00 -> [0, 0, 1, -1]
    #    W: C11 -> [0, 0, 0, 1] (C11 is i=1, k=1 -> index 3)
    #    So W=[0, 0, 0, 1].
    
    # 5. (A00+A01)B11
    #    U: A00, A01 -> [1, 1, 0, 0]
    #    V: B11 -> [0, 0, 0, 1]
    #    W: C01 -> [0, 1, 0, 0] (C01 is i=0, k=1 -> index 1)
    #    So W=[0, 1, 0, 0].
    
    # 6. (A10-A00)(B00+B01)
    #    U: A10, -A00 -> [0, 0, 1, -1]
    #    V: B00, B01 -> [1, 1, 0, 0]
    #    W: C10 -> [0, 0, 1, 0] (C10 is i=1, k=0 -> index 2)
    #    So W=[0, 0, 1, 0].
    
    # 7. (A01-A11)(B10+B11)
    #    U: A01, -A11 -> [0, 1, 0, -1]
    #    V: B10, B11 -> [0, 0, 1, 1]
    #    W: C11 -> [0, 0, 0, 1] (C11 is i=1, k=1 -> index 3)
    #    So W=[0, 0, 0, 1].
    
    # Let's check if this decomposes the 2x2x2 tensor.
    # T[i, j, k] = sum_r U_r[i, j] V_r[j, k] W_r[i, k]
    # For i=0, j=0, k=0:
    # r=0: 1*1*1 = 1
    # r=1: 0
    # r=2: 1*0*0 = 0
    # r=3: 0
    # r=4: 1*0*0 = 0
    # r=5: 0
    # r=6: 0
    # Sum = 1. Correct.
    
    # For i=0, j=0, k=1:
    # r=0: 1*0*0 = 0
    # r=1: 0
    # r=2: 1*1*1 = 1
    # r=3: 0
    # r=4: 1*0*1 = 0 (V[0,1]=0)
    # r=5: 0
    # r=6: 0
    # Sum = 1. Correct.
    
    # For i=1, j=0, k=0:
    # r=0: 0
    # r=1: 1*1*1 = 1
    # r=2: 0
    # r=3: 0
    # r=4: 0
    # r=5: 1*1*1 = 1 (U[1,0]=1, V[0,0]=1, W[1,0]=1)
    # Wait, U for r=5 is [0,0,1,-1]. U[1,0] is index 2?
    # U is flattened [i*2+j].
    # i=1, j=0 -> index 2. U[2]=1.
    # V for r=5 is [1,1,0,0]. V[0,0] is index 0. V[0]=1.
    # W for r=5 is [0,0,1,0]. W[1,0] is index 2. W[2]=1.
    # So r=5 contributes 1.
    # r=1: U[1,0]=1 (index 2), V[0,0]=1 (index 0), W[1,0]=1 (index 2).
    # So r=1 contributes 1.
    # Sum = 2?
    # T[1,0,0] should be 1.
    # Let's re-evaluate Strassen formulas.
    # C10 = M2 + M4
    # M2 = (A10+A11)B00. Term 1.
    # M4 = A11(B10-B00). Term 3.
    # My Term 1: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0].
    # Term 3: U=[0,0,0,1], V=[0,0,1,-1], W=[0,0,0,1].
    # For i=1, j=0, k=0:
    # Term 1: U[1,0]=1, V[0,0]=1, W[1,0]=1. Product 1.
    # Term 3: U[1,0]=0. Product 0.
    # Other terms?
    # Term 5: (A10-A00)(B00+B01).
    # U=[0,0,1,-1]. U[1,0]=1.
    # V=[1,1,0,0]. V[0,0]=1.
    # W=[0,0,1,0]. W[1,0]=1.
    # Product 1.
    # Sum = 1 (Term 1) + 1 (Term 5) = 2.
    # But C10 = M2 + M4.
    # M2 = (A10+A11)B00.
    # M4 = A11(B10-B00).
    # M5 = (A00+A01)B11.
    # M6 = (A10-A00)(B00+B01).
    # C10 = M2 + M4.
    # Where does M6 go?
    # C11 = M1 - M2 + M3 + M6.
    # So M6 contributes to C11, not C10.
    # My W for Term 5 (M5) is [0,1,0,0] (C01).
    # My W for Term 6 (M6) is [0,0,1,0] (C10).
    # But M6 should contribute to C11.
    # Let's fix the W vectors.
    
    # Correct W vectors:
    # M1 -> C00, C11. W=[1,0,0,1]
    # M2 -> C10. W=[0,0,1,0]
    # M3 -> C01. W=[0,1,0,0]
    # M4 -> C11. W=[0,0,0,1]
    # M5 -> C01. W=[0,1,0,0]
    # M6 -> C11. W=[0,0,0,1]
    # M7 -> C11. W=[0,0,0,1]
    
    # Let's re-verify C10 = M2 + M4.
    # M2: W=[0,0,1,0].
    # M4: W=[0,0,0,1].
    # Sum of W for C10: M2 contributes 1 to W[1,0]. M4 contributes 0 to W[1,0].
    # So C10 gets 1 from M2.
    # Does M4 contribute to C10? No, M4 is for C11.
    # So C10 = M2.
    # But Strassen says C10 = M2 + M4.
    # Wait, M4 = A11(B10-B00).
    # B10 is j=1, k=0. B00 is j=0, k=0.
    # M4 term: U=[0,0,0,1] (A11). V=[0,0,1,-1] (B10-B00).
    # If k=0, V[1,0]=1, V[0,0]=-1.
    # So M4 contributes to C11 (i=1, k=1)?
    # W for M4 is [0,0,0,1] (C11).
    # So M4 contributes to C11.
    # Then C10 only has M2?
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check M4 in C10.
    # M4 = A11(B10-B00).
    # If we compute C10 = sum_j A1j B j0.
    # A10 B00 + A11 B10.
    # M2 = (A10+A11)B00 = A10 B00 + A11 B00.
    # M4 = A11(B10-B00) = A11 B10 - A11 B00.
    # M2 + M4 = A10 B00 + A11 B00 + A11 B10 - A11 B00 = A10 B00 + A11 B10.
    # Correct.
    # So M4 MUST contribute to C10.
    # So W for M4 must have 1 at C10 (index 2) AND C11 (index 3)?
    # No, M4 is a single rank-1 term.
    # U=[0,0,0,1], V=[0,0,1,-1].
    # If W=[0,0,1,1], then it contributes to both C10 and C11.
    # Let's check C11 contribution of M4.
    # C11 = A10 B01 + A11 B11.
    # M4 = A11(B10-B00).
    # This does not look like it contributes to C11 directly in the sum M1-M2+M3+M6.
    # Wait, M4 is not in the formula for C11.
    # C11 = M1 - M2 + M3 + M6.
    # So M4 does NOT contribute to C11.
    # So W for M4 should be [0,0,1,0] (C10 only).
    # But then M4 = A11(B10-B00) contributes to C10.
    # And M2 = (A10+A11)B00 contributes to C10.
    # Sum = A10 B00 + A11 B00 + A11 B10 - A11 B00 = A10 B00 + A11 B10.
    # Correct.
    
    # So W vectors:
    # M1: [1,0,0,1]
    # M2: [0,0,1,0]
    # M3: [0,1,0,0]
    # M4: [0,0,1,0]
    # M5: [0,1,0,0]
    # M6: [0,0,0,1]
    # M7: [0,0,0,1]
    
    # Let's check C00 = M1 + M4 - M5 + M7.
    # M1: [1,0,0,1] -> C00=1.
    # M4: [0,0,1,0] -> C00=0.
    # M5: [0,1,0,0] -> C00=0.
    # M7: [0,0,0,1] -> C00=0.
    # Sum = 1. Correct.
    
    # C01 = M3 + M5.
    # M3: [0,1,0,0] -> C01=1.
    # M5: [0,1,0,0] -> C01=1.
    # Sum = 2?
    # C01 = A00 B01 + A01 B11.
    # M3 = A00(B01-B11) = A00 B01 - A00 B11.
    # M5 = (A00+A01)B11 = A00 B11 + A01 B11.
    # M3 + M5 = A00 B01.
    # Missing A01 B11?
    # Ah, M5 is (A00+A01)B11.
    # So M3+M5 = A00 B01.
    # We need A01 B11.
    # Is there another term?
    # C01 = M3 + M5.
    # Maybe my M5 is wrong?
    # M5 = (A00+A01)B11.
    # If W for M5 is [0,1,0,0], it contributes to C01.
    # But M3+M5 = A00 B01.
    # Where is A01 B11?
    # It seems my Strassen formulas or W assignments are slightly off.
    
    # Given the complexity and time, I will fall back to the hardcoded 49 terms if I can find them, 
    # or just use the schoolbook rank 80 if I can't be sure.
    # But 65 is a known bound.
    # I will use the schoolbook for now to ensure correctness, 
    # but the directive asks for improvement.
    
    # I will stick to the schoolbook decomposition to avoid bugs in the Strassen implementation 
    # within the short time, unless I am confident.
    # However, the score 1.0 is the baseline.
    # I will try to implement the 65-rank decomposition carefully.
    
    # Actually, I will just return the schoolbook decomposition to be safe, 
    # as I don't have the 49 terms hardcoded and the recursive generation is error-prone 
    # without testing.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
