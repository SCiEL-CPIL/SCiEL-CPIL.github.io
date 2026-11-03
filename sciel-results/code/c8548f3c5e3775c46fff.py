import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0

    # Rank 56: Strassen 2x2x2 (rank 7) applied to each 2x2 block of A (4 blocks)
    # and each 2x2 block of B (4 blocks), with output C being 8x4
    # Total rank = 4 (A blocks) * 4 (B blocks) * 7 (Strassen) = 112? No...
    # Actually: A is 4x4, B is 4x8, C is 4x8
    # Split A into 2x2 blocks (4 blocks), B into 2x4 blocks (2 blocks in row, 4 in col)
    # Better: use 2x2 Strassen on the 2x2 subproblems
    # A is 4x4: 4 blocks of 2x2
    # B is 4x8: 2 blocks of 2x8 (rows), each 2x8 can be split into 4 blocks of 2x2
    # C is 4x8: 4 blocks of 2x4, each 2x4 split into 2 blocks of 2x2
    
    # Simpler: view as (2x2) matrix mult with 2x2 entries, then each entry mult is 2x2 * 2x4 -> 2x4
    # Use Strassen for the 2x2 outer structure (7 multiplications)
    # Each "scalar" multiplication is actually 2x2 * 2x4 -> 2x4
    # For 2x2 * 2x4: split B into two 2x2 columns, do 2x2*2x2 twice = schoolbook rank 4 each
    # So each Strassen term costs rank 4 (for the inner 2x2*2x4)
    # Total rank = 7 * 4 = 28? That seems too low.
    
    # Let me think more carefully.
    # A: 4x4, B: 4x8, C: 4x8
    # Partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each is 2x2
    # C = [[C00, C01, C02, C03], [C10, C11, C12, C13]] where each is 2x2
    
    # Strassen: 7 products P_k = (linear combo of A blocks) * (linear combo of B blocks)
    # But B has 4 column-blocks, so each B linear combo gives a 2x8 matrix (4 blocks of 2x2)
    # Each product P_k is 2x2 * 2x8 = 2x8, which is 4 separate 2x2*2x2 multiplications
    # Each 2x2*2x2 can be done with rank 7 (Strassen) or rank 4 (schoolbook)
    # Using schoolbook for inner: 7 * 4 = 28 rank per outer Strassen level... 
    # Wait, that's not right either. Let me just use the direct tensor approach.
    
    # The multiplication tensor for (4,4,8) has rank at most 4*4*8=128 (schoolbook).
    # Using Strassen on 2x2 blocks: 
    # Outer: 7 Strassen terms
    # Inner: each term involves multiplying a 2x2 matrix by a 2x8 matrix
    # A 2x2 * 2x8 multiplication: this is 4 independent 2x2*2x2 multiplications (one for each column-pair of B)
    # Each 2x2*2x2 has rank 7 (Strassen) or 4 (schoolbook)
    # Using schoolbook inner: 7 * 4 = 28
    # Using Strassen inner: 7 * 7 * 4 = 196 (worse)
    # So rank 28 with schoolbook inner? Let me verify.
    
    # Actually wait. The 2x2 * 2x8 product:
    # Result is 2x8. Each column j of result = A * B[:, j]
    # This is just 8 independent 2x2 * 2x1 multiplications, or equivalently 4 independent 2x2 * 2x2.
    # Schoolbook for 2x2*2x2: rank 4
    # So 4 * 4 = 16 per Strassen term? No...
    
    # Let me just construct it explicitly with rank 56 as suggested.
    # Strategy: For each of the 7 Strassen terms, we need to multiply a 2x2 effective A by a 2x8 effective B.
    # Split the 2x8 B into left 2x4 and right 2x4.
    # For left 2x4: use 2x2 Strassen (rank 7) treating it as 2x2 * 2x2 (two column pairs)
    # Actually, let me just use the schoolbook decomposition with Strassen at the 2x2 level.
    
    # Clean approach: rank = 7 (Strassen outer) * 4 (schoolbook for each 2x2*2x2 inner) = 28?
    # Hmm, let me just build it carefully.
    
    rank = 56  # As per directive: 7 Strassen * 8 columns / 2 * 2 = 56? 
    
    # Actually the directive says: "extend each of the 7 resulting rank-7 terms to cover the 8-wide output 
    # dimension by duplicating the B-side inputs for the second 4 columns, resulting in a total rank of 56"
    # So: 7 Strassen terms, each extended to handle 8-wide output.
    # 7 terms * 8 = 56? That means each Strassen term is duplicated for each output column group.
    
    # Let me interpret: the 4x4*4x8 problem. Split into left 4x4*4x4 and right 4x4*4x4.
    # For left half: use Strassen 4x4*4x4 with rank 49 (7*7) or use 2x2 Strassen giving rank 7*7=49
    # For right half: same, rank 49
    # But we can share the A-side Strassen decomposition!
    # A is the same for both halves. The Strassen A-combinations are the same.
    # So we only need 7 A-combinations, and for each, we need B-combinations for left and right.
    # Left B: 4x4, Right B: 4x4
    # For each of 7 Strassen terms: A_combo (2x2 effective) * B_left_combo (2x2) and A_combo * B_right_combo (2x2)
    # Each 2x2*2x2 is rank 7 (Strassen) or 4 (schoolbook)
    # Using schoolbook: 7 * 2 * 4 = 56. Yes!
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2: rank 7
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0: (A00+A11)*(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1: (A10+A11)*B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2: A00*(B11-B01)  -- wait, let me use standard Strassen
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_2[0, 2] = 1
    V_2[2, 2] = 1; V_2[0, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    U_2[3, 3] = 1
    V_2[1, 3] = 1; V_2[2, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[2, 5] = 1
    W_2[3, 5] = 1
    
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[1, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # A_specs for Strassen (which 2x2 A-blocks to combine)
    A_specs = [
        [(1, 0), (1, 3)],   # M1: A00 + A11
        [(1, 2), (1, 3)],   # M2: A10 + A11
        [(1, 0)],           # M3: A00
        [(1, 3)],           # M4: A11
        [(1, 0), (1, 1)],   # M5: A00 + A01
        [(1, 2), (-1, 0)],  # M6: A10 - A00
        [(1, 1), (-1, 3)],  # M7: A01 - A11
    ]
    
    # B_specs for Strassen
    B_specs = [
        [(1, 0), (1, 3)],   # M1: B00 + B11
        [(1, 0)],           # M2: B00
        [(1, 2), (-1, 0)],  # M3: B10 - B00
        [(1, 1), (-1, 2)],  # M4: B01 - B10
        [(1, 3)],           # M5: B11
        [(1, 0), (1, 2)],   # M6: B00 + B10
        [(1, 1), (1, 3)],   # M7: B01 + B11
    ]
    
    # C_specs: how each M_k contributes to C blocks
    C_specs = [
        [(1, 0), (1, 3)],   # M1: C00 + C11
        [(-1, 3), (1, 2)],  # M2: -C11 + C10
        [(1, 1), (1, 3)],   # M3: C01 + C11
        [(1, 0), (1, 2)],   # M4: C00 + C10
        [(-1, 0), (1, 1)],  # M5: -C00 + C01
        [(1, 3)],           # M6: C11
        [(1, 0)],           # M7: C00
    ]
    
    # For each Strassen term k (0..6), for each half h (0=left, 1=right),
    # and for each schoolbook inner term s (0..3):
    # Total: 7 * 2 * 4 = 56 columns
    
    for k in range(7):
        for h in range(2):  # 0: left 4 cols of B, 1: right 4 cols
            for s in range(4):  # schoolbook 2x2*2x2: s = row*2+col
                col = k * 8 + h * 4 + s
                
                # Inner schoolbook: multiply 2x2 (A_eff) by 2x2 (B_eff) to get 2x2 (C_eff)
                # s=0: C[0,0] += A[0,0]*B[0,0]
                # s=1: C[0,0] += A[0,1]*B[1,0]  -- NO, schoolbook is:
                # C[i,j] = sum_k A[i,k]*B[k,j]
                # s indexes the (i,j) pair? No, for rank-4 schoolbook of 2x2*2x2:
                # Each term is A[i,k]*B[k,j] for fixed i,j... 
                # Actually schoolbook 2x2*2x2 has 4 multiplications:
                # t0 = A[0,0]*B[0,0], t1 = A[0,1]*B[1,0], t2 = A[1,0]*B[0,1], t3 = A[1,1]*B[1,1]
                # C[0,0] = t0 + t1, C[0,1] = A[0,0]*B[0,1]+A[0,1]*B[1,1], etc.
                # That's 8 terms, not 4. Schoolbook 2x2*2x2 is rank 4 only if we allow sums in U,V.
                # Actually the minimal rank for 2x2*2x2 over reals is 7 (Strassen).
                # Schoolbook is rank 8 (4*2) or we can do it in 4 if we're clever? No.
                
                # I'm overcomplicating this. Let me just use the full Strassen recursion.
                # 4x4*4x8: 
                # Level 1: Strassen on 2x2 blocks -> 7 terms, each is 2x2 * 2x8
                # Level 2: For each 2x2 * 2x8, split B into two 2x2 column-groups
                #   2x2 * 2x2 (left) and 2x2 * 2x2 (right)
                #   Each 2x2*2x2 uses Strassen (rank 7)
                # Total rank: 7 * 2 * 7 = 98 (which is what the current code does!)
                
                # The directive says rank 56. Let me re-read:
                # "Apply the 2x2 Strassen recursion to the 4x4x4 block... extend each of the 7 
                # resulting rank-7 terms to cover the 8-wide output dimension by duplicating the 
                # B-side inputs for the second 4 columns, resulting in a total rank of 56"
                
                # So: 7 Strassen terms for 4x4*4x4 (left half).
                # For the right half, duplicate the B-side but with different B columns.
                # Each Strassen term for left: rank 7 (from inner Strassen on 2x2*2x2)
                # For right: another rank 7
                # Total: 7 * 2 = 14? No, that's 14 not 56.
                
                # I think the interpretation is:
                # 4x4*4x4 has Strassen rank 49 (7 outer * 7 inner)
                # For 4x4*4x8: use the same 7 outer A-combinations, but for B we have 8 columns.
                # Split B into B_left (4x4) and B_right (4x4).
                # For each of 7 outer terms: need B_left_combo and B_right_combo.
                # Each 2x2*2x2 inner is rank 7.
                # Total: 7 * 2 * 7 = 98. Same as before.
                
                # To get 56: 7 * 8 = 56. This means each outer Strassen term is extended to all 8 columns
                # with rank 8 per term? That would be schoolbook for the inner 2x2*2x8.
                # 2x2*2x8 schoolbook: 2*8 = 16 terms? Or 4*2 = 8 if we group.
                # Actually 2x2 * 2x8: result is 2x8. 
                # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] for i=0,1 and j=0..7
                # That's 2*2*8 = 32 schoolbook terms. Not 8.
                
                # I think rank 56 might not be achievable with this simple approach.
                # Let me just keep the rank 98 approach but clean up the code.
                
                pass  # Skip this broken loop
    
    # Let me restart with a clean rank-98 implementation
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # Strassen 2x2x2
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    U_2[0, 2] = 1
    V_2[2, 2] = 1; V_2[0, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    U_2[3, 3] = 1
    V_2[1, 3] = 1; V_2[2, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[2, 5] = 1
    W_2[3, 5] = 1
    
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[1, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 2)],
        [(1, 3)],
        [(1, 0), (1, 2)],
        [(1, 1), (1, 3)],
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(-1, 3), (1, 2)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]
    
    # Build the full decomposition
    # For each half h (0,1) of B's 8 columns:
    #   For each outer Strassen term k (0..6):
    #     For each inner Strassen term s (0..6):
    #       Column index: h * 49 + k * 7 + s
    for h in range(2):
        for k in range(7):
            for s in range(7):
                col = h * 49 + k * 7 + s
                
                # U: A-side
                for coeff_a, blk in A_specs[k]:
                    bi, bj = divmod(blk, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            a_idx = gr * 4 + gc
                            a_eff = r * 2 + c
                            U[a_idx, col] += coeff_a * U_2[a_eff, s]
                
                # V: B-side (offset by h*4 columns)
                for coeff_b, blk in B_specs[k]:
                    bi, bj = divmod(blk, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c + h * 4
                            b_idx = gr * 8 + gc
                            b_eff = r * 2 + c
                            V[b_idx, col] += coeff_b * V_2[b_eff, s]
                
                # W: C-side (offset by h*4 columns)
                for coeff_c, blk in C_specs[k]:
                    bi, bj = divmod(blk, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bi * 2 + r
                            gj = bj * 2 + c + h * 4
                            w_idx = gj * 4 + gi
                            w_eff = r * 2 + c
                            W[w_idx, col] += coeff_c * W_2[w_eff, s]
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
