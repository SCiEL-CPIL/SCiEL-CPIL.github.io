"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 tensor decomposition for 2x4 by 4x6 matrix multiplication
    using Strassen-like block decomposition.
    
    The decomposition uses 2x2 Strassen blocks on the 4x4 middle dimension,
    with the 2x2 and 2x6 outer dimensions handled directly.
    """
    n, m, p = 2, 4, 6
    
    # We decompose the 4x4 middle into 2x2 blocks.
    # Matrix A is 2x4 = [A1 A2] where A1, A2 are 2x2
    # Matrix B is 4x6 = [B1; B2] where B1, B2 are 2x6
    # Product C = A*B = A1*B1 + A2*B2 (2x6)
    
    # Using Strassen on the 2x2x2x6 blocks:
    # We have two 2x2 matrices (A1, A2) and two 2x6 matrices (B1, B2)
    # But Strassen works on 2x2 x 2x2, so we need to handle the 6 differently.
    
    # Alternative approach: Use the known rank-42 decomposition for 2x4x6.
    # This is based on viewing the problem as a composition.
    
    # Let me use a direct construction based on rank reduction.
    # The schoolbook rank is 48. We can reduce by 6 using Strassen on the 4x4 part.
    
    # Strategy: Group the 4 middle indices into 2 groups of 2.
    # For each pair of middle indices (j, j+2) for j=0,1:
    #   We have A[:, j:j+2] (2x2) and B[j:j+2, :] (2x6)
    #   Their product is A[:, j:j+2] @ B[j:j+2, :] (2x6)
    #   Instead of 4 multiplications, use 7 Strassen-like multiplications
    #   But 2x2 x 2x6 can be done in 7 multiplications by embedding into 2x2 x 2x2 blocks.
    
    # Actually, let me use a simpler approach: direct rank-42 construction.
    
    # The tensor T_{i,j,k} = delta_{i,j,k} for matrix mult (i in 0..1, j in 0..3, k in 0..5)
    # We need U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k}
    
    # Rank 42 construction:
    # Use the fact that 4x4x6 can be decomposed with rank 24 (vs schoolbook 24... wait)
    
    # Let me just build the Strassen-based decomposition explicitly.
    
    # View A as 2x2 blocks: A = [[A00, A01], [A10, A11]] each 2x2
    # Wait, A is 2x4, so A = [A_left | A_right] where A_left, A_right are 2x2
    # B is 4x6, so B = [B_top; B_bottom] where B_top, B_bottom are 2x6
    # C = A_left @ B_top + A_right @ B_bottom (2x6)
    
    # For 2x2 @ 2x6: We can use 7 multiplications per "Strassen" style
    # by writing 2x6 as two 2x3 blocks.
    # 2x2 @ 2x3: use 7 multiplications? No, standard is 6 for 2x2x2, 
    # but 2x2x3 needs at least... let's use 8 multiplications (schoolbook is 6 for 2x2x2, 
    # for 2x2x3 it's 2*2*3=12 schoolbook, but we can do better).
    
    # Actually, the cleanest approach: 
    # 2x2 @ 2x2 = 7 (Strassen)
    # 2x2 @ 2x6 = 2x2 @ [2x2 | 2x2 | 2x2] = 3 * 7 = 21? No that's overkill.
    
    # Let me use a different grouping. 
    # The key insight: rank(2,4,6) <= rank(2,2,6) + rank(2,2,6) = 2 * rank(2,2,6)
    # rank(2,2,6) <= 2*rank(2,2,3) and rank(2,2,3) <= 7 (Strassen-like for rectangular)
    # Actually rank(2,2,3) = 7? Let me verify.
    
    # For 2x2 @ 2x3, schoolbook is 12. 
    # We can do: 2x2 @ 2x3 = [2x2 @ 2x1 | 2x2 @ 2x1 | 2x2 @ 2x1] = 3 * rank(2,2,1)
    # rank(2,2,1) = 4 (since it's just linear, you need 4 terms)
    # So that gives 12. Not helpful.
    
    # Better: Use Strassen on 2x2x2 blocks within the 2x2x6.
    # 2x2 @ 2x6: write B as [B1 B2] where B1, B2 are 2x3.
    # A @ B1 + A @ B2. Each A @ Bi is 2x3.
    # 2x2 @ 2x3: We can use the fact that this is a "slice" of the 2x2x2 problem.
    # Actually, the border rank of 2x2x2 is 7, and 2x2x3 has border rank 8 or 9.
    
    # Let me just use the known result: rank(2,4,6) = 42.
    # Construction: Split the 4 middle indices into two groups of 2.
    # For each group, we have a 2x2 @ 2x6 subproblem.
    # rank(2,2,6) = 21 (this is a known result: 2*7 + 7 = 21? No...)
    # Actually rank(2,2,6) <= 2*rank(2,2,3) and if rank(2,2,3)=7 then 14.
    # Hmm, I'm not sure of the exact values.
    
    # Let me use a computational approach to find a rank-42 decomposition.
    # I'll construct it using the Strassen recursion properly.
    
    # 2x4 @ 4x6:
    # Split A into two 2x2 blocks: A = [A1 A2]
    # Split B into two 2x6 blocks: B = [B1; B2]
    # C = A1*B1 + A2*B2
    #
    # Now, 2x2 @ 2x6:
    # Split the 6 columns into three 2-column blocks: B = [Bc1 Bc2 Bc3] each 2x2
    # 2x2 @ 2x2 = 7 multiplications (Strassen)
    # So 2x2 @ 2x6 = 3 * 7 = 21 multiplications
    # Total: 2 * 21 = 42
    
    # This gives rank 42! Let me construct it.
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen formulas for 2x2 @ 2x2:
    # Given M = [[m00, m01], [m10, m11]] and N = [[n00, n01], [n10, n11]]
    # P1 = (m00 + m11) * (n00 + n11)
    # P2 = (m10 + m11) * n00
    # P3 = m00 * (n01 - n11)
    # P4 = m11 * (n10 - n00)
    # P5 = (m00 + m01) * n11
    # P6 = (m10 - m00) * (n00 + n10)
    # P7 = (m01 - m11) * (n01 + n11)
    # Result:
    # r00 = P1 + P4 - P5 + P7
    # r01 = P3 + P5
    # r10 = P2 + P4
    # r11 = P1 - P2 + P3 + P6
    
    # For 2x2 @ 2x6, we have M (2x2) and N1, N2, N3 (each 2x2)
    # R = M*N1 + M*N2 + M*N3
    # We compute M*Ni for each i using Strassen, giving 7 multiplications per Ni.
    # The multiplications are of the form (linear combo of M) * (linear combo of N).
    # Each such product contributes to U, V, W.
    
    # Let me enumerate the 42 rank-1 tensors.
    
    col = 0
    for block in range(2):  # Two 2x2 blocks of A, corresponding to A1 and A2
        for subblock in range(3):  # Three 2x2 sub-blocks of B (each 2x6 split into three 2x2)
            # This gives a 2x2 @ 2x2 Strassen decomposition
            # M = A[:, block*2:(block+1)*2] (2x2)
            # N = B[block*2:(block+1)*2, subblock*2:(subblock+1)*2] (2x2)
            # Wait, I need to be more careful.
            
            # Actually, the middle dimension is 4, split into two 2s.
            # A is 2x4: columns 0,1 are "block 0", columns 2,3 are "block 1"
            # B is 4x6: rows 0,1 are "block 0", rows 2,3 are "block 1"
            # So A1 = A[:, 0:2], A2 = A[:, 2:4]
            # B1 = B[0:2, :], B2 = B[2:4, :]
            # C = A1 @ B1 + A2 @ B2
            
            # For A1 @ B1: A1 is 2x2, B1 is 2x6
            # Split B1 into three 2x2: B1 = [B1a B1b B1c] where B1a = B1[:, 0:2], etc.
            # A1 @ B1 = A1 @ B1a + A1 @ B1b + A1 @ B1c
            
            # Similarly for A2 @ B2.
            
            # So we have 6 subproblems of type 2x2 @ 2x2.
            # Each uses 7 Strassen multiplications.
            # Total: 6 * 7 = 42.
            
            # The 6 subproblems:
            # (A1, B1a), (A1, B1b), (A1, B1c), (A2, B2a), (A2, B2b), (A2, B2c)
            
            # A1 corresponds to A columns 0,1 -> middle indices 0,1
            # A2 corresponds to A columns 2,3 -> middle indices 2,3
            # B1a corresponds to B rows 0,1, cols 0,1 -> middle indices 0,1, output cols 0,1
            # etc.
            
            # Let me define the mapping:
            # For a subproblem with A-block starting at middle col 'am' and B-block starting at middle row 'bm'
            # and output col block 'oc':
            # A submatrix rows: 0,1 (output rows), cols: am, am+1 (middle)
            # B submatrix rows: bm, bm+1 (middle), cols: oc*2, oc*2+1 (output cols)
            
            # Strassen formulas for this subproblem:
            # The 7 multiplications produce linear combinations.
            # For each multiplication, we have:
            # U factor: linear combo of A entries
            # V factor: linear combo of B entries  
            # W factor: contributes to C entries
            
            # Let me write out the 7 Strassen multiplications for a generic 2x2 @ 2x2:
            # P1 = (M00 + M11) * (N00 + N11)
            # P2 = (M10 + M11) * N00
            # P3 = M00 * (N01 - N11)
            # P4 = M11 * (N10 - N00)
            # P5 = (M00 + M01) * N11
            # P6 = (M10 - M00) * (N00 + N10)
            # P7 = (M01 - M11) * (N01 + N11)
            
            # Result:
            # R00 = P1 + P4 - P5 + P7
            # R01 = P3 + P5
            # R10 = P2 + P4
            # R11 = P1 - P2 + P3 + P6
            
            # For our subproblem, M entries are A[?, am], A[?, am+1]
            # N entries are B[bm, ?], B[bm+1, ?]
            
            # Let me define the 7 multiplications and their contributions.
            
            # For multiplication q (0..6):
            # u_coeffs: dict from (row, mid_col) to coefficient for A
            # v_coeffs: dict from (mid_row, col) to coefficient for B
            # w_contrib: dict from (out_row, out_col) to coefficient
            
            # P1: u = {A00: 1, A11: 1}, v = {N00: 1, N11: 1}
            #   w = {R00: 1, R11: 1}
            # P2: u = {A10: 1, A11: 1}, v = {N00: 1}
            #   w = {R10: 1, R11: -1}
            # P3: u = {A00: 1}, v = {N01: 1, N11: -1}
            #   w = {R01: 1, R11: 1}
            # P4: u = {A11: 1}, v = {N10: 1, N00: -1}
            #   w = {R00: 1, R10: 1}
            # P5: u = {A00: 1, A01: 1}, v = {N11: 1}
            #   w = {R00: -1, R01: 1}
            # P6: u = {A10: 1, A00: -1}, v = {N00: 1, N10: 1}
            #   w = {R11: 1}
            # P7: u = {A01: 1, A11: -1}, v = {N01: 1, N11: 1}
            #   w = {R00: 1, R11: 1}
            
            # Wait, I need to double-check the Strassen formulas. Let me re-derive.
            # Standard Strassen:
            # M = [[a, b], [c, d]], N = [[e, f], [g, h]]
            # P1 = (a+d)(e+h)
            # P2 = (c+d)e
            # P3 = a(f-h)
            # P4 = d(g-e)
            # P5 = (a+b)h
            # P6 = (c-a)(e+g)
            # P7 = (b-d)(f+h)
            # R = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
            
            # So mapping:
            # a=M00, b=M01, c=M10, d=M11
            # e=N00, f=N01, g=N10, h=N11
            
            # P1: u={a:1, d:1}, v={e:1, h:1}, w={R00:1, R11:1}
            # P2: u={c:1, d:1}, v={e:1}, w={R10:1, R11:-1}
            # P3: u={a:1}, v={f:1, h:-1}, w={R01:1, R11:1}
            # P4: u={d:1}, v={g:1, e:-1}, w={R00:1, R10:1}
            # P5: u={a:1, b:1}, v={h:1}, w={R00:-1, R01:1}
            # P6: u={c:1, a:-1}, v={e:1, g:1}, w={R11:1}
            # P7: u={b:1, d:-1}, v={f:1, h:1}, w={R00:1, R11:1}
            
            strassen_defs = [
                # (u_coeffs, v_coeffs, w_coeffs)
                # u: {(0,0): 1, (1,1): 1} means A[0, mid0] * 1 + A[1, mid1] * 1
                # v: {(0,0): 1, (1,1): 1} means B[mid0, col0] * 1 + B[mid1, col1] * 1
                # w: {(0,0): 1, (1,1): 1} means C[0, col0] += 1 * product, C[1, col1] += 1 * product
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # P1
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),  # P2
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),  # P3
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),  # P4
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),  # P5
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),  # P6
                ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # P7
            ]
            
            # Now iterate over the 6 subproblems
            for sub_idx, (a_mid, b_mid, out_col_block) in enumerate(
                [(0, 0, 0), (0, 0, 1), (0, 0, 2),
                 (1, 1, 0), (1, 1, 1), (1, 1, 2)]
            ):
                # a_mid: which 2-block of A (0 or 1), corresponds to middle cols 2*a_mid, 2*a_mid+1
                # b_mid: which 2-block of B (0 or 1), corresponds to middle rows 2*b_mid, 2*b_mid+1
                # out_col_block: which 2-block of output cols, corresponds to cols 2*out_col_block, 2*out_col_block+1
                
                for q, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_defs):
                    for (u_r, u_c), u_val in u_coeffs.items():
                        # u_r is local row (0 or 1), u_c is local col (0 or 1)
                        # Global A row: u_r, Global A col: 2*a_mid + u_c
                        global_a_row = u_r
                        global_a_col = 2 * a_mid + u_c
                        U[global_a_row * m + global_a_col, col] += u_val
                    
                    for (v_r, v_c), v_val in v_coeffs.items():
                        # v_r is local row (0 or 1), v_c is local col (0 or 1)
                        # Global B row: 2*b_mid + v_r, Global B col: 2*out_col_block + v_c
                        global_b_row = 2 * b_mid + v_r
                        global_b_col = 2 * out_col_block + v_c
                        V[global_b_row * p + global_b_col, col] += v_val
                    
                    for (w_r, w_c), w_val in w_coeffs.items():
                        # w_r is local row (0 or 1), w_c is local col (0 or 1)
                        # Global C row: w_r, Global C col: 2*out_col_block + w_c
                        global_c_row = w_r
                        global_c_col = 2 * out_col_block + w_c
                        # W index: k*n + i = global_c_col * n + global_c_row
                        W[global_c_col * n + global_c_row, col] += w_val
                    
                    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
