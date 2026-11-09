"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build a Strassen-based decomposition for 4x4 * 4x7
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # Split B (4x7) into B1 (4x4) and B2 (4x3)
    # Split B1 into 2x2 blocks: B11, B12, B21, B22
    # Split B2 into B2_top (2x3) and B2_bot (2x3)
    #
    # C = A * B = A * B1 + A * B2
    # C1 = A * B1 (4x4 result) computed via Strassen (7 mults of 2x2)
    # C2 = A * B2 (4x3 result) computed via schoolbook (12 mults of 2x2)
    #
    # Total: 7 + 12 = 19 multiplications of 2x2 blocks
    # Each 2x2 mult = 4 scalar mults, so total rank = 19 * 4 = 76
    # vs schoolbook rank = 4*4*7 = 112
    
    # Indexing: A[i,j] where i in 0..3, j in 0..3
    # B[j,k] where j in 0..3, k in 0..6
    # C[i,k] where i in 0..3, k in 0..6
    
    # U has shape (n*m, rank) = (16, rank), indexed by (i*m+j)
    # V has shape (m*p, rank) = (28, rank), indexed by (j*p+k)
    # W has shape (p*n, rank) = (28, rank), indexed by (k*n+i)
    
    # We'll build the decomposition column by column.
    # Each "multiplication" in the decomposition corresponds to one rank-1 term.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        """Add a rank-1 term where U[u_idx] += u_val, V[v_idx] += v_val, W[w_idx] += w_val"""
        U_list.append(u_idx)
        V_list.append(v_idx)
        W_list.append(w_idx)
        U_list.append(u_val)
        V_list.append(v_val)
        W_list.append(w_val)
    
    # Actually, let me think more carefully.
    # The tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix mult: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So each rank-1 term picks out specific (i,j) from A, specific (j,k) from B, and contributes to C[i,k]
    
    # Let me rebuild this properly.
    # Each term r has:
    #   U row: i*m + j (selects A[i,j])
    #   V row: j*p + k (selects B[j,k])
    #   W row: k*n + i (selects C[i,k])
    #   with coefficients u_r, v_r, w_r
    
    # For schoolbook, each term has u=v=w=1 and one specific (i,j,k)
    
    # For Strassen, we combine terms with coefficients.
    
    # Let me define the Strassen terms for the 4x4 x 4x4 part.
    # A blocks: A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B1 blocks: B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    # C1 blocks: C11 = C[0:2,0:2], C12 = C[0:2,2:4], C21 = C[2:4,0:2], C22 = C[2:4,2:4]
    
    # Strassen multiplications (each is a 2x2 matrix product):
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # Results:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product, which in tensor decomposition terms means
    # 4 scalar multiplications with appropriate coefficients.
    
    # Let me enumerate all 19*4 = 76 terms.
    
    terms = []  # Each term: (i, j, k, u_coeff, v_coeff, w_coeff)
    
    # Helper: add a 2x2 block multiplication
    # A_block is identified by (row_offset, col_offset) in the 4x4 A
    # B_block is identified by (row_offset, col_offset) in the 4x4 B1
    # C_block is identified by (row_offset, col_offset) in the 4x4 C
    # u_coeff, v_coeff, w_coeff are the multipliers
    
    def add_block_mult(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, u_c, v_c, w_c):
        """Add terms for multiplying 2x2 block of A by 2x2 block of B, contributing to 2x2 block of C"""
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = a_r0 + di
                    j = a_c0 + dj
                    # B[j, b_c0 + dk]
                    # C[i, c_c0 + dk]
                    k = b_c0 + dk
                    # The term: A[i,j] * B[j,k] contributes to C[i,k]
                    # With coefficients u_c, v_c, w_c
                    terms.append((i, j, k, u_c, v_c, w_c))
    
    # M1 = (A11+A22) * (B11+B22)
    # This expands to 4 block multiplications:
    # A11*B11, A11*B22, A22*B11, A22*B22
    # All with u=1, v=1, w=1
    # C11 gets +1, C12 gets +1, C21 gets +1, C22 gets +1 from M1
    
    # Actually, I need to be more careful. Let me think about this differently.
    
    # In the tensor decomposition, each rank-1 term is (i,j,k) with coefficients.
    # The Strassen algorithm computes:
    # C11 = M1 + M4 - M5 + M7
    # where each M is a 2x2 product.
    
    # M1 = (A11+A22)*(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # Each of these is a 2x2*2x2 product contributing to a 2x2 block of C.
    
    # Let me just enumerate all the scalar-level terms.
    
    # For M1 = (A11+A22)*(B11+B22):
    # The (di,dj) element of M1 = sum over (di2,dj2) of (A11[di,di2]+A22[di,di2])*(B11[dj2,dj]+B22[dj2,dj])
    # = sum over (di2,dj2) of A11[di,di2]*B11[dj2,dj] + A11[di,di2]*B22[dj2,dj] + A22[di,di2]*B11[dj2,dj] + A22[di,di2]*B22[dj2,dj]
    
    # This is getting complex. Let me use a different approach.
    
    # I'll build the decomposition by directly specifying which (i,j,k) indices get which coefficients.
    
    # Reset terms
    terms = []
    
    # For the 4x4 x 4x4 part (columns 0-3 of B):
    # Strassen terms:
    
    # M1 = (A11+A22)*(B11+B22)
    # A11 = A[0:2,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B22 = B[2:4,2:4]
    # Contribution to C11 = C[0:2,0:2], C12 = C[0:2,2:4], C21 = C[2:4,0:2], C22 = C[2:4,2:4]
    
    # M1 contributes +1 to C11, C12, C21, C22
    
    # Let me define: for a 2x2 block product A_block * B_block -> C_block
    # where A_block covers A[ar:ar+2, ac:ac+2]
    # B_block covers B[br:br+2, bc:bc+2]
    # C_block covers C[cr:cr+2, cc:cc+2]
    # The scalar terms are: for each (di,dj,dk) in {0,1}^3:
    #   i = ar+di, j = ac+dj, k = bc+dk
    #   term: A[i,j] * B[j,k] -> C[i, cr+dk] ... wait, this doesn't work simply.
    
    # Actually, the issue is that in a 2x2 block product:
    # C_block[di,dk] = sum_dj A_block[di,dj] * B_block[dj,dk]
    # So A_block[di,dj] = A[ar+di, ac+dj]
    # B_block[dj,dk] = B[br+dj, bc+dk]
    # C_block[di,dk] = C[cr+di, cc+dk]
    
    # For this to be a valid matrix multiplication, we need br = ac (the column of A must match the row of B)
    # And the result goes to C[cr+di, cc+dk]
    
    # So for A_block at (ar, ac) and B_block at (br, bc) with br=ac:
    # C_block at (cr, cc)
    # Terms: for di in 0,1; dj in 0,1; dk in 0,1:
    #   i = ar+di, j = ac+dj = br+dj, k = bc+dk
    #   A[i,j] * B[j,k] contributes to C[i, cc+dk]
    #   But we want it to contribute to C[cr+di, cc+dk]
    #   So we need i = cr+di, meaning ar = cr
    #   And the column: cc+dk must equal the actual column index k=bc+dk, so cc=bc
    
    # So: A[ar:ar+2, ac:ac+2] * B[ac:ac+2, bc:bc+2] -> C[ar:ar+2, bc:bc+2]
    # with ar=cr and bc=cc
    
    # OK so for M1 = (A11+A22)*(B11+B22):
    # A11 at (0,0), A22 at (2,2)
    # B11 at (0,0), B22 at (2,2)
    # Products:
    # A11*B11: A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2]
    # A11*B22: A[0:2,0:2]*B[2:4,0:2] -> C[0:2,0:2]  (but B row 2:4, A col 0:2, mismatch!)
    
    # Wait, I think I'm confusing myself. In Strassen, the blocks are:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[A11*B11+A12*B21, A11*B12+A12*B22], [A21*B11+A22*B21, A21*B12+A22*B22]]
    
    # So A11 is top-left, A12 is top-right, A21 is bottom-left, A22 is bottom-right
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    
    # C11 = A11*B11 + A12*B21 (top-left of C)
    # C12 = A11*B12 + A12*B22 (top-right of C)
    # C21 = A21*B11 + A22*B21 (bottom-left of C)
    # C22 = A21*B12 + A22*B22 (bottom-right of C)
    
    # Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now, M1 = (A11+A22)*(B11+B22)
    # A11+A22 is a 2x2 matrix: (A11+A22)[di,dj] = A[di,dj] + A[di+2,dj+2]
    # B11+B22 is a 2x2 matrix: (B11+B22)[dj,dk] = B[dj,dk] + B[dj+2,dk+2]
    # M1[di,dk] = sum_dj (A[di,dj]+A[di+2,dj+2])*(B[dj,dk]+B[dj+2,dk+2])
    # = sum_dj A[di,dj]*B[dj,dk] + A[di,dj]*B[dj+2,dk+2] + A[di+2,dj+2]*B[dj,dk] + A[di+2,dj+2]*B[dj+2,dk+2]
    
    # So M1 contributes to C11 (which is C[0:2,0:2]):
    # C[di,dk] += A[di,dj]*B[dj,dk] (standard)
    # C[di,dk] += A[di,dj]*B[dj+2,dk+2] (cross term)
    # C[di,dk] += A[di+2,dj+2]*B[dj,dk] (cross term)
    # C[di+2,dj+2]... wait no, M1[di,dk] is the (di,dk) element of the 2x2 result, which goes to C11[di,dk] = C[di,dk]
    
    # So the terms for M1 contributing to C11:
    # For each di,dj,dk in {0,1}:
    # Term 1: A[di,dj]*B[dj,dk] -> C[di,dk]  (i=di, j=dj, k=dk)
    # Term 2: A[di,dj]*B[dj+2,dk+2] -> C[di,dk]  (i=di, j=dj, k=dk+2)
    # Term 3: A[di+2,dj+2]*B[dj,dk] -> C[di,dk]  (i=di+2, j=dj+2, k=dk)
    # Term 4: A[di+2,dj+2]*B[dj+2,dk+2] -> C[di,dk]  (i=di+2, j=dj+2, k=dk+2)
    
    # But wait, C[di,dk] for di,dk in {0,1} is C11. But term 3 has i=di+2 which is in {2,3}, so it should go to C21, not C11!
    
    # I think I'm making an error. Let me reconsider.
    
    # M1 is a 2x2 matrix. Its (di,dk) element is M1[di,dk].
    # C11 = M1 + M4 - M5 + M7, so C11[di,dk] = M1[di,dk] + M4[di,dk] - M5[di,dk] + M7[di,dk]
    # C11[di,dk] = C[di,dk] (since C11 is the top-left 2x2 block of C)
    
    # M1[di,dk] = sum_dj (A11+A22)[di,dj] * (B11+B22)[dj,dk]
    # (A11+A22)[di,dj] = A11[di,dj] + A22[di,dj] = A[di,dj] + A[di+2,dj+2]
    # (B11+B22)[dj,dk] = B11[dj,dk] + B22[dj,dk] = B[dj,dk] + B[dj+2,dk+2]
    
    # So M1[di,dk] = sum_dj (A[di,dj] + A[di+2,dj+2]) * (B[dj,dk] + B[dj+2,dk+2])
    # = sum_dj A[di,dj]*B[dj,dk] + A[di,dj]*B[dj+2,dk+2] + A[di+2,dj+2]*B[dj,dk] + A[di+2,dj+2]*B[dj+2,dk+2]
    
    # This all contributes to C[di,dk] (the top-left block).
    
    # So in the tensor decomposition, we have terms:
    # (i=di, j=dj, k=dk, u=1, v=1, w=1)  -> A[di,dj]*B[dj,dk] -> C[di,dk]
    # (i=di, j=dj, k=dk+2, u=1, v=1, w=1)  -> A[di,dj]*B[dj,dk+2] -> C[di,dk]
    # (i=di+2, j=dj+2, k=dk, u=1, v=1, w=1)  -> A[di+2,dj+2]*B[dj,dk] -> C[di,dk]
    # (i=di+2, j=dj+2, k=dk+2, u=1, v=1, w=1)  -> A[di+2,dj+2]*B[dj+2,dk+2] -> C[di,dk]
    
    # But wait, the third term: A[di+2,dj+2]*B[dj,dk] should contribute to C[di+2, dk] in standard mult, but here it's being added to C[di,dk]. That's the "cross" effect of Strassen.
    
    # OK so this is correct for the Strassen decomposition. The key insight is that the W matrix maps (k,i) to the output position, and in Strassen, the same (k,i) pair can appear with different coefficients in different terms.
    
    # Let me just build all the terms systematically.
    
    # I'll create a dictionary: (i,j,k) -> coefficient for U, V, W
    # Actually, each term is independent. Let me list all terms as (i, j, k, u_coeff, v_coeff, w_coeff)
    
    # For M1 = (A11+A22)*(B11+B22), contributing +1 to C11, C12, C21, C22:
    # Wait, M1 contributes to all four C blocks? Let me recheck.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M1 contributes to C11 (+1) and C22 (+1)
    
    # M1[di,dk] contributes to C11[di,dk] = C[di,dk] with coefficient +1
    # And to C22[di,dk] = C[di+2,dk+2] with coefficient +1
    
    # So for M1, for each di,dj,dk in {0,1}:
    # Term to C11: (i=di, j=dj, k=dk, u=1, v=1, w=1)
    # Term to C11: (i=di, j=dj, k=dk+2, u=1, v=1, w=1)
    # Term to C11: (i=di+2, j=dj+2, k=dk, u=1, v=1, w=1)
    # Term to C11: (i=di+2, j=dj+2, k=dk+2, u=1, v=1, w=1)
    # Term to C22: same (i,j,k) but w goes to C[di+2,dk+2]
    
    # Hmm, I think the way to handle this is:
    # Each "block multiplication" M = X*Y where X and Y are 2x2 blocks
    # produces 8 scalar terms (2*2*2)
    # Each term has specific (i,j,k) indices
    # The contribution to C is determined by which C block the M result goes to
    
    # Let me just hard-code all 76 terms.
    
    terms = []
    
    # M1 = (A11+A22)*(B11+B22)
    # X = A11+A22: X[di,dj] = A[di,dj] + A[di+2,dj+2]
    # Y = B11+B22: Y[dj,dk] = B[dj,dk] + B[dj+2,dk+2]
    # M1[di,dk] = sum_dj X[di,dj]*Y[dj,dk]
    # Contributes +1 to C11 (C[0:2,0:2]) and +1 to C22 (C[2:4,2:4])
    
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # X[di,dj] = A[di,dj] + A[di+2,dj+2]
                # Y[dj,dk] = B[dj,dk] + B[dj+2,dk+2]
                # Product: A[di,dj]*B[dj,dk] + A[di,dj]*B[dj+2,dk+2] + A[di+2,dj+2]*B[dj,dk] + A[di+2,dj+2]*B[dj+2,dk+2]
                # All contribute to C[di,dk] (C11) with +1 and C[di+2,dk+2] (C22) with +1
                
                # To C11: C[di,dk]
                terms.append((di, dj, dk, 1, 1, 1))
                terms.append((di, dj, dk+2, 1, 1, 1))
                terms.append((di+2, dj+2, dk, 1, 1, 1))
                terms.append((di+2, dj+2, dk+2, 1, 1, 1))
                
                # To C22: C[di+2,dk+2]
                terms.append((di, dj, dk, 1, 1, 1))
                terms.append((di, dj, dk+2, 1, 1, 1))
                terms.append((di+2, dj+2, dk, 1, 1, 1))
                terms.append((di+2, dj+2, dk+2, 1, 1, 1))
    
    # This is getting very complex and error-prone. Let me take a step back.
    
    # Given the complexity and the risk of errors, and given that the current score is 1.0 (schoolbook rank 112),
    # let me try a simpler approach: just return the schoolbook but verify it's correct.
    # Or, let me try a different, simpler decomposition.
    
    # Actually, let me just return the schoolbook for now to ensure correctness,
    # as implementing Strassen correctly in tensor form is quite involved.
    
    # Build schoolbook
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
# EVOLVE-BLOCK-END
