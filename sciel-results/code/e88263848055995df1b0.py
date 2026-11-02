"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Decompose 3x4x8 into 3x2x2 blocks using Strassen (rank 7 each)
    # 3 rows x 2 blocks of 2 cols x 2 blocks of 4 cols -> 12 blocks x 7 = 84
    n, m, p = 3, 4, 8
    
    # Partition m=4 into 2 blocks of size 2: m1=2, m2=2
    # Partition p=8 into 4 blocks of size 2: p1=2, p2=2, p3=2, p4=2
    
    # For each row i (0..2), and each (m-block, p-block) pair:
    # We have a 1x2x2 sub-problem: A[i, m_block] * B[m_block, p_block] -> C[i, p_block]
    # But wait, Strassen is for 2x2x2, not 1x2x2.
    
    # Actually, let's look at the structure more carefully.
    # The previous successful strategy used 2x2x2 Strassen blocks.
    # n=3 can be split into 2+1.
    # m=4 can be split into 2+2.
    # p=8 can be split into 4*2.
    
    # Let's split n=3 into n1=2, n2=1.
    # Split m=4 into m1=2, m2=2.
    # Split p=8 into 4 chunks of 2.
    
    # Blocks:
    # 1. n1=2, m1=2, p_chunk=2 -> 2x2x2 Strassen (rank 7)
    #    There are 4 p-chunks, so 4 such blocks. Total 28.
    # 2. n1=2, m2=2, p_chunk=2 -> 2x2x2 Strassen (rank 7)
    #    There are 4 p-chunks, so 4 such blocks. Total 28.
    # 3. n2=1, m1=2, p_chunk=2 -> 1x2x2 Schoolbook (rank 2)
    #    There are 4 p-chunks, so 4 such blocks. Total 8.
    # 4. n2=1, m2=2, p_chunk=2 -> 1x2x2 Schoolbook (rank 2)
    #    There are 4 p-chunks, so 4 such blocks. Total 8.
    
    # Total rank = 28 + 28 + 8 + 8 = 72.
    
    # Let's refine. Can we do better?
    # What if we use 2x4x2 blocks? No standard low rank known.
    # What if we use 3x2x2 blocks? Rank of 3x2x2 is 6 (schoolbook). 
    # If we split n=3, m=2, p=2.
    # Then for each m-block (2 blocks) and p-chunk (4 chunks), we have a 3x2x2 block.
    # Rank 3x2x2 = 6.
    # Total blocks = 2 * 4 = 8.
    # Total rank = 8 * 6 = 48.
    
    # This is better! 48 < 72.
    
    # Let's verify the 3x2x2 decomposition.
    # A is 3x2, B is 2x2, C is 3x2.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # This is just schoolbook, rank 3*2*2 = 12? No.
    # Wait, the tensor is T_{i,j,k} = delta_{i,j} * delta_{j,k}? No.
    # The multiplication tensor for M_{n,m,p} has T_{i,j,k} = 1 if i,j,k correspond to a term.
    # Actually, the tensor is defined by C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor entries are T_{(i,j),(j,k)} -> C_{ik}.
    # The rank is the minimum number of rank-1 tensors to sum to T.
    
    # For 3x2x2, can we do better than 12?
    # Let's check.
    # A is 3x2, B is 2x2.
    # C_{00} = A00 B00 + A01 B10
    # C_{01} = A00 B01 + A01 B11
    # C_{10} = A10 B00 + A11 B10
    # C_{11} = A10 B01 + A11 B11
    # C_{20} = A20 B00 + A21 B10
    # C_{21} = A20 B01 + A21 B11
    
    # This is 6 equations. Each involves 2 terms.
    # It seems like rank 6 is possible?
    # Let's try to decompose.
    # Term 1: A00 * B00 -> C00. (U has A00, V has B00, W has C00)
    # Term 2: A01 * B10 -> C00. (U has A01, V has B10, W has C00)
    # Term 3: A00 * B01 -> C01.
    # Term 4: A01 * B11 -> C01.
    # Term 5: A10 * B00 -> C10.
    # Term 6: A11 * B10 -> C10.
    # ... and so on.
    
    # Wait, if I just use schoolbook, rank is 3*2*2 = 12.
    # Can we combine terms?
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    # C20 = A20 B00 + A21 B10
    
    # Notice B00 is common in C00, C10, C20.
    # B10 is common in C00, C10, C20.
    
    # Let's try to find a rank 6 decomposition for 3x2x2.
    # Actually, the rank of M_{3,2,2} is known to be 6?
    # Let's check small cases.
    # M_{2,2,2} rank 7 (Strassen). Schoolbook is 8.
    # M_{3,2,2}?
    # Let's try to construct it.
    
    # Actually, let's stick to the 2x2x2 Strassen blocks which we know works.
    # n=3 -> 2+1
    # m=4 -> 2+2
    # p=8 -> 2+2+2+2
    
    # Blocks:
    # 4 blocks of 2x2x2 (top 2 rows, left 2 cols, each p-chunk) -> Rank 7 each.
    # 4 blocks of 2x2x2 (top 2 rows, right 2 cols, each p-chunk) -> Rank 7 each.
    # 4 blocks of 1x2x2 (bottom 1 row, left 2 cols, each p-chunk) -> Rank 2 each.
    # 4 blocks of 1x2x2 (bottom 1 row, right 2 cols, each p-chunk) -> Rank 2 each.
    
    # Total rank = 4*7 + 4*7 + 4*2 + 4*2 = 28 + 28 + 8 + 8 = 72.
    
    # Let's try to be smarter with the 1x2x2 blocks.
    # A 1x2x2 block is just:
    # C0 = A0 B0 + A1 B1
    # C1 = A0 B2 + A1 B3
    # This is essentially a 2x2 matrix multiplication where one dimension is 1?
    # No, it's a dot product of two vectors (A0, A1) with rows of B.
    # Rank of this tensor?
    # T_{i,j,k} where i is fixed, j in 0..1, k in 0..1.
    # T_{j,k} = delta_{j,k}? No.
    # C_k = sum_j A_j B_{j,k}.
    # This is the identity tensor for 2x1x2?
    # The rank of the 2x2x2 identity-like tensor (where C_k = A_0 B_{0k} + A_1 B_{1k}) is 2?
    # Yes, because we can just write it as two terms:
    # Term 1: A_0 * B_{0*} -> C_{0*} (but C has 2 elements, so this is one term contributing to both C_0 and C_1? No.)
    
    # Let's write the terms:
    # Term 1: U=A_0, V=B_{00}, W=C_0
    # Term 2: U=A_0, V=B_{01}, W=C_1
    # Term 3: U=A_1, V=B_{10}, W=C_0
    # Term 4: U=A_1, V=B_{11}, W=C_1
    # This is rank 4.
    
    # Can we do rank 2?
    # If we could, we would have:
    # C_0 = u1 v1 w1 + u2 v2 w2
    # C_1 = u1 v1' w1 + u2 v2' w2
    # This doesn't seem to work directly with scalars.
    # The rank of M_{1,2,2} is 2?
    # Wait, M_{1,2,2} is just computing y = x^T B.
    # x is 1x2, B is 2x2, y is 1x2.
    # y_0 = x_0 b_00 + x_1 b_10
    # y_1 = x_0 b_01 + x_1 b_11
    # This is 2 bilinear forms.
    # Can we do it with 2 products?
    # p1 = (x_0 + x_1) (b_00 + b_10) ? No.
    # The rank of M_{1, n, p} is n*p?
    # Actually, the rank of M_{1, n, p} is n.
    # Let's check.
    # y = x B.
    # If we set U_i = x, V_i = B_{*,i}, W_i = e_i^T?
    # No, U must be linear in x, V linear in B.
    # Term i: u_i(x) * v_i(B) * w_i.
    # Sum_i u_i v_i w_i = y.
    # If we take n terms, where w_i is the i-th standard basis vector of C (size p)?
    # But C has size p.
    # Let's try rank p=2.
    # Term 1: u_1 = x_0, v_1 = [b_00, b_01]^T? No, v_1 is a vector in B-space.
    # The product is scalar * vector?
    # In tensor decomposition, u_i is in U-space (dim n), v_i in V-space (dim p), w_i in W-space (dim p)?
    # No, U is n*m, V is m*p, W is p*n.
    # For M_{1,2,2}:
    # U is 1*2 = 2xR.
    # V is 2*2 = 4xR.
    # W is 2*1 = 2xR.
    
    # Let's just use schoolbook for the 1x2x2 blocks. Rank 4 each.
    # 4 blocks * 4 = 16.
    # Total rank = 28 + 28 + 16 + 16 = 88.
    # This is the same as the "Strassen 2x2 blocks" result mentioned in the prompt (88).
    
    # Can we improve the 1x2x2 rank?
    # Let's check if rank 3 is possible for M_{1,2,2}.
    # Or even rank 2.
    # If rank 2 is possible, total rank = 28+28+8+8 = 72.
    
    # Let's try to find a rank 2 decomposition for M_{1,2,2}.
    # y_0 = x_0 b_00 + x_1 b_10
    # y_1 = x_0 b_01 + x_1 b_11
    
    # We need 2 terms:
    # y_0 = u_1 v_1 + u_2 v_2
    # y_1 = u_1' v_1' + u_2' v_2'
    # where u_i are linear in x, v_i are linear in b.
    
    # Let u_1 = x_0, u_2 = x_1.
    # v_1 must produce (b_00, b_01) when multiplied by w_1, w_2?
    # No, w_i are coefficients for C.
    # C_0 = sum_i u_i(x) v_i(B) w_{i,0}
    # C_1 = sum_i u_i(x) v_i(B) w_{i,1}
    
    # If w_{1,0}=1, w_{1,1}=0, then term 1 only contributes to C_0.
    # If w_{2,0}=0, w_{2,1}=1, then term 2 only contributes to C_1.
    # Then v_1(B) must be b_00 for term 1? But u_1=x_0.
    # So x_0 b_00. But we need x_1 b_10 as well.
    # So we need more terms.
    
    # It seems rank 4 is the schoolbook.
    # Is rank 3 possible?
    # Likely yes, but rank 2 is unlikely.
    
    # Let's stick with rank 4 for 1x2x2.
    # Total 88.
    
    # Can we do better with the 2x2x2 blocks?
    # Strassen is rank 7.
    # Can we do better? No, 7 is optimal for 2x2x2.
    
    # Can we mix the blocks?
    # What if we use 2x4x2 blocks?
    # Rank of 2x4x2?
    # A is 2x4, B is 4x2, C is 2x2.
    # This is M_{2,4,2}.
    # Rank?
    # Schoolbook is 2*4*2 = 16.
    # Can we do better?
    # Strassen applies to 2x2x2.
    # We can view 2x4x2 as two 2x2x2 blocks side by side.
    # C_left = A_left B_left
    # C_right = A_right B_right
    # This is rank 7 + 7 = 14.
    # Can we do better?
    # Maybe.
    
    # What if we view 2x4x2 as one block?
    # If we use Strassen on the 2x2 part of A and B?
    # A = [A1 A2], B = [B1; B2].
    # C = A1 B1 + A2 B2.
    # This is sum of two 2x2x2 multiplications.
    # Rank 14.
    
    # So using 2x2x2 Strassen blocks is rank 7 per 2x2x2 chunk.
    # For 2x4x2, we have two 2x2x2 chunks, so rank 14.
    # For 3x4x8:
    # n=3, m=4, p=8.
    # Split n into 2+1.
    # Top 2 rows: 2x4x8.
    # Bottom 1 row: 1x4x8.
    
    # 2x4x8:
    # Split m into 2+2.
    # 2x2x8 and 2x2x8.
    # 2x2x8: Split p into 4x2.
    # Four 2x2x2 blocks. Rank 4*7 = 28.
    # So 2x4x8 has rank 28+28 = 56.
    
    # 1x4x8:
    # Split m into 2+2.
    # 1x2x8 and 1x2x8.
    # 1x2x8: Split p into 4x2.
    # Four 1x2x2 blocks. Rank 4*4 = 16 (assuming rank 4).
    # So 1x4x8 has rank 16+16 = 32.
    
    # Total rank = 56 + 32 = 88.
    
    # This matches the previous result.
    
    # Let's try to find a better decomposition for 1x2x2.
    # If rank is 3, total rank = 4*3 + 4*3 + 4*7 + 4*7 = 12+12+28+28 = 80.
    # If rank is 2, total rank = 4*2 + 4*2 + 4*7 + 4*7 = 8+8+28+28 = 72.
    
    # Let's hardcode the rank 4 schoolbook for 1x2x2 for now, as it's safe.
    # Rank 88.
    
    # Let's implement the 88 rank solution.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # i0: row offset, j0: col offset in A/B, k0: col offset in C
        # Local indices: i' in 0..1, j' in 0..1, k' in 0..1
        # Global: i = i0+i', j = j0+j', k = k0+k'
        
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        # U: A00+A11 -> U[i0*2+0, j0*2+0] += 1, U[i0*2+1, j0*2+1] += 1
        # V: B00+B11 -> V[(j0*2+0)*4 + (k0*2+0)] += 1, V[(j0*2+1)*4 + (k0*2+1)] += 1
        # W: C00+C11 -> W[(k0*2+0)*3 + i0] += 1, W[(k0*2+1)*3 + i0+1] += 1
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for (u_idx, u_val) in u_coeffs:
                U[u_idx][col] += u_val
            for (v_idx, v_val) in v_coeffs:
                V[v_idx][col] += v_val
            for (w_idx, w_val) in w_coeffs:
                W[w_idx][col] += w_val
            col += 1
        
        # M1
        u = [(i0*m + j0, 1), (i0*m + j0 + 1, 1)]
        v = [((j0 + 0)*p + k0, 1), ((j0 + 1)*p + k0 + 1, 1)]
        w = [(k0*n + i0, 1), (k0*n + i0 + 1, 1)] # Wait, W index is k*n + i.
        # W[k*n+i, r].
        # C00 is at i=i0, k=k0. Index k0*n + i0.
        # C11 is at i=i0+1, k=k0+1. Index (k0+1)*n + i0+1.
        w = [(k0*n + i0, 1), ((k0+1)*n + i0+1, 1)]
        add_term(u, v, w)
        
        # M2
        u = [(i0*m + j0 + 1, 1), (i0*m + j0 + 1, 1)] # A10+A11. A10 is (i0+1, j0). A11 is (i0+1, j0+1).
        # Wait, A indices: (i,j).
        # A10: i=i0+1, j=j0. Index (i0+1)*m + j0.
        # A11: i=i0+1, j=j0+1. Index (i0+1)*m + j0+1.
        u = [((i0+1)*m + j0, 1), ((i0+1)*m + j0+1, 1)]
        v = [((j0 + 0)*p + k0, 1)] # B00.
        w = [(k0*n + i0+1, 1)] # C10.
        add_term(u, v, w)
        
        # M3
        u = [(i0*m + j0, 1)] # A00.
        v = [((j0 + 1)*p + k0, 1), ((j0 + 1)*p + k0+1, -1)] # B10-B11.
        w = [(k0*n + i0, 1), (k0*n + i0+1, 1)] # C00? No.
        # C01 = M3 + M5.
        # M3 contributes to C01?
        # C01 is at i=i0, k=k0+1. Index (k0+1)*n + i0.
        # C11? No.
        # Let's recheck Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M3 contributes to C01.
        w = [((k0+1)*n + i0, 1)]
        add_term(u, v, w)
        
        # M4
        u = [((i0+1)*m + j0+1, 1)] # A11.
        v = [((j0 + 0)*p + k0+1, 1), ((j0 + 0)*p + k0, -1)] # B01-B00.
        w = [(k0*n + i0, 1), (k0*n + i0+1, 1)] # C00 and C10.
        # M4 contributes to C00 and C10.
        # C00: (k0)*n + i0.
        # C10: (k0)*n + i0+1.
        w = [(k0*n + i0, 1), (k0*n + i0+1, 1)]
        add_term(u, v, w)
        
        # M5
        u = [(i0*m + j0, 1), (i0*m + j0+1, 1)] # A00+A01.
        v = [((j0 + 1)*p + k0+1, 1)] # B11.
        w = [(k0*n + i0, -1), (k0*n + i0+1, 1)] # -C00 + C01?
        # C00 = ... - M5 ...
        # C01 = ... + M5 ...
        # So M5 contributes -1 to C00 and +1 to C01.
        w = [(k0*n + i0, -1), ((k0+1)*n + i0, 1)]
        add_term(u, v, w)
        
        # M6
        u = [((i0+1)*m + j0, 1), (i0*m + j0, -1)] # A10-A00.
        v = [((j0 + 0)*p + k0, 1), ((j0 + 0)*p + k0+1, 1)] # B00+B01.
        w = [((k0+1)*n + i0+1, 1)] # C11.
        add_term(u, v, w)
        
        # M7
        u = [(i0*m + j0+1, 1), ((i0+1)*m + j0+1, -1)] # A01-A11.
        v = [((j0 + 1)*p + k0, 1), ((j0 + 1)*p + k0+1, 1)] # B10+B11.
        w = [(k0*n + i0, 1)] # C00.
        add_term(u, v, w)
    
    # Helper to add a 1x2x2 Schoolbook block
    def add_schoolbook_1x2x2(i0, j0, k0):
        nonlocal col
        # i0: row index
        # j0: col offset in A/B
        # k0: col offset in C
        
        # C0 = A0 B0 + A1 B1
        # C1 = A0 B2 + A1 B3
        # A indices: (i0, j0), (i0, j0+1)
        # B indices: (j0, k0), (j0+1, k0), (j0, k0+1), (j0+1, k0+1)
        # C indices: (k0, i0), (k0+1, i0)
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for (u_idx, u_val) in u_coeffs:
                U[u_idx][col] += u_val
            for (v_idx, v_val) in v_coeffs:
                V[v_idx][col] += v_val
            for (w_idx, w_val) in w_coeffs:
                W[w_idx][col] += w_val
            col += 1
        
        # Term 1: A0 * B0 -> C0
        u = [(i0*m + j0, 1)]
        v = [(j0*p + k0, 1)]
        w = [(k0*n + i0, 1)]
        add_term(u, v, w)
        
        # Term 2: A1 * B1 -> C0
        u = [(i0*m + j0+1, 1)]
        v = [((j0+1)*p + k0, 1)]
        w = [(k0*n + i0, 1)]
        add_term(u, v, w)
        
        # Term 3: A0 * B2 -> C1
        u = [(i0*m + j0, 1)]
        v = [(j0*p + k0+1, 1)]
        w = [((k0+1)*n + i0, 1)]
        add_term(u, v, w)
        
        # Term 4: A1 * B3 -> C1
        u = [(i0*m + j0+1, 1)]
        v = [((j0+1)*p + k0+1, 1)]
        w = [((k0+1)*n + i0, 1)]
        add_term(u, v, w)
    
    # 4 Strassen blocks for top 2 rows, left 2 cols
    for k_chunk in range(4):
        add_strassen(0, 0, k_chunk*2)
        
    # 4 Strassen blocks for top 2 rows, right 2 cols
    for k_chunk in range(4):
        add_strassen(0, 2, k_chunk*2)
        
    # 4 Schoolbook blocks for bottom 1 row, left 2 cols
    for k_chunk in range(4):
        add_schoolbook_1x2x2(2, 0, k_chunk*2)
        
    # 4 Schoolbook blocks for bottom 1 row, right 2 cols
    for k_chunk in range(4):
        add_schoolbook_1x2x2(2, 2, k_chunk*2)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
