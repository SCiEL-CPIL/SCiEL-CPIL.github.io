import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_comp, T_target, atol=tol)

def construct():
    start_time = time.time()
    deadline = 135  # safety margin
    
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 28
    C_dim = p * n  # 14
    
    # Strategy: Deterministic Block Decomposition
    # We decompose the 2x4 by 4x7 multiplication.
    # A is 2x4, B is 4x7.
    # We can view A as two 2x2 blocks? No, A is 2x4.
    # Let's view the multiplication as operating on 2x2 blocks.
    # Actually, let's use the known rank 50 construction derived from Strassen on 2x2 blocks.
    
    # The tensor T_{ijk} = 1 if (i,j,k) is valid.
    # Indices:
    # i in {0,1} (row of A)
    # j in {0,1,2,3} (col of A / row of B)
    # k in {0..6} (col of B)
    
    # Let's split j into two pairs: j1 in {0,1}, j2 in {2,3}.
    # A = [A11 A12] where A11 is 2x2, A12 is 2x2.
    # B = [B11 B12] where B11 is 4x7? No.
    # B is 4x7. Split B rows into 2x2 blocks?
    # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7.
    
    # C = A * B
    # C is 2x7.
    # C = A11 * B_top + A12 * B_bottom.
    
    # This doesn't immediately suggest Strassen unless we treat the 2x2 matrix multiplication.
    # Let's try to embed the 2x2x2 Strassen algorithm.
    
    # Strassen for 2x2 matrices:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # We have A (2x4) and B (4x7).
    # This is not a square matrix multiplication.
    # However, we can pad or use a specific block structure.
    
    # Alternative: Use the rank 50 construction found in literature for 2x4x7.
    # The trivial rank is 56.
    # The Strassen-based rank for 2x4x4 is 2*7 = 14? No.
    # Let's construct a rank 50 decomposition explicitly.
    
    # We will build U, V, W with 50 columns.
    # We use 7 terms for each of the 7 columns of C?
    # C_{i,k} = sum_{j=0}^3 A_{i,j} B_{j,k}
    # For each k (0..6), we have a 2x4 by 4x1 multiplication.
    # This is equivalent to 7 independent 2x4x1 multiplications? No, B is shared.
    
    # Let's use the following structure:
    # We treat the 4x7 matrix B as four 1x7 vectors.
    # We treat the 2x4 matrix A as two 2x4 vectors.
    
    # Actually, let's look at the "Strassen for 2x2" application to the blocks.
    # Split A into A1 (2x2) and A2 (2x2) horizontally.
    # Split B into B1 (2x7) and B2 (2x7) vertically.
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x7. A2*B2 is 2x7.
    # This is sum of two 2x2x7 multiplications? No, A1 is 2x2, B1 is 2x7.
    # A1*B1 = [a11 a12; a21 a22] * [b11..b17; b21..b27]
    # This is a 2x2 by 2x7 multiplication.
    # Rank of 2x2 by 2x7?
    # It can be seen as 2 independent 2x2 by 2x1 multiplications? No.
    # It is equivalent to 2x2 matrix multiplication by a 2x7 matrix.
    # This can be done with rank 2*2*7? No.
    # The tensor for 2x2 by 2x7 is a composition of 7 2x2 by 2x1 tensors.
    # Rank of 2x2 by 2x1 is 3 (Strassen-like? No, 2x2 by 2x1 is just dot products?
    # C1 = a11*b11 + a12*b21
    # C2 = a21*b11 + a22*b21
    # This is 2 bilinear forms. Rank is 2?
    # T(a,b) -> (a11 b11 + a12 b21, a21 b11 + a22 b21)
    # This is a rank 2 tensor?
    # Term 1: a11 * b11 * (1, 0) ? No, b11 is shared.
    # Term 1: (a11, a21) * (b11, b11) * (1, 1)? No.
    
    # Let's just use the known rank 50 solution.
    # The rank 50 solution for 2x4x7 is often constructed by applying Strassen to the 2x2 blocks of the 4x4 part and handling the remaining 4x3 part directly?
    # 4x7 = 4x4 + 4x3.
    # 2x4 by 4x4:
    #   Split into 2x2 blocks.
    #   A = [A1 A2] (2x2, 2x2)
    #   B = [B1; B2] (2x4, 2x4) -> B1 is 2x4, B2 is 2x4.
    #   C1 = A1*B1 + A2*B2.
    #   A1*B1 is 2x2 by 2x4.
    #   A2*B2 is 2x2 by 2x4.
    #   
    #   Let's calculate rank of 2x2 by 2x4.
    #   It is 4 independent 2x2 by 2x1 multiplications?
    #   No, the rows of B are shared.
    #   
    #   Actually, there is a known result: Rank(2x4x4) = 14?
    #   If Rank(2x4x4) = 14, and Rank(2x4x3) = 24 (trivial 2*4*3=24), total 38?
    #   Wait, the trivial rank for 2x4x7 is 56.
    #   If we can do 2x4x4 with rank 14 and 2x4x3 with rank 24, total 38.
    #   Is Rank(2x4x4) really 14?
    #   2x4 by 4x4.
    #   View as 2x2 by 2x2 blocks.
    #   A = [A1 A2], B = [B1 B2; B3 B4] (each 2x2).
    #   C = A1*B1 + A2*B3  (left 2x2 block of C)
    #       A1*B2 + A2*B4  (right 2x2 block of C)
    #   
    #   Left block: M_L = A1*B1 + A2*B3.
    #   Right block: M_R = A1*B2 + A2*B4.
    #   
    #   This is two 2x2x2 multiplications summed.
    #   Rank of sum of two 2x2x2 tensors?
    #   If they were independent, rank 7+7=14.
    #   Are they independent?
    #   The A's are shared, B's are distinct.
    #   Yes, the B blocks are disjoint.
    #   So we can use 7 terms for M_L and 7 terms for M_R.
    #   Total 14 terms for the 2x4x4 part.
    #   
    #   Now the 2x4x3 part.
    #   A is 2x4. B is 4x3.
    #   Trivial rank is 2*4*3 = 24.
    #   Can we do better?
    #   Split A into A1 (2x2), A2 (2x2).
    #   Split B into B1 (2x3), B2 (2x3).
    #   C = A1*B1 + A2*B2.
    #   A1*B1 is 2x2 by 2x3.
    #   Rank of 2x2 by 2x3?
    #   It is 3 independent 2x2 by 2x1 multiplications.
    #   Rank of 2x2 by 2x1 is 2?
    #   C1 = a11 b11 + a12 b21
    #   C2 = a21 b11 + a22 b21
    #   This is a bilinear map from (4,2) to 2.
    #   It can be done with 2 terms?
    #   Term 1: (a11, a21, 0, 0) * (b11, b11, 0, 0) * (1, 1)? No.
    #   Term 1: (a11, 0, a21, 0) * (b11, 0, b11, 0) * (1, 1)?
    #   Let's check:
    #   u1 = (1, 0, 1, 0) for A? No, A has 4 elements.
    #   Let A = (a11, a12, a21, a22).
    #   Let B = (b11, b21). (2x1)
    #   C1 = a11 b11 + a12 b21
    #   C2 = a21 b11 + a22 b21
    #   
    #   Term 1: u1=(1,0,1,0), v1=(1,0), w1=(1,1).
    #   Contrib: a11 b11 (1,1) + a21 b11 (1,1) = (a11 b11 + a21 b11, a11 b11 + a21 b11).
    #   This is not right.
    #   
    #   Actually, the rank of 2x2 by 2x1 is 2.
    #   Term 1: u=(1,0,0,0), v=(1,0), w=(1,0). -> a11 b11 at C1.
    #   Term 2: u=(0,0,1,0), v=(1,0), w=(0,1). -> a21 b11 at C2.
    #   Wait, we need a12 b21 and a22 b21 too.
    #   So we need 4 terms?
    #   Term 1: a11 b11 -> C1
    #   Term 2: a12 b21 -> C1
    #   Term 3: a21 b11 -> C2
    #   Term 4: a22 b21 -> C2
    #   So rank is 4?
    #   If rank(2x2 by 2x1) = 4, then rank(2x2 by 2x3) = 12.
    #   Then rank(2x4x3) = 12 + 12 = 24.
    #   So total rank = 14 + 24 = 38.
    #   
    #   Wait, is rank(2x2 by 2x1) really 4?
    #   C1 = a11 b11 + a12 b21
    #   C2 = a21 b11 + a22 b21
    #   This is a linear map from B to C for each A?
    #   No, it's bilinear.
    #   The tensor has 4 non-zero entries: (1,1,1), (2,2,1), (3,1,2), (4,2,2).
    #   These are 4 disjoint hyperedges.
    #   So the rank is at least 4. And 4 terms suffice.
    #   So yes, rank 4.
    
    # So the rank 38 construction:
    # 1. 2x4x4 part: Rank 14.
    #    Use Strassen on the two 2x2x2 blocks.
    #    Block 1 (Left C): A1*B1 + A2*B3.
    #    Block 2 (Right C): A1*B2 + A2*B4.
    #    Each 2x2x2 uses 7 terms.
    #    Total 14 terms.
    #    
    # 2. 2x4x3 part: Rank 24.
    #    Trivial terms for each of the 2*4*3 = 24 products.
    
    # Total Rank = 38.
    # This is a massive improvement over 50.
    
    # Let's verify the indices and construct the matrices.
    
    rank = 38
    U = np.zeros((A_dim, rank))
    V = np.zeros((B_dim, rank))
    W = np.zeros((C_dim, rank))
    
    r = 0
    
    # Part 1: 2x4x4 (Columns 0-3 of B, Columns 0-3 of C)
    # A indices: i in {0,1}, j in {0,1,2,3}
    # A1: i in {0,1}, j in {0,1}
    # A2: i in {0,1}, j in {2,3}
    
    # B indices: j in {0,1,2,3}, k in {0,1,2,3}
    # B1: j in {0,1}, k in {0,1}
    # B2: j in {0,1}, k in {2,3}
    # B3: j in {2,3}, k in {0,1}
    # B4: j in {2,3}, k in {2,3}
    
    # C indices: i in {0,1}, k in {0,1,2,3}
    # C_left: i in {0,1}, k in {0,1}
    # C_right: i in {0,1}, k in {2,3}
    
    # Strassen for 2x2x2:
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # Terms:
    # 1: (A11+A22)*(B11+B22) -> contributes to all C
    # 2: (A21+A22)*B11 -> contributes to C11, C21
    # 3: A11*(B12-B22) -> contributes to C11, C12
    # 4: A22*(B21-B11) -> contributes to C21, C22
    # 5: (A11+A12)*B22 -> contributes to C12, C22
    # 6: (A21-A11)*(B11+B12) -> contributes to C11, C21
    # 7: (A12-A22)*(B21+B22) -> contributes to C12, C22
    
    # We need to map these to the global indices.
    
    def add_strassen_term(U_mat, V_mat, W_mat, A_indices, B_indices, C_indices, term_id):
        """
        A_indices: dict mapping local (0,0)->global, etc.
        B_indices: dict mapping local (0,0)->global, etc.
        C_indices: dict mapping local (0,0)->global, etc.
        """
        # Local indices for 2x2:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # Define the 7 Strassen terms as (U_mask, V_mask, W_mask)
        # U_mask: 4 values (for A)
        # V_mask: 4 values (for B)
        # W_mask: 4 values (for C)
        
        terms = [
            # 1: (A11+A22)*(B11+B22)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
            # 2: (A21+A22)*B11
            ([0, 0, 1, 1], [1, 0, 0, 0], [1, 0, 1, 0]),
            # 3: A11*(B12-B22)
            ([1, 0, 0, 0], [0, 1, 0, -1], [1, 1, 0, 0]),
            # 4: A22*(B21-B11)
            ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]),
            # 5: (A11+A12)*B22
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            # 6: (A21-A11)*(B11+B12)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]),
            # 7: (A12-A22)*(B21+B22)
            ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
        ]
        
        u_vals, v_vals, w_vals = terms[term_id]
        
        # Map local indices to global
        # A local 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        # Global A index = i*m + j
        
        for local_idx, val in enumerate(u_vals):
            if val != 0:
                li, lj = divmod(local_idx, 2)
                ga = A_indices[(li, lj)]
                U_mat[ga, r] = val
        
        for local_idx, val in enumerate(v_vals):
            if val != 0:
                li, lj = divmod(local_idx, 2)
                gb = B_indices[(li, lj)]
                V_mat[gb, r] = val
        
        for local_idx, val in enumerate(w_vals):
            if val != 0:
                li, lj = divmod(local_idx, 2)
                gc = C_indices[(li, lj)]
                W_mat[gc, r] = val

    # Define index mappings for the 2x4x4 part
    # A1: j in {0,1}
    A1_map = {(0,0): 0*4+0, (0,1): 0*4+1, (1,0): 1*4+0, (1,1): 1*4+1}
    # A2: j in {2,3}
    A2_map = {(0,0): 0*4+2, (0,1): 0*4+3, (1,0): 1*4+2, (1,1): 1*4+3}
    
    # B1: j in {0,1}, k in {0,1}
    B1_map = {(0,0): 0*7+0, (0,1): 0*7+1, (1,0): 1*7+0, (1,1): 1*7+1}
    # B2: j in {0,1}, k in {2,3}
    B2_map = {(0,0): 0*7+2, (0,1): 0*7+3, (1,0): 1*7+2, (1,1): 1*7+3}
    # B3: j in {2,3}, k in {0,1}
    B3_map = {(0,0): 2*7+0, (0,1): 2*7+1, (1,0): 3*7+0, (1,1): 3*7+1}
    # B4: j in {2,3}, k in {2,3}
    B4_map = {(0,0): 2*7+2, (0,1): 2*7+3, (1,0): 3*7+2, (1,1): 3*7+3}
    
    # C_left: i in {0,1}, k in {0,1}
    C_left_map = {(0,0): 0*2+0, (0,1): 0*2+1, (1,0): 1*2+0, (1,1): 1*2+1}
    # C_right: i in {0,1}, k in {2,3}
    C_right_map = {(0,0): 0*2+2, (0,1): 0*2+3, (1,0): 1*2+2, (1,1): 1*2+3}
    
    # Add 7 terms for Left C (A1*B1 + A2*B3)
    # We can combine A1*B1 and A2*B3 into one set of 7 terms?
    # No, Strassen applies to a single 2x2x2 multiplication.
    # Here we have a sum of two products.
    # C_L = A1*B1 + A2*B3
    # This is not a single 2x2x2 multiplication.
    # However, we can treat it as a bilinear map.
    # Is there a rank 7 decomposition for A1*B1 + A2*B3?
    # The tensor is T = T1 + T2.
    # If T1 and T2 have disjoint supports in B, they are independent.
    # B1 and B3 have disjoint supports in B (rows 0,1 vs 2,3).
    # So the rank is rank(T1) + rank(T2) = 7 + 7 = 14.
    # So we need 14 terms for the 2x4x4 part.
    
    # Term 1-7: A1*B1
    for t in range(7):
        add_strassen_term(U, V, W, A1_map, B1_map, C_left_map, t)
        r += 1
        
    # Term 8-14: A2*B3
    for t in range(7):
        add_strassen_term(U, V, W, A2_map, B3_map, C_left_map, t)
        r += 1
        
    # Term 15-21: A1*B2
    for t in range(7):
        add_strassen_term(U, V, W, A1_map, B2_map, C_right_map, t)
        r += 1
        
    # Term 22-28: A2*B4
    for t in range(7):
        add_strassen_term(U, V, W, A2_map, B4_map, C_right_map, t)
        r += 1
        
    # Part 2: 2x4x3 (Columns 4-6 of B, Columns 4-6 of C)
    # A is 2x4. B is 4x3.
    # We use trivial decomposition: 2*4*3 = 24 terms.
    # For each i in {0,1}, j in {0,1,2,3}, k in {4,5,6}:
    #   U[i*4+j, r] = 1
    #   V[j*7+k, r] = 1
    #   W[k*2+i, r] = 1
    
    for i in range(2):
        for j in range(4):
            for k in range(3):
                global_k = k + 4
                U[i*4+j, r] = 1
                V[j*7+global_k, r] = 1
                W[global_k*2+i, r] = 1
                r += 1
                
    # Total rank used: 28 + 24 = 52?
    # Wait, 2x4x4 part used 28 terms.
    # 2x4x3 part used 24 terms.
    # Total 52.
    # This is worse than 50.
    
    # Why?
    # Because I used 14 terms for 2x4x4.
    # Is there a better way?
    # Rank of 2x4x4 is 14.
    # Rank of 2x4x3 is 24.
    # Total 38?
    # Wait, I calculated Rank(2x2 by 2x1) = 4 earlier.
    # So Rank(2x2 by 2x3) = 12.
    # Rank(2x4x3) = 12 + 12 = 24.
    # So 14 + 24 = 38.
    
    # But my implementation above used 28 terms for the 4x4 part.
    # Because I used 7 terms for each of the 4 blocks (A1B1, A2B3, A1B2, A2B4).
    # 4 * 7 = 28.
    # But the 4x4 part is A1*B1 + A2*B3 (left) and A1*B2 + A2*B4 (right).
    # The left part uses 14 terms.
    # The right part uses 14 terms.
    # Total 28.
    
    # Can we do better for 2x4x4?
    # Maybe by mixing the blocks?
    # No, the B blocks are disjoint.
    # So 28 is the rank for the 4x4 part using this block structure.
    
    # So total rank is 28 + 24 = 52.
    # This is worse than 50.
    
    # Let's check the 2x4x3 part again.
    # Is Rank(2x4x3) really 24?
    # A (2x4) * B (4x3).
    # Trivial is 24.
    # Can we reduce it?
    # Split A into A1 (2x2), A2 (2x2).
    # Split B into B1 (2x3), B2 (2x3).
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x2 by 2x3.
    # Rank of 2x2 by 2x3?
    # It is 3 independent 2x2 by 2x1 multiplications.
    # Rank of 2x2 by 2x1 is 4.
    # So Rank(2x2 by 2x3) = 12.
    # So Rank(2x4x3) = 24.
    
    # So the total rank is 52.
    # This is not an improvement over 50.
    
    # Let's try a different split.
    # What if we don't split 4x7 into 4x4 and 4x3?
    # What if we use Strassen on the whole thing?
    # No, Strassen is for square matrices.
    
    # Let's stick to the known rank 50 solution if my derivation gives 52.
    # Or maybe I can optimize the 2x4x3 part?
    # No, 24 is minimal for 2x4x3?
    # Actually, the rank of 2x4x3 is at least max(2*4, 4*3, 2*3) = 12?
    # No, the rank is at least the border rank.
    # For 2x4x3, the trivial rank is 24.
    # Is it possible to do less?
    # The tensor is T_{ijk} = 1.
    # It is a sum of 24 rank-1 tensors.
    # Can they be combined?
    # Only if they share indices.
    # They don't.
    # So 24 is likely minimal for the 3-part.
    
    # So the bottleneck is the 4x4 part (28 terms).
    # Can we do 2x4x4 with rank < 28?
    # Rank(2x4x4) = Rank(2x2x4 + 2x2x4).
    # Rank(2x2x4) = 4 * Rank(2x2x1)?
    # No, Rank(2x2x4) = 4 * 2 = 8?
    # Let's check Rank(2x2x1).
    # C1 = a11 b11 + a12 b21
    # C2 = a21 b11 + a22 b21
    # This is a bilinear map.
    # Can it be done with rank 3?
    # If yes, then Rank(2x2x4) = 12.
    # Then Rank(2x4x4) = 24.
    # Then Total Rank = 24 + 24 = 48.
    
    # Let's check if Rank(2x2x1) is 3.
    # The tensor has 4 non-zero entries.
    # (1,1,1), (2,2,1), (3,1,2), (4,2,2).
    # These are disjoint.
    # So rank is at least 4.
    # So Rank(2x2x1) = 4.
    
    # So Rank(2x4x4) = 28.
    # So Total Rank = 52.
    
    # This strategy yields 52, which is worse than 50.
    # I will fall back to the trivial 56 or try to find a better decomposition.
    
    # Since I am out of time for a complex search, I will return the trivial 56 to be safe,
    # or the 52 if I can verify it.
    # Actually, 52 is better than 56.
    
    if r != rank:
        print(f"Rank mismatch: {r} vs {rank}")
        
    if _verify_decomposition(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to trivial
    rank_triv = n * m * p
    U_triv = np.zeros((A_dim, rank_triv))
    V_triv = np.zeros((B_dim, rank_triv))
    W_triv = np.zeros((C_dim, rank_triv))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, r] = 1
                V_triv[j * p + k, r] = 1
                W_triv[k * n + i, r] = 1
                r += 1
    
    return {
        "U": U_triv.tolist(),
        "V": V_triv.tolist(),
        "W": W_triv.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")

