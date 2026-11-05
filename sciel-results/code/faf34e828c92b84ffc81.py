import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Target: Rank 36 decomposition using Strassen blocks for 2x2 submatrices
    # and Schoolbook for the remaining column.
    
    rank = 36
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    # Note: The task description says U, V, W shapes are (n*m, rank), (m*p, rank), (p*n, rank).
    # However, standard matrix multiplication tensor T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # where C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Indices:
    # A is n x m. Index i in [0,n), j in [0,m).
    # B is m x p. Index j in [0,m), k in [0,p).
    # C is n x p. Index i in [0,n), k in [0,p).
    # The task specifies:
    # U shape: (n*m, rank). So index for A is i*m + j.
    # V shape: (m*p, rank). So index for B is j*p + k.
    # W shape: (p*n, rank). "Output factor follows the official trace convention W[k*n+i,r]".
    # So W index is k*n + i.
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    current_rank = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_rows, a_cols: indices of rows/cols in A for the 2x2 block
    # b_rows, b_cols: indices of rows/cols in B for the 2x2 block
    # c_rows, c_cols: indices of rows/cols in C for the 2x2 block
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal current_rank
        
        # Strassen 7 multiplications
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Reconstruction
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Coefficients for U (A part)
        # Each multiplication r uses a specific linear combination of A entries.
        # A entries involved: A00, A01, A10, A11
        # Indices in U:
        # A00: a_rows[0]*m + a_cols[0]
        # A01: a_rows[0]*m + a_cols[1]
        # A10: a_rows[1]*m + a_cols[0]
        # A11: a_rows[1]*m + a_cols[1]
        
        u_indices = [
            a_rows[0]*m + a_cols[0],
            a_rows[0]*m + a_cols[1],
            a_rows[1]*m + a_cols[0],
            a_rows[1]*m + a_cols[1]
        ]
        
        # Coefficients for V (B part)
        # B entries involved: B00, B01, B10, B11
        # Indices in V:
        # B00: b_rows[0]*p + b_cols[0]
        # B01: b_rows[0]*p + b_cols[1]
        # B10: b_rows[1]*p + b_cols[0]
        # B11: b_rows[1]*p + b_cols[1]
        
        v_indices = [
            b_rows[0]*p + b_cols[0],
            b_rows[0]*p + b_cols[1],
            b_rows[1]*p + b_cols[0],
            b_rows[1]*p + b_cols[1]
        ]
        
        # Coefficients for W (C part)
        # C entries involved: C00, C01, C10, C11
        # Indices in W:
        # C00: c_cols[0]*n + c_rows[0]
        # C01: c_cols[1]*n + c_rows[0]
        # C10: c_cols[0]*n + c_rows[1]
        # C11: c_cols[1]*n + c_rows[1]
        
        w_indices = [
            c_cols[0]*n + c_rows[0],
            c_cols[1]*n + c_rows[0],
            c_cols[0]*n + c_rows[1],
            c_cols[1]*n + c_rows[1]
        ]
        
        # Define the 7 terms
        # term: (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs is vector of size 4 for [A00, A01, A10, A11]
        # v_coeffs is vector of size 4 for [B00, B01, B10, B11]
        # w_coeffs is vector of size 4 for [C00, C01, C10, C11]
        
        terms = [
            # M1
            (
                [1, 0, 0, 1],   # A00 + A11
                [1, 0, 0, 1],   # B00 + B11
                [1, 0, 0, -1]   # C00=1, C01=0, C10=0, C11=-1 (Wait, C11 coeff is -1? No. C11 = M1 - M2 + M3 + M6. So coeff for M1 in C11 is +1. Let's re-verify.)
            ),
            # M2
            (
                [0, 0, 1, 1],   # A10 + A11
                [1, 0, 0, 0],   # B00
                [0, 0, 1, -1]   # C10=1, C11=-1
            ),
            # M3
            (
                [1, 0, 0, 0],   # A00
                [0, -1, 1, 0],  # B10 - B11 (Wait, B10 is index 2, B11 is index 3. So [0,0,1,-1]?)
                # Let's check indices: B00(0), B01(1), B10(2), B11(3).
                # M3 uses B10 - B11. So coeffs [0, 0, 1, -1].
                [0, 1, 0, 1]    # C01=1, C11=1
            ),
            # M4
            (
                [0, 0, 0, 1],   # A11
                [0, 1, 0, -1],  # B01 - B00. B01(1), B00(0). So [ -1, 1, 0, 0]?
                # Let's be careful.
                # B coeffs: B00, B01, B10, B11.
                # M4 = A11 * (B01 - B00).
                # So coeffs: [-1, 1, 0, 0].
                [1, 0, 1, 0]    # C00=1, C10=1
            ),
            # M5
            (
                [1, 1, 0, 0],   # A00 + A01
                [0, 0, 0, 1],   # B11
                [-1, 1, 0, 0]   # C00=-1, C01=1
            ),
            # M6
            (
                [-1, 0, 1, 0],  # A10 - A00
                [1, 1, 0, 0],   # B00 + B01
                [0, 0, 0, 1]    # C11=1
            ),
            # M7
            (
                [0, 1, 0, -1],  # A01 - A11
                [0, 0, 1, 1],   # B10 + B11
                [1, 0, 0, 0]    # C00=1
            ),
        ]
        
        # Correcting the coefficients based on standard Strassen formulas:
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
        
        # Let's rebuild terms with correct V coefficients:
        # V indices: 0:B00, 1:B01, 2:B10, 3:B11
        
        terms = [
            # M1
            (
                [1, 0, 0, 1],   # A: A00+A11
                [1, 0, 0, 1],   # B: B00+B11
                [1, 0, 0, 0],   # W: C00 gets +1 (from M1)
            ),
            # M2
            (
                [0, 0, 1, 1],   # A: A10+A11
                [1, 0, 0, 0],   # B: B00
                [0, 0, 1, -1],  # W: C10 gets +1, C11 gets -1
            ),
            # M3
            (
                [1, 0, 0, 0],   # A: A00
                [0, 0, 1, -1],  # B: B10-B11
                [0, 1, 0, 1],   # W: C01 gets +1, C11 gets +1
            ),
            # M4
            (
                [0, 0, 0, 1],   # A: A11
                [-1, 1, 0, 0],  # B: B01-B00
                [1, 0, 1, 0],   # W: C00 gets +1, C10 gets +1
            ),
            # M5
            (
                [1, 1, 0, 0],   # A: A00+A01
                [0, 0, 0, 1],   # B: B11
                [-1, 1, 0, 0],  # W: C00 gets -1, C01 gets +1
            ),
            # M6
            (
                [-1, 0, 1, 0],  # A: A10-A00
                [1, 1, 0, 0],   # B: B00+B01
                [0, 0, 0, 1],   # W: C11 gets +1
            ),
            # M7
            (
                [0, 1, 0, -1],  # A: A01-A11
                [0, 0, 1, 1],   # B: B10+B11
                [1, 0, 0, 0],   # W: C00 gets +1
            ),
        ]
        
        # Now, for each of the 7 multiplications, we add one rank-1 component
        for t in range(7):
            r_idx = current_rank + t
            
            u_c, v_c, w_c = terms[t]
            
            # Add to U
            for i in range(4):
                if u_c[i] != 0:
                    U[u_indices[i], r_idx] += u_c[i]
            
            # Add to V
            for i in range(4):
                if v_c[i] != 0:
                    V[v_indices[i], r_idx] += v_c[i]
            
            # Add to W
            for i in range(4):
                if w_c[i] != 0:
                    W[w_indices[i], r_idx] += w_c[i]
        
        current_rank += 7

    # Helper to add a Schoolbook block for A (2x2) * B (2x1) -> C (2x1)
    # a_rows, a_cols: 2x2 block in A
    # b_rows: 2 rows in B, b_col: 1 col in B
    # c_rows: 2 rows in C, c_col: 1 col in C
    def add_schoolbook_2x2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col):
        nonlocal current_rank
        
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # 4 terms
        # T1: A00 * B0 -> C0
        # T2: A01 * B1 -> C0
        # T3: A10 * B0 -> C1
        # T4: A11 * B1 -> C1
        
        a_idx_00 = a_rows[0]*m + a_cols[0]
        a_idx_01 = a_rows[0]*m + a_cols[1]
        a_idx_10 = a_rows[1]*m + a_cols[0]
        a_idx_11 = a_rows[1]*m + a_cols[1]
        
        b_idx_0 = b_rows[0]*p + b_col
        b_idx_1 = b_rows[1]*p + b_col
        
        c_idx_0 = c_col*n + c_rows[0]
        c_idx_1 = c_col*n + c_rows[1]
        
        # T1
        r = current_rank
        U[a_idx_00, r] = 1
        V[b_idx_0, r] = 1
        W[c_idx_0, r] = 1
        current_rank += 1
        
        # T2
        r = current_rank
        U[a_idx_01, r] = 1
        V[b_idx_1, r] = 1
        W[c_idx_0, r] = 1
        current_rank += 1
        
        # T3
        r = current_rank
        U[a_idx_10, r] = 1
        V[b_idx_0, r] = 1
        W[c_idx_1, r] = 1
        current_rank += 1
        
        # T4
        r = current_rank
        U[a_idx_11, r] = 1
        V[b_idx_1, r] = 1
        W[c_idx_1, r] = 1
        current_rank += 1

    # Tiling strategy:
    # A is 2x4. Split cols into [0,1] and [2,3].
    # B is 4x5. Split rows into [0,1] and [2,3].
    # C is 2x5. Split cols into [0,1], [2,3], [4].
    
    # We have 4 combinations of A-block and B-block.
    # A-block 0 (cols 0,1), A-block 1 (cols 2,3)
    # B-block 0 (rows 0,1), B-block 1 (rows 2,3)
    
    # For each (A_block, B_block) pair, we compute C contributions.
    # C has 5 columns. We can group C columns into pairs for Strassen if possible.
    # C cols 0,1 -> Strassen
    # C cols 2,3 -> Strassen
    # C col 4 -> Schoolbook
    
    # However, Strassen requires 2x2x2.
    # A_block is 2x2. B_block is 2x2 (rows from B, cols from C).
    # So for C cols 0,1: B uses rows (from B_block) and cols (0,1).
    # For C cols 2,3: B uses rows (from B_block) and cols (2,3).
    # For C col 4: B uses rows (from B_block) and col (4).
    
    # Case 1: A_block 0 (rows 0,1; cols 0,1)
    a_rows_0 = [0, 1]
    a_cols_0 = [0, 1]
    
    # Case 2: A_block 1 (rows 0,1; cols 2,3)
    a_rows_1 = [0, 1]
    a_cols_1 = [2, 3]
    
    # B_block 0 (rows 0,1)
    b_rows_0 = [0, 1]
    
    # B_block 1 (rows 2,3)
    b_rows_1 = [2, 3]
    
    # C rows are always 0,1.
    c_rows = [0, 1]
    
    # C cols 0,1
    c_cols_01 = [0, 1]
    # C cols 2,3
    c_cols_23 = [2, 3]
    # C col 4
    c_col_4 = 4
    
    # 1. A_block 0, B_block 0
    # C cols 0,1: Strassen
    add_strassen(a_rows_0, a_cols_0, b_rows_0, c_cols_01, c_rows, c_cols_01)
    # C cols 2,3: Strassen
    add_strassen(a_rows_0, a_cols_0, b_rows_0, c_cols_23, c_rows, c_cols_23)
    # C col 4: Schoolbook
    add_schoolbook_2x2x1(a_rows_0, a_cols_0, b_rows_0, c_col_4, c_rows, c_col_4)
    
    # 2. A_block 0, B_block 1
    # C cols 0,1: Strassen
    add_strassen(a_rows_0, a_cols_0, b_rows_1, c_cols_01, c_rows, c_cols_01)
    # C cols 2,3: Strassen
    add_strassen(a_rows_0, a_cols_0, b_rows_1, c_cols_23, c_rows, c_cols_23)
    # C col 4: Schoolbook
    add_schoolbook_2x2x1(a_rows_0, a_cols_0, b_rows_1, c_col_4, c_rows, c_col_4)
    
    # 3. A_block 1, B_block 0
    # C cols 0,1: Strassen
    add_strassen(a_rows_1, a_cols_1, b_rows_0, c_cols_01, c_rows, c_cols_01)
    # C cols 2,3: Strassen
    add_strassen(a_rows_1, a_cols_1, b_rows_0, c_cols_23, c_rows, c_cols_23)
    # C col 4: Schoolbook
    add_schoolbook_2x2x1(a_rows_1, a_cols_1, b_rows_0, c_col_4, c_rows, c_col_4)
    
    # 4. A_block 1, B_block 1
    # C cols 0,1: Strassen
    add_strassen(a_rows_1, a_cols_1, b_rows_1, c_cols_01, c_rows, c_cols_01)
    # C cols 2,3: Strassen
    add_strassen(a_rows_1, a_cols_1, b_rows_1, c_cols_23, c_rows, c_cols_23)
    # C col 4: Schoolbook
    add_schoolbook_2x2x1(a_rows_1, a_cols_1, b_rows_1, c_col_4, c_rows, c_col_4)
    
    # Total rank:
    # 4 blocks * (7 + 7 + 4) = 4 * 18 = 72.
    # Wait, my previous calculation was 36.
    # Let's re-evaluate.
    # In the previous thought process, I said:
    # Block 1.1: 7
    # Block 1.2: 7
    # Block 1.3: 4
    # Block 2.1: 7
    # Block 2.2: 7
    # Block 2.3: 4
    # Total 36.
    # This corresponds to only 2 A-blocks and 2 B-blocks?
    # No, 2 A-blocks and 2 B-blocks means 4 combinations.
    # Each combination has 3 sub-blocks (2 Strassen, 1 Schoolbook).
    # So 4 * 3 = 12 sub-blocks.
    # 8 Strassen blocks * 7 = 56.
    # 4 Schoolbook blocks * 4 = 16.
    # Total 72.
    
    # Why did I think 36 before?
    # Maybe I was assuming a different tiling?
    # If I only use Strassen for the full 4x4 part of B?
    # B is 4x5.
    # If I split B into 4x4 and 4x1.
    # 4x4 part:
    # A is 2x4.
    # We can do 2x2 Strassen on A and 2x2 Strassen on B?
    # But Strassen is for 2x2 matrix multiplication.
    # If we treat A as two 2x2 blocks and B as two 2x2 blocks...
    # Actually, the standard way to get rank 36 for 2x4x5 might be different.
    
    # Let's check if rank 36 is even possible.
    # Rank of 2x4x5 matmul tensor.
    # Schoolbook: 2*4*5 = 40.
    # Strassen on 2x2 parts:
    # If we could do it in 36, that would be good.
    # But 72 is valid.
    # Can we do better than 40?
    # Yes, 72 is worse than 40.
    # So my current implementation (72) is WORSE than schoolbook (40).
    # The score is rank (lower is better).
    # So I should just return Schoolbook rank 40 if I can't find better.
    # Or maybe I can find a better decomposition.
    
    # Let's try to construct a rank 40 schoolbook decomposition as a fallback.
    # And maybe try to optimize.
    
    # But wait, the previous program had score 0.0000.
    # The inspirations had score 1.1111.
    # 1.1111 * 40 = 44.44? No.
    # Score is seed/value.
    # If seed is 40 (schoolbook), and value is 36, score = 40/36 = 1.11.
    # So a valid rank 36 would score 1.11.
    # A valid rank 40 would score 1.00.
    # My current 72 ranks would score 40/72 = 0.55.
    
    # So I MUST find a decomposition with rank <= 40.
    # Is rank 36 possible?
    # The previous program claimed rank 36 but failed validation.
    # Let's look at the previous program's tiling.
    # It used:
    # Block 1: A cols 0,1; B rows 0,1. (7+7+4 = 18)
    # Block 2: A cols 2,3; B rows 2,3. (7+7+4 = 18)
    # Total 36.
    # This covers:
    # A[0:2, 0:2] * B[0:2, :] -> C[:, 0:2] and C[:, 4]?
    # No, B rows 0,1.
    # So it computes C = A[:, 0:2] * B[0:2, :].
    # This gives C[:, :] using only B rows 0,1.
    # But B has rows 0,1,2,3.
    # So this is only half the sum!
    # The other half is A[:, 2:4] * B[2:4, :].
    # So the previous program's logic was flawed because it didn't sum the contributions from all B rows for each C element?
    # Wait.
    # C = A * B = sum_k A[:, k] * B[k, :].
    # The tensor decomposition T = sum_r u_r v_r w_r.
    # Each term contributes to all C[i, j] where u_r has i, v_r has j... no.
    # U[i, r], V[j, r], W[k, r].
    # C[i, k] = sum_j sum_r A[i, j] U[i, r] V[j, r] W[k, r]?
    # No.
    # The standard form is:
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T_{i,j,k} = delta? No.
    # The multiplication tensor T is defined by T_{i,j,k} such that:
    # sum_{j} A_{i,j} B_{j,k} = C_{i,k}.
    # This is not a simple outer product sum.
    # The rank of the multiplication tensor is the minimal number of rank-1 tensors needed to sum to it.
    # A rank-1 tensor is U \otimes V \otimes W.
    # (U \otimes V \otimes W)_{i,j,k} = U_i V_j W_k.
    # Sum_r U_r(i) V_r(j) W_r(k) = C_{i,k} / B_{j,k}? No.
    # The identity is:
    # C_{i,k} = sum_j sum_r U_{i,r} A_{i,j}? No.
    
    # Let's stick to the definition:
    # We want to find U, V, W such that for all i, j, k:
    # sum_r U_{i,r} V_{j,r} W_{k,r} = [j is valid index?]
    # Actually, the tensor for matrix multiplication A(n x m) * B(m x p) -> C(n x p)
    # is T in R^{n x m x p}.
    # T_{i,j,k} is 1 if... no.
    # The condition is:
    # sum_{j} A_{i,j} B_{j,k} = C_{i,k}.
    # We can rewrite this as:
    # C_{i,k} = sum_{j} sum_{r} (A_{i,j} U_{j,r}) (B_{j,k} V_{r,j}) W_{i,r} ? No.
    
    # Standard tensor decomposition for matmul:
    # C_{i,k} = sum_r M_r(i, k)
    # where M_r is a rank-1 matrix? No.
    
    # The rank of the multiplication tensor T_{n,m,p} is the minimal rank of the bilinear map.
    # It is the minimal R such that:
    # C_{i,k} = sum_{j=1}^m sum_{r=1}^R A_{i,j} B_{j,k} \cdot \text{coeff}?
    # No.
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # This is a trilinear form? No, it's bilinear in A and B.
    # But we treat A, B, C as tensors.
    # The tensor T is defined by T_{i,j,k} = 1 if ...?
    # Actually, T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, the tensor T has T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # But j is summed over.
    # So T is an element of R^{n} \otimes R^{m} \otimes R^{p}.
    # T = sum_{j} E_{i,j} \otimes E_{j,j} \otimes E_{j,k}?
    # No.
    # T = sum_{i,j,k} T_{i,j,k} E_i \otimes E_j \otimes E_k.
    # The condition is that for any A, B:
    # C = A * B.
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # So T_{i,j,k} is not simply 1.
    # The tensor decomposition means:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # Then <T, A \otimes B> = sum_r <u_r, A> <v_r, B> <w_r, C>?
    # No.
    # <T, A \otimes B \otimes C> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}?
    
    # Let's use the explicit algorithm I wrote.
    # I mapped A indices to U, B indices to V, C indices to W.
    # U[i*m + j, r].
    # V[j*p + k, r].
    # W[k*n + i, r].
    # And I filled them such that:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = ?
    
    # If I use the Strassen block logic:
    # For a 2x2x2 block, I added 7 terms.
    # Each term r had:
    # U[i, r] for i in {00, 01, 10, 11} of A.
    # V[j, r] for j in {00, 01, 10, 11} of B.
    # W[k, r] for k in {00, 01, 10, 11} of C.
    
    # Does this correctly decompose the 2x2x2 matmul tensor?
    # The 2x2x2 matmul tensor T_{2,2,2} has rank 7.
    # The Strassen algorithm provides a rank 7 decomposition.
    # So the block logic is correct for a 2x2x2 sub-problem.
    
    # The issue is how these blocks combine.
    # The full problem is 2x4x5.
    # If I tile it into independent 2x2x2 blocks, they are NOT independent.
    # They share indices.
    # For example, C[0,0] receives contributions from A[0,0]*B[0,0], A[0,1]*B[1,0], A[0,2]*B[2,0], A[0,3]*B[3,0].
    # In my tiling:
    # Block (A0, B0) handles A[0:2, 0:2] and B[0:2, 0:2].
    # It produces C[0:2, 0:2].
    # Specifically, it produces the correct C[0:2, 0:2] from that part of A and B.
    # Block (A0, B1) handles A[0:2, 0:2] and B[2:4, 0:2].
    # It produces C[0:2, 0:2] from that part.
    # So C[0:2, 0:2] = (Block A0,B0) + (Block A0,B1) + (Block A1,B0) + (Block A1,B1)?
    # No, A is 2x4.
    # A[0:2, 0:2] is top-left 2x2.
    # A[0:2, 2:4] is top-right 2x2.
    # B[0:2, :] is top 2x5.
    # B[2:4, :] is bottom 2x5.
    
    # C = A * B.
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2].
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4].
    # C[:, 4:5] = A[:, 0:2] * B[0:2, 4:5] + A[:, 2:4] * B[2:4, 4:5].
    
    # So we need 8 Strassen blocks (for the 4x4 part of B) and 4 Schoolbook blocks (for the 1x1 part of B).
    # Total rank 8*7 + 4*4 = 56 + 16 = 72.
    
    # This is definitely > 40.
    
    # So, I will implement a simple Schoolbook rank 40 decomposition.
    # It is guaranteed to be valid.
    # Score will be 1.0.
    
    # But wait, can we do better?
    # The tensor rank of 2x4x5 matmul.
    # It is known that R(2,4,5) <= 40.
    # Is it < 40?
    # Lower bound for R(n,m,p) is max(nm, mp, np) = max(8, 20, 10) = 20.
    # So rank could be as low as 20.
    # But finding it is hard.
    
    # I will provide the Schoolbook rank 40 solution.
    
    def construct_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    r += 1
        
        return {"U": U, "V": V, "W": W}
    
    return construct_schoolbook()
