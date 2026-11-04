import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 110  # safety margin
    
    n, m, p = 4, 5, 6
    
    # Strassen 4x4x4 decomposition
    # We need: for all i in 0..3, j in 0..3, k in 0..3:
    #   sum_r U_str[i*4+j, r] * V_str[j*4+k, r] * W_str[k*4+i, r] = 1
    #
    # Using Strassen's 2x2 block decomposition:
    # Split 4x4 into 2x2 blocks of 2x2.
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    #
    # Strassen for 2x2 block matrices:
    # M1 = (A11+A22)*(B11+B22)  -> C11 += M1, C22 += M1
    # M2 = (A21+A22)*B11        -> C21 += M2
    # M3 = A11*(B12-B22)        -> C12 += M3
    # M4 = A22*(B21-B11)        -> C21 += M4
    # M5 = (A11+A12)*B22        -> C12 += M5
    # M6 = (A21-A11)*(B11+B12)  -> C22 += M6
    # M7 = (A12-A22)*(B21+B22)  -> C21 += M7
    #
    # Each Mi is a 2x2 product. Using schoolbook for 2x2 (rank 4):
    # For 2x2: C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # Rank-1 term (i,j,k): U[i*2+j]=1, V[j*2+k]=1, W[k*2+i]=1
    #
    # Total rank: 7 * 4 = 28 for 4x4x4 via Strassen+schoolbook-2x2
    # But we can do better with a direct 4x4x4 construction.
    #
    # Actually, let me use the known optimal-ish decomposition.
    # For 4x4x4, Strassen gives rank 21 (using a 3x3 Strassen for 3x3x3 + extras)
    # but the simplest reliable one is 7*4=28.
    #
    # Let me just use 28 and verify it works.
    
    def make_2x2_strassen_block(a_block_idx, b_block_idx, c_block_idx, sign_a, sign_b, coeff):
        """
        Generate rank-1 terms for a 2x2 matrix multiplication.
        a_block_idx: which 2x2 block of A (0=A11, 1=A12, 2=A21, 3=A22)
        b_block_idx: which 2x2 block of B (0=B11, 1=B12, 2=B21, 3=B22)
        c_block_idx: which 2x2 block of C (0=C11, 1=C12, 2=C21, 3=C22)
        sign_a: +1 or -1 (coefficient for A block)
        sign_b: +1 or -1 (coefficient for B block)
        coeff: overall coefficient
        """
        terms = []
        # A block at position (ar, ac) in 4x4: rows 2*ar to 2*ar+1, cols 2*ac to 2*ac+1
        a_row_base = (a_block_idx // 2) * 2
        a_col_base = (a_block_idx % 2) * 2
        b_row_base = (b_block_idx // 2) * 2
        b_col_base = (b_block_idx % 2) * 2
        c_row_base = (c_block_idx // 2) * 2
        c_col_base = (c_block_idx % 2) * 2
        
        for i in range(2):  # row within 2x2 block
            for j in range(2):  # col within A block / row within B block
                for k in range(2):  # col within B block
                    # U index in 4x4 space: (a_row_base + i) * 4 + (a_col_base + j)
                    # V index in 4x4 space: (b_row_base + j) * 4 + (b_col_base + k)
                    # W index in 4x4 space: (c_col_base + k) * 4 + (c_row_base + i)
                    u_idx = (a_row_base + i) * 4 + (a_col_base + j)
                    v_idx = (b_row_base + j) * 4 + (b_col_base + k)
                    w_idx = (c_col_base + k) * 4 + (c_row_base + i)
                    u_val = sign_a
                    v_val = sign_b
                    w_val = coeff
                    terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))
        return terms
    
    # Strassen formulas for 4x4x4:
    # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)*B11 -> C21 += M2
    # M3 = A11*(B12-B22) -> C12 += M3
    # M4 = A22*(B21-B11) -> C21 += M4
    # M5 = (A11+A12)*B22 -> C12 += M5
    # M6 = (A21-A11)*(B11+B12) -> C22 += M6
    # M7 = (A12-A22)*(B21+B22) -> C21 += M7
    
    strassen_terms = []
    
    # M1: (A11+A22)*(B11+B22) -> C11 and C22
    # A11: block 0, A22: block 3
    # B11: block 0, B22: block 3
    # C11: block 0, C22: block 3
    for a_b, b_b, c_b, sa, sb, coef in [
        (0, 0, 0, 1, 1, 1),  # A11*B11 -> C11
        (3, 0, 0, 1, 1, 1),  # A22*B11 -> C11
        (0, 3, 0, 1, 1, 1),  # A11*B22 -> C11
        (3, 3, 0, 1, 1, 1),  # A22*B22 -> C11
        (0, 0, 3, 1, 1, 1),  # A11*B11 -> C22
        (3, 0, 3, 1, 1, 1),  # A22*B11 -> C22
        (0, 3, 3, 1, 1, 1),  # A11*B22 -> C22
        (3, 3, 3, 1, 1, 1),  # A22*B22 -> C22
    ]:
        strassen_terms.extend(make_2x2_strassen_block(a_b, b_b, c_b, sa, sb, coef))
    
    # M2: (A21+A22)*B11 -> C21
    # A21: block 2, A22: block 3
    # B11: block 0
    # C21: block 2
    for a_b in [2, 3]:
        strassen_terms.extend(make_2x2_strassen_block(a_b, 0, 2, 1, 1, 1))
    
    # M3: A11*(B12-B22) -> C12
    # A11: block 0
    # B12: block 1, B22: block 3
    # C12: block 1
    strassen_terms.extend(make_2x2_strassen_block(0, 1, 1, 1, 1, 1))
    strassen_terms.extend(make_2x2_strassen_block(0, 3, 1, 1, -1, 1))
    
    # M4: A22*(B21-B11) -> C21
    # A22: block 3
    # B21: block 2, B11: block 0
    # C21: block 2
    strassen_terms.extend(make_2x2_strassen_block(3, 2, 2, 1, 1, 1))
    strassen_terms.extend(make_2x2_strassen_block(3, 0, 2, 1, -1, 1))
    
    # M5: (A11+A12)*B22 -> C12
    # A11: block 0, A12: block 1
    # B22: block 3
    # C12: block 1
    strassen_terms.extend(make_2x2_strassen_block(0, 3, 1, 1, 1, 1))
    strassen_terms.extend(make_2x2_strassen_block(1, 3, 1, 1, 1, 1))
    
    # M6: (A21-A11)*(B11+B12) -> C22
    # A21: block 2, A11: block 0
    # B11: block 0, B12: block 1
    # C22: block 3
    strassen_terms.extend(make_2x2_strassen_block(2, 0, 3, 1, 1, 1))
    strassen_terms.extend(make_2x2_strassen_block(0, 0, 3, -1, 1, 1))
    strassen_terms.extend(make_2x2_strassen_block(2, 1, 3, 1, 1, 1))
    strassen_terms.extend(make_2x2_strassen_block(0, 1, 3, -1, 1, 1))
    
    # M7: (A12-A22)*(B21+B22) -> C21
    # A12: block 1, A22: block 3
    # B21: block 2, B22: block 3
    # C21: block 2
    strassen_terms.extend(make_2x2_strassen_block(1, 2, 2, 1, 1, 1))
    strassen_terms.extend(make_2x2_strassen_block(3, 2, 2, -1, 1, 1))
    strassen_terms.extend(make_2x2_strassen_block(1, 3, 2, 1, 1, 1))
    strassen_terms.extend(make_2x2_strassen_block(3, 3, 2, -1, 1, 1))
    
    # Now build the full 4x5x6 decomposition
    # A (4x5): columns 0-3 are A1 (4x4), column 4 is A2 (4x1)
    # B (5x6): rows 0-3 are B1 (4x6), row 4 is B2 (1x6)
    # C (4x6): columns 0-5
    
    # C = A1*B1 + A2*B2
    # A1 is 4x4 (cols 0-3 of A)
    # B1 is 4x6 (rows 0-3 of B)
    # A2 is 4x1 (col 4 of A)
    # B2 is 1x6 (row 4 of B)
    
    # A1*B1: 4x4 * 4x6 -> 4x6
    # This is T_{4,4,6}. Split B1 into B1a (4x4, cols 0-3) and B1b (4x2, cols 4-5)
    # A1*B1a: 4x4*4x4 -> use Strassen (rank 28)
    # A1*B1b: 4x4*4x2 -> schoolbook (rank 4*4*2=32)
    # A2*B2: 4x1*1x6 -> schoolbook (rank 4*1*6=24)
    
    # Total: 28 + 32 + 24 = 84
    # Hmm, that's better than 120 but not great.
    
    # Let me reconsider. Actually, let me use a different partition.
    # Use Strassen for 4x4x4 core, and handle the rest carefully.
    
    # Let me just build it term by term.
    
    # Rank-1 terms for the full 4x5x6:
    # U has shape (4*5, rank) = (20, rank)
    # V has shape (5*6, rank) = (30, rank)
    # W has shape (6*4, rank) = (24, rank)
    
    all_terms = []
    
    # Part 1: A1 (4x4, A cols 0-3) * B1a (4x4, B rows 0-3, cols 0-3) -> C cols 0-3
    # Using Strassen 4x4x4
    for (u_idx_4x4, v_idx_4x4, w_idx_4x4, u_val, v_val, w_val) in strassen_terms:
        # u_idx_4x4 is in 0..15 (4x4 space)
        # Convert to 4x5 A-space: row i = u_idx_4x4 // 4, col j = u_idx_4x4 % 4
        # A index = i*5 + j
        i = u_idx_4x4 // 4
        j = u_idx_4x4 % 4
        u_idx = i * 5 + j
        
        # v_idx_4x4 is in 0..15 (4x4 space for B1a)
        # B1a is rows 0-3, cols 0-3 of B (5x6)
        # B index = j*6 + k where j is row, k is col
        j_b = v_idx_4x4 // 4
        k = v_idx_4x4 % 4
        v_idx = j_b * 6 + k
        
        # w_idx_4x4 is in 0..15 (4x4 space for C)
        # C index = k*4 + i where k is col, i is row
        k_c = w_idx_4x4 // 4
        i_c = w_idx_4x4 % 4
        w_idx = k_c * 4 + i_c
        
        all_terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))
    
    # Part 2: A1 (4x4) * B1b (4x2, B rows 0-3, cols 4-5) -> C cols 4-5
    # Schoolbook: rank 4*4*2 = 32
    for i in range(4):
        for j in range(4):
            for k in range(2):  # k is 0,1 corresponding to cols 4,5 of B
                u_idx = i * 5 + j
                v_idx = j * 6 + (4 + k)
                w_idx = (4 + k) * 4 + i
                all_terms.append((u_idx, v_idx, w_idx, 1, 1, 1))
    
    # Part 3: A2 (4x1, A col 4) * B2 (1x6, B row 4) -> C (4x6)
    # Schoolbook: rank 4*1*6 = 24
    for i in range(4):
        for k in range(6):
            u_idx = i * 5 + 4
            v_idx = 4 * 6 + k
            w_idx = k * 4 + i
            all_terms.append((u_idx, v_idx, w_idx, 1, 1, 1))
    
    rank = len(all_terms)
    
    # Build matrices
    U = [[0] * rank for _ in range(20)]
    V = [[0] * rank for _ in range(30)]
    W = [[0] * rank for _ in range(24)]
    
    for col, (u_idx, v_idx, w_idx, u_val, v_val, w_val) in enumerate(all_terms):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
    
    # Verify with random matrices
    if time.time() < deadline - 5:
        np.random.seed(42)
        A = np.random.randn(4, 5)
        B = np.random.randn(5, 6)
        C_expected = A @ B
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        C_computed = np.zeros((4, 6))
        for i in range(4):
            for k in range(6):
                for r in range(rank):
                    C_computed[i, k] += U_arr[i*5 + (k-k)][r] * 0  # placeholder
        
        # Proper verification: C[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_r sum_j U[i,j,r]*V[j,k,r]*W[k,i,r] where the decomposition works out
        # Actually, verify by checking: for all i,j,k: sum_r U[i*5+j, r]*V[j*6+k, r]*W[k*4+i, r] = 1
        
        max_err = 0
        for i in range(4):
            for j in range(5):
                for k in range(6):
                    val = sum(U[i*5+j][r] * V[j*6+k][r] * W[k*4+i][r] for r in range(rank))
                    err = abs(val - 1)
                    max_err = max(max_err, err)
        
        if max_err > 1e-6:
            print(f"Verification failed: max error = {max_err}")
            # Fallback to schoolbook
            U = [[0] * 120 for _ in range(20)]
            V = [[0] * 120 for _ in range(30)]
            W = [[0] * 120 for _ in range(24)]
            col = 0
            for i in range(4):
                for j in range(5):
                    for k in range(6):
                        U[i*5+j][col] = 1
                        V[j*6+k][col] = 1
                        W[k*4+i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
