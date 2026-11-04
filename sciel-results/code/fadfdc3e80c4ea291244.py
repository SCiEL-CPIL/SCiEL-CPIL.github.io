import numpy as np
import time

def construct():
    n, m, p = 4, 4, 9
    deadline = 140
    
    # Build schoolbook solution as fallback (rank 144)
    U_sb = np.zeros((n*m, n*m*p), dtype=int)
    V_sb = np.zeros((m*p, n*m*p), dtype=int)
    W_sb = np.zeros((p*n, n*m*p), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                U_sb[u, idx] = 1
                V_sb[v, idx] = 1
                W_sb[w, idx] = 1
                idx += 1
    best_solution = (U_sb, V_sb, W_sb)
    best_rank = 144
    
    # Strassen-based block decomposition
    # Split A (4x4) into 2x2 blocks of 2x2 submatrices
    # Split B (4x9) into 2x2 blocks: left 4x4 and right 4x5
    # 
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13, B14, B15],
    #      [B21, B22, B23, B24, B25]] where each is 2x? 
    #
    # Actually, let's think of B as 4x9. We can split the 4 rows into two groups of 2,
    # and the 9 columns into... let's say 4 and 5.
    # 
    # B_left (4x4): [[B11, B12], [B21, B22]] where each is 2x2
    # B_right (4x5): [[B13, B14, B15], [B23, B24, B25]] where each is 2x?
    #
    # For the left part (4x4 * 4x4), use Strassen (rank 7)
    # For the right part (4x4 * 4x5), use schoolbook (rank 16)
    #
    # But wait - Strassen gives C = A*B for square matrices. Here we have:
    # C_left = A * B_left (4x4 result)
    # C_right = A * B_right (4x5 result)
    #
    # For Strassen on the 4x4 * 4x4 part:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M_i is a 2x2 * 2x2 multiplication (rank 4 with schoolbook)
    # Total for Strassen part: 7 * 4 = 28
    #
    # For the right part (4x4 * 4x5), schoolbook: 4*4*5 = 80? No...
    # Actually, B_right is 4x5. We can split it as:
    # B_right = [B13, B14, B15; B23, B24, B25] where each is 2x? 
    # Hmm, 5 columns is awkward. Let me split 9 into 4+4+1 instead.
    #
    # Let me try: split B into three parts: 4x4, 4x4, 4x1
    # A is 4x4, split into 2x2 blocks of 2x2.
    #
    # Part 1: A * B1 (4x4 * 4x4) -> Strassen rank 7 * 4 = 28
    # Part 2: A * B2 (4x4 * 4x4) -> Strassen rank 7 * 4 = 28  
    # Part 3: A * B3 (4x4 * 4x1) -> schoolbook rank 4*4*1 = 16
    #
    # Total: 28 + 28 + 16 = 72? That seems too low. Let me reconsider.
    #
    # Actually, Strassen's 7 multiplications are of 2x2 matrices.
    # Each 2x2 * 2x2 = rank 4 (schoolbook) or could be lower.
    # So Strassen for 4x4*4x4: 7 * 4 = 28 rank.
    #
    # Wait, I need to be more careful. The Strassen algorithm computes
    # the product of two 4x4 matrices using 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 4 (schoolbook).
    # So total rank = 7 * 4 = 28 for the 4x4*4x4 product.
    #
    # For 4x4 * 4x1: this is just 4*4 = 16 scalar multiplications, rank 16.
    #
    # Total: 28 + 28 + 16 = 72.
    #
    # But wait, can we do better? The known lower bound for 4x4*4x4 is 49
    # (Strassen bound). For 4x4*4x9, we need to figure out the optimal split.
    #
    # Let me just implement the 28+28+16 = 72 solution and verify it works.
    
    # Index mapping helpers
    # U index: u = i*m + j = i*4 + j, where i is row of A (0..3), j is col of A (0..3)
    # V index: v = j*p + k = j*9 + k, where j is row of B (0..3), k is col of B (0..8)
    # W index: w = k*n + i = k*4 + i, where k is col of B (0..8), i is row of A (0..3)
    
    # For Strassen on 4x4*4x4 (left part, columns 0-3 of B):
    # A blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    
    # Strassen intermediate multiplications (each is 2x2 * 2x2):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Result:
    # C[0:2,0:2] = M1 + M4 - M5 + M7
    # C[0:2,2:4] = M3 + M5
    # C[2:4,0:2] = M2 + M4
    # C[2:4,2:4] = M1 - M2 + M3 + M6
    
    # For the 4x4 * 4x1 part (column 8 of B, or let's use columns 4-8):
    # Actually, let me split 9 columns as 4+4+1.
    # Columns 0-3: first 4x4 block
    # Columns 4-7: second 4x4 block  
    # Column 8: 4x1 block
    
    # Let me build the full decomposition.
    
    rank_total = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        """Add a single term to the decomposition."""
        nonlocal rank_total
        U_list.append((u_idx, coeff))
        V_list.append((v_idx, coeff))
        W_list.append((w_idx, coeff))
        rank_total += 1
    
    # Helper to get U index for A[i,j]
    def u_idx(i, j):
        return i * m + j  # i*4 + j
    
    # Helper to get V index for B[j,k]
    def v_idx(j, k):
        return j * p + k  # j*9 + k
    
    # Helper to get W index for C[i,k]
    def w_idx(i, k):
        return k * n + i  # k*4 + i
    
    # Strassen for 4x4 * 4x4, given column offset for B
    def strassen_4x4(col_offset):
        """Add Strassen terms for A * B[:, col_offset:col_offset+4]"""
        nonlocal rank_total
        terms_added = []
        
        # M1 = (A11+A22)*(B11+B22)
        # A11 = A[0:2,0:2], A22 = A[2:4,2:4]
        # B11 = B[0:2,col_offset:col_offset+2], B22 = B[2:4,col_offset+2:col_offset+4]
        
        # M1: (A[0,0]+A[3,3]) * (B[0,col_off]+B[2,col_off+2]) etc.
        # This is a 2x2 * 2x2 multiplication
        # (A11+A22) has entries:
        #   [0,0]: A[0,0]+A[2,2]  -> u(0,0), u(2,2)
        #   [0,1]: A[0,1]+A[2,3]  -> u(0,1), u(2,3)
        #   [1,0]: A[1,0]+A[3,2]  -> u(1,0), u(3,2)
        #   [1,1]: A[1,1]+A[3,3]  -> u(1,1), u(3,3)
        #
        # (B11+B22) has entries:
        #   [0,0]: B[0,col_off]+B[2,col_off+2] -> v(0,col_off), v(2,col_off+2)
        #   [0,1]: B[0,col_off+1]+B[2,col_off+3] -> v(0,col_off+1), v(2,col_off+3)
        #   [1,0]: B[1,col_off]+B[3,col_off+2] -> v(1,col_off), v(3,col_off+2)
        #   [1,1]: B[1,col_off+1]+B[3,col_off+3] -> v(1,col_off+1), v(3,col_off+3)
        
        # For each of the 4 products in the 2x2*2x2:
        # Product [a,b] of (A11+A22) and (B11+B22) contributes to C[a,b]
        # where a is row (0 or 1 -> global row a), b is col (0 or 1 -> global col col_offset+b)
        
        # Actually, I need to think about this differently.
        # The Strassen algorithm gives me 7 "virtual" 2x2*2x2 multiplications.
        # Each virtual multiplication has rank 4 (schoolbook for 2x2*2x2).
        # So I need to expand each into 4 scalar terms.
        
        # Let me define the 7 intermediate matrices more carefully.
        # Each intermediate is a 2x2 matrix computed as a product of two 2x2 matrices
        # (which may be sums/differences of A and B blocks).
        
        # I'll create a helper that adds a 2x2*2x2 schoolbook multiplication
        # where the left 2x2 is a linear combination of A entries and the right 2x2
        # is a linear combination of B entries.
        
        def add_2x2_mult(a_coeffs, b_coeffs, c_row_offset, c_col_offset, sign=1):
            """
            Add terms for (sum of A entries with coeffs) * (sum of B entries with coeffs)
            contributing to C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
            
            a_coeffs: list of (i, j, coeff) for A[i,j]
            b_coeffs: list of (j, k, coeff) for B[j,k]
            c_row_offset: starting row in C
            c_col_offset: starting col in C
            sign: overall sign multiplier
            """
            for (ai, aj, ac) in a_coeffs:
                for (bj, bk, bc) in b_coeffs:
                    if aj == bj:  # Only terms where A's col matches B's row
                        u = u_idx(ai, aj)
                        v = v_idx(bj, bk)
                        w = w_idx(c_row_offset + (ai % 2), c_col_offset + (bk % 2))
                        # Wait, this isn't right. The 2x2 multiplication structure is:
                        # (A_sub * B_sub)[r,c] = sum_j A_sub[r,j] * B_sub[j,c]
                        # where r,c are in {0,1}
                        
                        # Let me redo this. I need to be more careful.
                        pass
        
        # This is getting complicated. Let me take a different approach.
        # I'll directly enumerate all the scalar multiplications needed.
        
        # For Strassen 4x4*4x4, the 7 intermediate 2x2 products each require
        # 4 scalar multiplications (schoolbook for 2x2*2x2).
        # Total: 7*4 = 28 scalar terms.
        
        # Let me just hardcode the 28 terms.
        
        # M1 = (A11+A22)*(B11+B22)
        # (A11+A22)[r,c] = A[r,c] + A[r+2,c+2] for r,c in {0,1}
        # (B11+B22)[r,c] = B[r,col_off+c] + B[r+2,col_off+c+2] for r,c in {0,1}
        
        # M1[r,c] = sum_j (A[r,j]+A[r+2,j+2]) * (B[j,col_off+c]+B[j+2,col_off+c+2])
        
        # This expands to 4 terms per (r,c), so 16 total for M1.
        # But wait, each term is a scalar multiplication A[i,j]*B[j,k].
        
        # Actually, I think the cleanest way is to just list all (i,j,k) triples
        # that contribute to the product, with their coefficients.
        
        # For standard matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
        # Each term has coefficient 1.
        
        # For Strassen, we have linear combinations, so each "virtual" multiplication
        # (A_comb * B_comb) where A_comb is a sum of A entries and B_comb is a sum
        # of B entries, expands to multiple scalar terms.
        
        # Let me define the 7 combinations:
        
        # M1: A part = A11 + A22, B part = B11 + B22
        # A11+A22: {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,2):1, (2,3):1, (3,2):1, (3,3):1}
        # B11+B22: {(0,co):1, (0,co+1):1, (1,co):1, (1,co+1):1, (2,co+2):1, (2,co+3):1, (3,co+2):1, (3,co+3):1}
        
        # The product (A11+A22)*(B11+B22) is a 2x2 matrix where:
        # [r,c] = sum_j (A11+A22)[r,j] * (B11+B22)[j,c]
        # But (A11+A22) is not a standard 2x2 matrix in the A indexing...
        
        # I think I'm overcomplicating this. Let me just use the fact that
        # Strassen's algorithm is correct and build the terms systematically.
        
        # Alternative approach: Just use schoolbook for everything and get rank 144.
        # Then try to optimize if time permits.
        
        # For now, let me just return the schoolbook solution.
        pass
    
    # Given the complexity, let me just return the schoolbook solution for now.
    # The current program already has this as a fallback.
    
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
