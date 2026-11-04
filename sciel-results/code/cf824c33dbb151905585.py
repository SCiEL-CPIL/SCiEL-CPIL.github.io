import numpy as np
import time
import sys

_START = time.time()
_HARD_LIMIT = 148.0

def _remaining():
    return _HARD_LIMIT - (time.time() - _START)

def construct():
    n, m, p = 4, 4, 7
    # Matrix multiplication tensor T: U (n*m) x rank, V (m*p) x rank, W (p*n) x rank
    # Convention: W[k*n + i, r] where k is output row, i is output col.
    # Standard: C[k,i] = sum_j A[i,j] B[j,k]
    # Here A is 4x4, B is 4x7.
    # U corresponds to A factors: index i*m + j (i row of A, j col of A)
    # V corresponds to B factors: index j*p + k (j row of B, k col of B)
    # W corresponds to C factors: index k*n + i (k col of C? No, k is output row?
    # Wait, "W[k*n+i,r]" where k*n+i.
    # Usually C is p x n? No, A is n x m, B is m x p. C is n x p.
    # The problem says "multiplying a 4 by 4 matrix by a 4 by 7 matrix".
    # So A is 4x4 (n=4, m=4), B is 4x7 (m=4, p=7).
    # C is 4x7 (n=4, p=7).
    # Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # W shape is (p*n, rank). So W index is something like k*n + i?
    # If C is n x p, indices are (i, k) where i in 0..n-1, k in 0..p-1.
    # W[k*n + i] suggests row index k (0..p-1) and col index i (0..n-1)?
    # Or is it flattened C^T?
    # Let's check the seed code's usage.
    # Seed: `W[k_final * n + i_final, r]`.
    # In seed: `i_final = 2 * br + li`, `k_final = 2 * bc + lj`.
    # `li` from x_sub (A indices), `lj` from y_sub (B indices)?
    # Let's trace Strassen 2x2.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = AB.
    # C00 = a00b00 + a01b10.
    # Seed term 0: x={(0,0):1,(1,1):1}, y={(0,0):1,(1,1):1}, c={(0,0):1,(1,1):1}
    # This is (a00+a11)(b00+b11).
    # Contribution to C00: 1*1 = 1. Contribution to C11: 1*1 = 1.
    # In W: `i_final = 0*2+0=0`, `k_final=0*2+0=0`. W[0*n+0] += 1. (C[0,0])
    # `i_final = 1*2+0=2`, `k_final=1*2+0=2`. W[2*n+2] += 1. (C[2,2] in 4x4?)
    # Wait, the indices in seed are mapped to 4x4.
    # Let's look at the 2x2x3 part in seed.
    # `i_final = 2 * cr + li`, `k_final = 4 + lj`.
    # `cr` is row block index (0 or 1). `li` is local row (0 or 1).
    # So `i_final` is the row index of C (0..3).
    # `lj` is local col (0 or 1). `k_final = 4 + lj` (cols 4, 5).
    # So W index is `k_final * n + i_final`.
    # This means W is stored in column-major order of C?
    # C is n x p. Index is k (col) * n + i (row).
    # So W[r, rank] contributes to C[i, k].
    # This is standard for tensor decomposition if we view C^T or just flatten column-wise.
    # Let's verify U and V.
    # U index: `i * m + j`. i is row of A, j is col of A.
    # V index: `i * p + j`. i is row of B (which is col of A), j is col of B.
    # So U[i*m+j], V[j*p+k], W[k*n+i].
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = C[i,k].
    # This is correct.

    # Strategy:
    # The current rank is 93.
    # 49 (Strassen 4x4x4) + 44 (4 * 2x2x3, each 11).
    # The 2x2x3 decomposition in the seed uses:
    # - Strassen for 2x2x2 part (rank 7)
    # - Schoolbook for the 3rd column (rank 4)
    # Total 11.
    # Can we do better for 2x2x3?
    # Rank of 2x2x3 matrix multiplication is 8?
    # No, rank of 2x2x2 is 7 (Strassen).
    # For 2x2x3, we compute C[0,2] and C[1,2] as well.
    # C[0,2] = a00 b02 + a01 b12
    # C[1,2] = a10 b02 + a11 b12
    # This is just two dot products.
    # If we use Strassen for the first two columns (rank 7), we get C[0,0], C[0,1], C[1,0], C[1,1].
    # Then we need C[0,2] and C[1,2].
    # These depend on b02, b12.
    # We can add terms:
    # 1. a00 b02 -> C[0,2]
    # 2. a01 b12 -> C[0,2]
    # 3. a10 b02 -> C[1,2]
    # 4. a11 b12 -> C[1,2]
    # This is rank 4. Total 11.
    # Is it possible to do better?
    # Maybe combine some terms?
    # Consider (a00+a01)(b02+b12) = a00b02 + a00b12 + a01b02 + a01b12.
    # We have a01b12 in the sum. We don't want a00b12 or a01b02.
    # If we could subtract them... but we don't have terms for a00b12 etc. in the Strassen part.
    # So we probably can't reduce below 11 for 2x2x3 easily without a clever trick.
    # However, the 4x4x4 part is Strassen (49).
    # Can we do 4x4x7 better?
    # Maybe split 7 into 4+3.
    # 4x4x4 (Strassen, rank 49) + 4x4x3 (Schoolbook, rank 48). Total 97.
    # Seed does 49 + 4*11 = 93.
    # Is there a better decomposition for 4x4x3?
    # 4x4x3 tensor rank?
    # It's 4 blocks of 1x4x3? No, 2x2x3 blocks.
    # We treat it as 4 independent 2x2x3 subproblems?
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2).
    # B = [[B00, B01], [B10, B11]] (2x2 blocks of 2x2x3? No, B is 4x7.
    # Split B into B_left (4x4) and B_right (4x3).
    # B_left is 4x4. B_right is 4x3.
    # We compute C_left = A B_left (4x4) and C_right = A B_right (4x3).
    # C_left uses Strassen 4x4x4 (rank 49).
    # C_right is 4x4x3.
    # We decompose C_right into 4 sub-problems?
    # A is 4x4. B_right is 4x3.
    # Row split A into 2x2 blocks.
    # C_right[0:2, :] = A[0:2, :] B_right.
    # A[0:2, :] is 2x4. B_right is 4x3.
    # This is 2x4x3.
    # Split A[0:2, :] into [A00, A01] (2x2 each).
    # B_right = [B0; B1] (2x3 each, rows 0-1 and 2-3).
    # C[0:2, :] = A00 B0 + A01 B1.
    # A00, A01 are 2x2. B0, B1 are 2x3.
    # This is 2x2x3 (A00 B0) + 2x2x3 (A01 B1).
    # Rank of 2x2x3 is at least 8? (2x2x2 is 7, plus 2 terms for col 2? No, 2x2x2 is 7, 2x2x1 is 4? No, 2x2x1 is 4 (rank 2? No, outer product is rank 1? No, 2x2 matrix product is rank 7. 2x2x1 is 2x2 product with 1 col. That's just a linear map R^4 -> R^2. Rank 2? No, it's a tensor.
    # Let's check rank of 2x2x1.
    # T[i,j,k] = A[i,j] B[j,k]. k=0.
    # T[i,j,0] = A[i,j] B[j,0].
    # This is just A times vector b.
    # Can we do it in rank 2?
    # (a00+a01)(b00+b10) - ... no.
    # Actually, 2x2x1 tensor is equivalent to matrix multiplication 2x2 by 2x1.
    # The rank of this tensor is 4?
    # Terms: a00b00, a01b10, a10b00, a11b10.
    # Can we combine?
    # (a00+a10)(b00) + (a01+a11)(b10) -> 2 terms?
    # This gives a00b00 + a10b00 + a01b10 + a11b10.
    # Yes! Rank is 2 for 2x2x1.
    # So 2x2x3 = 2x2x2 (rank 7) + 2x2x1 (rank 2)?
    # Total 9?
    # Let's check.
    # C[0,2] = a00 b02 + a01 b12.
    # C[1,2] = a10 b02 + a11 b12.
    # Using rank 2 decomposition for col 2:
    # Term 5: (a00+a10) b02 -> contributes to C[0,2] (via a00b02) and C[1,2] (via a10b02).
    # Wait, W factor for C[0,2] is 1, for C[1,2] is 1.
    # So U factor: a00+a10. V factor: b02. W factor: (1, 1) for rows 0,1?
    # No, W is for specific output entries.
    # If we use term: U = a00+a10, V = b02, W[0,2]=1, W[1,2]=1.
    # This produces (a00+a10)b02 in C[0,2] and C[1,2].
    # We need a01b12 and a11b12.
    # Term 6: U = a01+a11, V = b12, W[0,2]=1, W[1,2]=1.
    # This produces (a01+a11)b12 in C[0,2] and C[1,2].
    # Total for col 2: 2 terms.
    # So 2x2x3 rank is 7 + 2 = 9?
    # Let's verify.
    # C[0,2] = a00b02 + a01b12.
    # From Term 5: (a00+a10)b02 -> a00b02. (a10b02 is extra? No, we want a10b02 in C[1,2]?)
    # C[1,2] = a10b02 + a11b12.
    # Term 5 contributes a10b02 to C[1,2]. Correct.
    # Term 6 contributes a01b12 to C[0,2]. Correct.
    # Term 6 contributes a11b12 to C[1,2]. Correct.
    # So yes, rank 2 for the 3rd column!
    # Total rank for 2x2x3 is 9.
    # In the seed, it used 4 terms for the 3rd column (schoolbook).
    # If we use rank 2, we save 2 terms per 2x2x3 block.
    # There are 4 such blocks.
    # Savings: 4 * 2 = 8 terms.
    # New rank: 93 - 8 = 85.

    # Let's construct the rank-9 decomposition for 2x2x3.
    # Blocks:
    # 0-6: Strassen 2x2x2 for cols 0,1.
    # 7: (a00+a10) * b02 -> W[0,2]=1, W[1,2]=1.
    # 8: (a01+a11) * b12 -> W[0,2]=1, W[1,2]=1.
    
    # We need to map these to the global indices.
    # The 4 blocks are:
    # (ar, ac) in {0,1} x {0,1}.
    # Block (ar, ac) corresponds to A submatrix rows 2*ar..2*ar+1, cols 2*ac..2*ac+1.
    # B submatrix rows 2*ac..2*ac+1? No.
    # In the seed:
    # add_2x2x3(0, 0, 0, 0) -> A rows 0-1, B rows 0-1?
    # Let's check seed's V index: `V[i * p + j, r]`.
    # `i = 2 * br + li`. `br` is B row block.
    # `j = 4 + lj`.
    # In seed, `add_2x2x3(ar, ac, br, cr)`.
    # U uses `ar, ac`.
    # V uses `br`.
    # W uses `cr` (row), `lj` (col).
    # The mapping of blocks to the 4x4x3 subproblem:
    # C_right is 4x3.
    # Rows 0-1 come from A rows 0-1.
    # C_right[0:2, :] = A[0:2, 0:4] * B_right[0:4, :].
    # A[0:2, 0:4] = [A00, A01] where A00 is 2x2 (rows 0-1, cols 0-1), A01 is 2x2 (rows 0-1, cols 2-3).
    # B_right[0:4, :] = [B0; B1] where B0 is 2x3 (rows 0-1), B1 is 2x3 (rows 2-3).
    # So C_right[0:2, :] = A00 B0 + A01 B1.
    # This is sum of two 2x2x3 products.
    # Seed calls:
    # add_2x2x3(0, 0, 0, 0) -> A00 * B0. (ar=0, ac=0, br=0, cr=0)
    # add_2x2x3(0, 1, 1, 0) -> A01 * B1. (ar=0, ac=1, br=1, cr=0)
    # Similarly for rows 2-3.
    # add_2x2x3(1, 0, 0, 1) -> A10 * B0.
    # add_2x2x3(1, 1, 1, 1) -> A11 * B1.
    
    # So for each block, we replace the last 4 terms with 2 terms.
    
    total_rank = 49 + 4 * 9 # 49 + 36 = 85
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2 definitions
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49)
    for m_idx in range(4): # Only 4 cols in first part?
        # Wait, seed did `for m_idx in range(7)`.
        # Ah, the seed treated the whole 4x4x7 as 7 columns of 4x4x1?
        # No, seed used `strassen_4x4_outer` which has 7 entries.
        # And `c_blocks_4x4` which has 7 entries.
        # This suggests the seed used Strassen for 4x4x4 on the FIRST 4 columns?
        # But `strassen_4x4_outer` has 7 entries. Strassen 4x4 uses 7 terms.
        # So it computed the 4x4x4 product (cols 0-3) using rank 49.
        # Then cols 4-6 using 2x2x3 blocks.
        # Let's re-read seed.
        # `for m_idx in range(7):`
        # `x_def, y_def = strassen_4x4_outer[m_idx]`
        # This computes the 4x4x4 product for the first 4 columns?
        # No, `x_def` and `y_def` define which 2x2 blocks of A and B are involved.
        # For 4x4x4, we only care about B cols 0-3.
        # The B matrix in the 4x4x4 part is 4x4.
        # So this part computes C[:, 0:4].
        # Yes.
        pass
        
    # I will copy the Part 1 logic from seed exactly, as it works for cols 0-3.
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c
    
    # Part 2: 4x4x3 (cols 4, 5, 6)
    # 4 blocks of 2x2x3, each rank 9.
    def add_2x2x3_optimized(ar, ac, br, cr):
        nonlocal current_rank
        
        # Strassen 2x2x2 for cols 4, 5 (local 0, 1)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj
                V[i * p + j, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj
                W[k_final * n + i_final, r] += cc
        
        # Optimized terms for col 6 (local col 2)
        # Term 7: (a00+a10) * b02
        r = current_rank
        current_rank += 1
        # U: a00 and a10
        U[(2 * ar + 0) * m + (2 * ac + 0), r] = 1.0
        U[(2 * ar + 1) * m + (2 * ac + 0), r] = 1.0
        # V: b02 (row 2*br, col 6)
        V[(2 * br + 0) * p + 6, r] = 1.0
        # W: C[0,2] and C[1,2] -> rows 2*cr, 2*cr+1, col 6
        W[6 * n + (2 * cr + 0), r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0
        
        # Term 8: (a01+a11) * b12
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 1), r] = 1.0
        U[(2 * ar + 1) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0

    add_2x2x3_optimized(0, 0, 0, 0)
    add_2x2x3_optimized(0, 1, 1, 0)
    add_2x2x3_optimized(1, 0, 0, 1)
    add_2x2x3_optimized(1, 1, 1, 1)
    
    assert current_rank == 85, f"Expected 85, got {current_rank}"
    
    U = U[:, :current_rank].tolist()
    V = V[:, :current_rank].tolist()
    W = W[:, :current_rank].tolist()
    
    return {"U": U, "V": V, "W": W}
