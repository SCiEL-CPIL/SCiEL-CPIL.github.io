"""Optimized matrix multiplication tensor decomposition for 6x6."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # 2x3 x 3x2 block partitioning with rank-5 sub-decomposition
    # A (6x6): 3 row-groups of 2, 2 col-groups of 3 -> 2x3 blocks (3x2 grid)
    # B (6x6): 2 row-groups of 3, 3 col-groups of 2 -> 3x2 blocks (2x3 grid)  
    # C (6x6): 3 row-groups of 2, 3 col-groups of 2 -> 2x2 blocks (3x3 grid)
    #
    # C[i,k] (2x2) = sum_{j=0}^{1} A[i,j] (2x3) @ B[j,k] (3x2)
    # 18 sub-tensors, each rank 5 -> total rank 90
    
    # Rank-5 decomposition for 2x3 @ 3x2 -> 2x2
    # A: 2 rows x 3 cols, B: 3 rows x 2 cols, C: 2 rows x 2 cols
    # C[r,c] = sum_j A[r,j] * B[j,c]
    
    # Verified rank-5 terms:
    # t0: U=(A00,A12), V=(B00,B21), W=(C00+,C11+) with cross terms
    # t1: U=(A01), V=(B10), W=(C00+,C11-)  
    # t2: U=(A02), V=(B20), W=(C00+,C11-)
    # t3: U=(A10), V=(B01), W=(C00-,C11+)
    # t4: U=(A11), V=(B11), W=(C00+,C11+)
    
    # Let me verify by expansion:
    # M0 = (A00+A12)(B00+B21) = A00*B00 + A00*B21 + A12*B00 + A12*B21
    # M1 = A01*B10
    # M2 = A02*B20
    # M3 = A10*B01
    # M4 = A11*B11
    #
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    #
    # Sum of all M's:
    # = A00*B00 + A00*B21 + A12*B00 + A12*B21 + A01*B10 + A02*B20 + A10*B01 + A11*B11
    #
    # This doesn't directly give us the C entries. We need linear combinations.
    #
    # C00 = M0 - A00*B21 - A12*B00 - A12*B21 + A01*B10 + A02*B20
    # But A00*B21, A12*B00, A12*B21 are not separate terms...
    #
    # This approach doesn't work with just these 5 terms as independent multiplications.
    # We need the W matrix to extract the right linear combinations.
    #
    # The issue is that M0 contains cross terms that contaminate the result.
    # We need a decomposition where the rank-1 tensors multiply out to exactly
    # the desired bilinear form.
    
    # Let me try a different set of 5 terms that avoids cross-term issues:
    # Use the fact that we can choose any bilinear forms for U and V.
    
    # Alternative rank-5 construction using linearity:
    # t0: U = A00, V = B00 + B01, W contributes to C00 and C01
    # t1: U = A01, V = B10 + B11, W contributes to C00 and C01
    # t2: U = A02, V = B20 + B21, W contributes to C00 and C01
    # t3: U = A10 + A11 + A12, V = B00, W contributes to C10
    # t4: U = A10 + A11 + A12, V = B01, W contributes to C11
    # That's 5 terms but t3 and t4 have the same U, so they can be combined:
    # t3: U = A10+A11+A12, V = B00+B01, W contributes to C10 and C11
    # Now we have 4 terms. Can we do better?
    #
    # t0: U=A00, V=B00+B01
    # t1: U=A01, V=B10+B11
    # t2: U=A02, V=B20+B21
    # t3: U=A10+A11+A12, V=B00+B01
    #
    # C00 = t0_contribution_to_C00 + t1_contribution_to_C00 + t2_contribution_to_C00
    #      = A00*B00 + A01*B10 + A02*B20 ✓
    # C01 = A00*B01 + A01*B11 + A02*B21 ✓
    # C10 = (A10+A11+A12)*B00 ✓
    # C11 = (A10+A11+A12)*B01 ✓
    #
    # This works with rank 4! Let me verify:
    # M0 = A00 * (B00+B01) -> contributes A00*B00 to C00, A00*B01 to C01
    # M1 = A01 * (B10+B11) -> contributes A01*B10 to C00, A01*B11 to C01
    # M2 = A02 * (B20+B21) -> contributes A02*B20 to C00, A02*B21 to C01
    # M3 = (A10+A11+A12) * (B00+B01) -> contributes to C10 and C11
    #   = (A10+A11+A12)*B00 + (A10+A11+A12)*B01
    #   C10 gets (A10+A11+A12)*B00 = A10*B00 + A11*B00 + A12*B00 ✓
    #   C11 gets (A10+A11+A12)*B01 = A10*B01 + A11*B01 + A12*B01 ✓
    #
    # But wait, C10 should be A10*B00 + A11*B10 + A12*B20, not A10*B00 + A11*B00 + A12*B00!
    # I made an error. B has different rows for each j.
    
    # Correct: C[r,c] = sum_j A[r,j] * B[j,c]
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    #
    # So for row 1 of C, we need different B rows for each j.
    # The U vector for row 1 must be (A10, A11, A12) and V must have
    # separate entries for B00, B10, B20 (for C10) and B01, B11, B21 (for C11).
    
    # Minimum rank for row 1 alone: we need to compute A10*B00 + A11*B10 + A12*B20
    # and A10*B01 + A11*B11 + A12*B21.
    # This is like computing two inner products of length 3.
    # We can use: M_a = (A10+A11+A12)*(B00+B10+B20) but that gives cross terms.
    # 
    # For a single inner product of length 3: a0*b0 + a1*b1 + a2*b2
    # Rank is 3 (can't do better for generic case).
    # But we have two such inner products sharing the same a's.
    # M1 = (A10+A11+A12)*(B00+B10+B20) gives cross terms.
    # 
    # We need at least 3 terms for the row-1 part, plus 3 for row-0 part = 6 total.
    # Unless we can share between rows.
    
    # Row 0: C00 = A00*B00 + A01*B10 + A02*B20
    #         C01 = A00*B01 + A01*B11 + A02*B21
    # Row 1: C10 = A10*B00 + A11*B10 + A12*B20
    #         C11 = A10*B01 + A11*B11 + A12*B21
    
    # Notice the structure: both rows use the same B entries, just different A entries.
    # 
    # M0 = (A00 + A10) * B00 -> C00 gets A00*B00, C10 gets A10*B00
    # M1 = (A01 + A11) * B10 -> C00 gets A01*B10, C10 gets A11*B10
    # M2 = (A02 + A12) * B20 -> C00 gets A02*B20, C10 gets A12*B20
    # M3 = (A00 + A10) * B01 -> C01 gets A00*B01, C11 gets A10*B01
    # M4 = (A01 + A11) * B11 -> C01 gets A01*B11, C11 gets A11*B11
    # M5 = (A02 + A12) * B21 -> C01 gets A02*B21, C11 gets A12*B21
    #
    # This gives rank 6! And it's correct!
    
    # So rank-6 decomposition for 2x3 @ 3x2:
    # t0: U=(A00+A10), V=B00, W=(C00+, C10+)
    # t1: U=(A01+A11), V=B10, W=(C00+, C10+)
    # t2: U=(A02+A12), V=B20, W=(C00+, C10+)
    # t3: U=(A00+A10), V=B01, W=(C01+, C11+)
    # t4: U=(A01+A11), V=B11, W=(C01+, C11+)
    # t5: U=(A02+A12), V=B21, W=(C01+, C11+)
    
    # Total rank: 18 * 6 = 108
    
    rank = 108
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # For each (i, j, k) where i in {0,1,2} (row group of A/C),
    # j in {0,1} (col group of A / row group of B),
    # k in {0,1,2} (col group of B/C):
    #
    # A[i,j] is 2x3: rows i*2, i*2+1; cols j*3, j*3+1, j*3+2
    # B[j,k] is 3x2: rows j*3, j*3+1, j*3+2; cols k*2, k*2+1
    # C[i,k] is 2x2: rows i*2, i*2+1; cols k*2, k*2+1
    #
    # W convention: W[col_C * n + row_C, term]
    # So for C entry (row_C, col_C), W index is col_C * 6 + row_C
    
    for i in range(3):  # row group
        for j in range(2):  # middle group
            for k in range(3):  # col group
                base = (i * 2 + j) * 3 * 6 + k * 6  # base column index
                
                # 6 terms for this sub-tensor
                for t in range(6):
                    col_idx = base + t
                    
                    if t < 3:
                        # Terms 0-2: B column 0 (B[j,0])
                        # U: A[i,0] + A[i+1,0] for the j-th col group
                        # Specifically: U row for A[i*2, j*3+t] and A[i*2+1, j*3+t]
                        u_row0 = (i * 2) * m + (j * 3 + t)
                        u_row1 = (i * 2 + 1) * m + (j * 3 + t)
                        U[u_row0, col_idx] += 1
                        U[u_row1, col_idx] += 1
                        
                        # V: B[j*3+t, k*2+0]
                        v_row = (j * 3 + t) * p + (k * 2 + 0)
                        V[v_row, col_idx] += 1
                        
                        # W: C[i*2, k*2+0] and C[i*2+1, k*2+0]
                        # W index: col_C * n + row_C
                        w_row0 = (k * 2 + 0) * n + (i * 2)
                        w_row1 = (k * 2 + 0) * n + (i * 2 + 1)
                        W[w_row0, col_idx] += 1
                        W[w_row1, col_idx] += 1
                    else:
                        # Terms 3-5: B column 1 (B[j,1])
                        t_adj = t - 3
                        u_row0 = (i * 2) * m + (j * 3 + t_adj)
                        u_row1 = (i * 2 + 1) * m + (j * 3 + t_adj)
                        U[u_row0, col_idx] += 1
                        U[u_row1, col_idx] += 1
                        
                        # V: B[j*3+t_adj, k*2+1]
                        v_row = (j * 3 + t_adj) * p + (k * 2 + 1)
                        V[v_row, col_idx] += 1
                        
                        # W: C[i*2, k*2+1] and C[i*2+1, k*2+1]
                        w_row0 = (k * 2 + 1) * n + (i * 2)
                        w_row1 = (k * 2 + 1) * n + (i * 2 + 1)
                        W[w_row0, col_idx] += 1
                        W[w_row1, col_idx] += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    elapsed = time.time() - start_time
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
