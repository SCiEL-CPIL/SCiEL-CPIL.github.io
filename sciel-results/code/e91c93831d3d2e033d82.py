"""Rewritten program for 2x5x6 matrix multiplication tensor decomposition using Strassen block decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    # Deadline guard: 120s limit, use 110s as safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Block decomposition:
    # Partition middle dimension m=5 into blocks of sizes [2, 2, 1]
    # A (2x5) = [A1 | A2 | A3] where A1,A2 are 2x2, A3 is 2x1
    # B (5x6) = [B1; B2; B3] where B1,B2 are 2x6, B3 is 1x6
    # C (2x6) = A1*B1 + A2*B2 + A3*B3
    #
    # For each 2x2 * 2x6 block: use Strassen on 3 column-pairs of 2x2x2
    #   Rank per 2x2x2 = 7, so rank per 2x2x6 = 7*3 = 21
    # For 2x1 * 1x6 block: rank = 1 (each output element is a single product)
    #
    # Total rank = 21 + 21 + 1 = 43
    
    rank_A1B1 = 7 * 3  # 21
    rank_A2B2 = 7 * 3  # 21
    rank_A3B3 = 1
    total_rank = rank_A1B1 + rank_A2B2 + rank_A3B3  # 43
    
    U = np.zeros((n * m, total_rank))  # (10, 43)
    V = np.zeros((m * p, total_rank))  # (30, 43)
    W = np.zeros((p * n, total_rank))  # (12, 43)
    
    col_offset = 0
    
    def add_strassen_2x2x2(a_row_start, a_col_start, b_row_start, c_col_start, col_off):
        """
        Add 7 Strassen terms for 2x2 * 2x2 multiplication.
        
        A block: rows a_row_start..a_row_start+1, cols a_col_start..a_col_start+1 (2x2)
        B block: rows b_row_start..b_row_start+1, cols c_col_start..c_col_start+1 (2x2)
        C block: rows 0..1, cols c_col_start..c_col_start+1 (2x2)
        
        Strassen:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B10-B11)
        M4 = A11*(B01-B10)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B10)
        M7 = (A01-A11)*(B01+B11)
        
        Z00 = M1 + M4 - M5 + M7
        Z01 = M3 + M5
        Z10 = M2 + M4
        Z11 = M1 - M2 + M3 + M6
        """
        # Index mapping:
        # A(i,j) -> U index: i*m + j  (i in {a_row_start, a_row_start+1}, j in {a_col_start, a_col_start+1})
        # B(j,k) -> V index: j*p + k  (j in {b_row_start, b_row_start+1}, k in {c_col_start, c_col_start+1})
        # C(i,k) -> W index: k*n + i  (i in {0,1}, k in {c_col_start, c_col_start+1})
        
        A00_idx = a_row_start * m + a_col_start
        A01_idx = a_row_start * m + a_col_start + 1
        A10_idx = (a_row_start + 1) * m + a_col_start
        A11_idx = (a_row_start + 1) * m + a_col_start + 1
        
        B00_idx = b_row_start * p + c_col_start
        B01_idx = b_row_start * p + c_col_start + 1
        B10_idx = (b_row_start + 1) * p + c_col_start
        B11_idx = (b_row_start + 1) * p + c_col_start + 1
        
        C00_idx = c_col_start * n + 0
        C01_idx = (c_col_start + 1) * n + 0
        C10_idx = c_col_start * n + 1
        C11_idx = (c_col_start + 1) * n + 1
        
        # 7 Strassen terms: (U_entries, V_entries, W_entries)
        # Each entry is (matrix_index, coefficient)
        
        terms = [
            # M1 = (A00+A11)*(B00+B11)
            (
                [(A00_idx, 1), (A11_idx, 1)],
                [(B00_idx, 1), (B11_idx, 1)],
                [(C00_idx, 1), (C11_idx, 1)]
            ),
            # M2 = (A10+A11)*B00
            (
                [(A10_idx, 1), (A11_idx, 1)],
                [(B00_idx, 1)],
                [(C10_idx, 1)]
            ),
            # M3 = A00*(B10-B11)
            (
                [(A00_idx, 1)],
                [(B10_idx, 1), (B11_idx, -1)],
                [(C01_idx, 1)]
            ),
            # M4 = A11*(B01-B10)
            (
                [(A11_idx, 1)],
                [(B01_idx, 1), (B10_idx, -1)],
                [(C00_idx, 1), (C10_idx, 1)]
            ),
            # M5 = (A00+A01)*B11
            (
                [(A00_idx, 1), (A01_idx, 1)],
                [(B11_idx, 1)],
                [(C00_idx, -1), (C01_idx, 1)]
            ),
            # M6 = (A10-A00)*(B00+B10)
            (
                [(A10_idx, 1), (A00_idx, -1)],
                [(B00_idx, 1), (B10_idx, 1)],
                [(C11_idx, 1)]
            ),
            # M7 = (A01-A11)*(B01+B11)
            (
                [(A01_idx, 1), (A11_idx, -1)],
                [(B01_idx, 1), (B11_idx, 1)],
                [(C00_idx, 1)]
            ),
        ]
        
        for t_idx, (u_entries, v_entries, w_entries) in enumerate(terms):
            col = col_off + t_idx
            for idx, coeff in u_entries:
                U[idx, col] += coeff
            for idx, coeff in v_entries:
                V[idx, col] += coeff
            for idx, coeff in w_entries:
                W[idx, col] += coeff
        
        return col_off + 7
    
    # Block 1: A1 (rows 0-1, cols 0-1 of A) * B1 (rows 0-1 of B, all 6 cols)
    # A1 is at A rows 0-1, cols 0-1
    # B1 is at B rows 0-1, cols 0-5
    # Use Strassen on 3 column pairs: (0,1), (2,3), (4,5)
    for pair in range(3):
        c_start = pair * 2
        col_offset = add_strassen_2x2x2(0, 0, 0, c_start, col_offset)
    
    # Block 2: A2 (rows 0-1, cols 2-3 of A) * B2 (rows 2-3 of B, all 6 cols)
    # A2 is at A rows 0-1, cols 2-3
    # B2 is at B rows 2-3, cols 0-5
    for pair in range(3):
        c_start = pair * 2
        col_offset = add_strassen_2x2x2(0, 2, 2, c_start, col_offset)
    
    # Block 3: A3 (rows 0-1, col 4 of A) * B3 (row 4 of B, all 6 cols)
    # This is a 2x1 * 1x6 multiplication
    # C[i,k] = A3[i] * B3[k] for i in {0,1}, k in {0,...,5}
    # Each output element is a single product, so we need 2*6 = 12 terms?
    # Wait, no. The tensor for 2x1 * 1x6 has rank 2*6 = 12? No.
    # Actually, for 2x1 * 1x6: C is 2x6, each C[i,k] = A[i]*B[k]
    # This can be done with rank 1 if we allow the W to depend on both i and k... 
    # But in our formulation, each rank-1 term contributes U[i,j]*V[j,k]*W[k,i]
    # For j=4 (the only middle index in this block):
    #   U[i,4]*V[4,k]*W[k,i] should sum to A[i]*B[k] for all i,k
    #   We can set U[i,4]=A[i], V[4,k]=B[k], W[k,i]=1 for all k,i
    #   But W[k,i] is shared across all blocks! We can't set it to 1 for all.
    #
    # Actually, the issue is that W is indexed by (k,i) and is shared.
    # For the 2x1*1x6 block, we need:
    #   sum_r U[i,4,r] * V[4,k,r] * W[k,i,r] = A[i]*B[k]
    #
    # The simplest way: use 12 separate rank-1 terms, one for each (i,k) pair:
    #   For (i,k): U[i,4,r]=1, V[4,k,r]=1, W[k,i,r]=1
    #   This gives rank 12 for this block.
    #
    # But wait, can we do better? The tensor for 2x1*1x6 is:
    #   T[i,4,k] = A[i]*B[k]
    # This is a rank-1 tensor in the (i,k) subspace for fixed j=4? No, it's not.
    # T[i,4,k] = A[i]*B[k] is actually a rank-1 tensor in the (i,k) space.
    # So we can write it as a single rank-1 term if we allow U,V,W to encode A and B.
    #
    # In our decomposition framework:
    #   sum_r U[i,4,r] * V[4,k,r] * W[k,i,r] = A[i]*B[k]
    #
    # If we set for r=0: U[i,4,0] = A[i], V[4,k,0] = B[k], W[k,i,0] = 1
    # Then sum = A[i]*B[k]*1 = A[i]*B[k]. 
    # But W[k,i,0] = 1 for all k,i conflicts with other blocks that use W.
    #
    # The problem is that W is shared. We need W[k,i,r] to be consistent.
    #
    # Alternative: use 2*6 = 12 terms, one per (i,k) pair.
    # For each (i,k): add a term with U[i,4]=1, V[4,k]=1, W[k,i]=1
    # This uses 12 columns.
    
    # Actually, let me reconsider. The total rank for 2x1*1x6 is indeed 12
    # if we can't share W entries. But maybe we can do it with fewer terms.
    
    # Let's think again. The tensor T[i,4,k] for i in {0,1}, k in {0,...,5}
    # We need: T[i,4,k] = sum_r U[i,4,r] * V[4,k,r] * W[k,i,r]
    #
    # This is equivalent to: the 2x6 matrix A3*B3 (outer product) must be represented
    # as a sum of rank-1 matrices, where the r-th rank-1 matrix is:
    #   (U[:,4,r] * V[4,:,r]) * W[:,r] (broadcast appropriately)
    #
    # Actually, for fixed j=4:
    #   C[i,k] = sum_r U[i,4,r] * V[4,k,r] * W[k,i,r]
    #
    # This is not a standard matrix product decomposition. The W factor depends on (k,i),
    # not just on r. This makes it harder to minimize rank.
    #
    # The safest approach: use 12 terms, one per (i,k) pair.
    # For (i,k) with i in {0,1}, k in {0,...,5}:
    #   U[i, 4, col] = 1
    #   V[4, k, col] = 1
    #   W[k, i, col] = 1
    #
    # This gives C[i,k] = 1*1*1 = 1 for the specific (i,k), and 0 for others.
    # But we need C[i,k] = A[i]*B[k], not 1.
    #
    # The issue is that the values of A and B are not fixed in the tensor decomposition.
    # The tensor T[i,j,k] = 1 for all i,j,k represents the multiplication map.
    # The actual matrix entries A[i,j] and B[j,k] are variables.
    #
    # So the tensor T[i,j,k] = 1 means: the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # In the decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k.
    #
    # For the 2x1*1x6 block (j=4):
    #   sum_r U[i,4,r]*V[4,k,r]*W[k,i,r] = 1 for all i in {0,1}, k in {0,...,5}
    #
    # We can set for each (i,k): one term with U[i,4]=1, V[4,k]=1, W[k,i]=1
    # That's 12 terms.
    
    # But can we do better? If we set:
    #   U[0,4,0] = 1, U[1,4,0] = 1
    #   V[4,k,0] = 1 for all k
    #   W[k,0,0] = 1 for all k, W[k,1,0] = 1 for all k
    # Then sum = 1*1*1 = 1 for all i,k. That's 1 term!
    # But W[k,i,0] = 1 for all k,i conflicts with other blocks.
    
    # The conflict is the issue. W is shared across all blocks.
    # In the Strassen blocks, W entries are used with specific coefficients.
    # If we set W[k,i,0] = 1 for all k,i, it will interfere.
    
    # So we need separate columns for the 2x1*1x6 block.
    # Using 12 terms is safe.
    
    # Actually, let's use 12 terms for simplicity and correctness.
    for i in range(2):
        for k in range(6):
            U[i * m + 4, col_offset] = 1.0
            V[4 * p + k, col_offset] = 1.0
            W[k * n + i, col_offset] = 1.0
            col_offset += 1
    
    # Wait, that's 12 terms, but I allocated only 1 for rank_A3B3.
    # Let me recalculate: total_rank should be 21 + 21 + 12 = 54.
    # But I set total_rank = 43. Let me fix this.
    
    # Actually, I made an error. Let me redo with correct ranks.
    
    # Let me restart with correct calculation.
    
    # For 2x1 * 1x6: we need 12 terms (one per i,k pair)
    # Total rank = 21 + 21 + 12 = 54
    
    # But wait, can we do the 2x1*1x6 with fewer terms?
    # The tensor T[i,4,k] = 1 for all i,k (12 entries).
    # We need sum_r U[i,4,r]*V[4,k,r]*W[k,i,r] = 1
    #
    # If we use 2 terms:
    #   r=0: U[0,4,0]=1, U[1,4,0]=1, V[4,k,0]=1 for all k, W[k,0,0]=1 for all k, W[k,1,0]=0 for all k
    #   r=1: U[0,4,1]=0, U[1,4,1]=1, V[4,k,1]=1 for all k, W[k,0,1]=0 for all k, W[k,1,1]=1 for all k
    # Then for i=0: sum = 1*1*1 + 0*1*0 = 1
    # For i=1: sum = 1*1*0 + 1*1*1 = 1
    # This works! And it uses only 2 terms.
    #
    # But W[k,0,0]=1 and W[k,1,1]=1 may conflict with other blocks.
    # In the Strassen blocks, W entries are used with various coefficients.
    # If we use new columns for these terms, there's no conflict.
    
    # So rank for 2x1*1x6 can be 2.
    # Total rank = 21 + 21 + 2 = 44.
    
    # Let me redo the construction with rank = 44.
    
    # Actually, let me just build it correctly from scratch.
    
    rank_A1B1 = 7 * 3  # 21
    rank_A2B2 = 7 * 3  # 21
    rank_A3B3 = 2       # 2 terms for 2x1*1x6
    total_rank = rank_A1B1 + rank_A2B2 + rank_A3B3  # 44
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    def add_strassen_2x2x2(a_row_start, a_col_start, b_row_start, c_col_start, col_off):
        A00_idx = a_row_start * m + a_col_start
        A01_idx = a_row_start * m + a_col_start + 1
        A10_idx = (a_row_start + 1) * m + a_col_start
        A11_idx = (a_row_start + 1) * m + a_col_start + 1
        
        B00_idx = b_row_start * p + c_col_start
        B01_idx = b_row_start * p + c_col_start + 1
        B10_idx = (b_row_start + 1) * p + c_col_start
        B11_idx = (b_row_start + 1) * p + c_col_start + 1
        
        C00_idx = c_col_start * n + 0
        C01_idx = (c_col_start + 1) * n + 0
        C10_idx = c_col_start * n + 1
        C11_idx = (c_col_start + 1) * n + 1
        
        terms = [
            # M1 = (A00+A11)*(B00+B11)
            (
                [(A00_idx, 1), (A11_idx, 1)],
                [(B00_idx, 1), (B11_idx, 1)],
                [(C00_idx, 1), (C11_idx, 1)]
            ),
            # M2 = (A10+A11)*B00
            (
                [(A10_idx, 1), (A11_idx, 1)],
                [(B00_idx, 1)],
                [(C10_idx, 1)]
            ),
            # M3 = A00*(B10-B11)
            (
                [(A00_idx, 1)],
                [(B10_idx, 1), (B11_idx, -1)],
                [(C01_idx, 1)]
            ),
            # M4 = A11*(B01-B10)
            (
                [(A11_idx, 1)],
                [(B01_idx, 1), (B10_idx, -1)],
                [(C00_idx, 1), (C10_idx, 1)]
            ),
            # M5 = (A00+A01)*B11
            (
                [(A00_idx, 1), (A01_idx, 1)],
                [(B11_idx, 1)],
                [(C00_idx, -1), (C01_idx, 1)]
            ),
            # M6 = (A10-A00)*(B00+B10)
            (
                [(A10_idx, 1), (A00_idx, -1)],
                [(B00_idx, 1), (B10_idx, 1)],
                [(C11_idx, 1)]
            ),
            # M7 = (A01-A11)*(B01+B11)
            (
                [(A01_idx, 1), (A11_idx, -1)],
                [(B01_idx, 1), (B11_idx, 1)],
                [(C00_idx, 1)]
            ),
        ]
        
        for t_idx, (u_entries, v_entries, w_entries) in enumerate(terms):
            col = col_off + t_idx
            for idx, coeff in u_entries:
                U[idx, col] += coeff
            for idx, coeff in v_entries:
                V[idx, col] += coeff
            for idx, coeff in w_entries:
                W[idx, col] += coeff
        
        return col_off + 7
    
    # Block 1: A1 (cols 0-1) * B1 (rows 0-1)
    for pair in range(3):
        c_start = pair * 2
        col_offset = add_strassen_2x2x2(0, 0, 0, c_start, col_offset)
    
    # Block 2: A2 (cols 2-3) * B2 (rows 2-3)
    for pair in range(3):
        c_start = pair * 2
        col_offset = add_strassen_2x2x2(0, 2, 2, c_start, col_offset)
    
    # Block 3: A3 (col 4) * B3 (row 4)
    # Use 2 terms:
    # Term 1: handles i=0
    #   U[0,4] = 1, U[1,4] = 1
    #   V[4,k] = 1 for all k
    #   W[k,0] = 1 for all k, W[k,1] = 0 for all k
    # Term 2: handles i=1
    #   U[0,4] = 0, U[1,4] = 1
    #   V[4,k] = 1 for all k
    #   W[k,0] = 0 for all k, W[k,1] = 1 for all k
    
    # Term 1
    col = col_offset
    U[0 * m + 4, col] = 1.0
    U[1 * m + 4, col] = 1.0
    for k in range(6):
        V[4 * p + k, col] = 1.0
        W[k * n + 0, col] = 1.0
        W[k * n + 1, col] = 0.0
    col_offset += 1
    
    # Term 2
    col = col_offset
    U[0 * m + 4, col] = 0.0
    U[1 * m + 4, col] = 1.0
    for k in range(6):
        V[4 * p + k, col] = 1.0
        W[k * n + 0, col] = 0.0
        W[k * n + 1, col] = 1.0
    col_offset += 1
    
    # Verify: check time
    if time.time() > deadline:
        pass
    
    # Convert to lists for compatibility
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-START
# The function above is the primary entry point.
# We keep the name construct() and the return interface.
# EVOLVE-BLOCK-END
