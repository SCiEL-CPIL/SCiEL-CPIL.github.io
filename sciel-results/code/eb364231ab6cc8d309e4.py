"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank exact tensor decomposition for 4x5 * 5x6 multiplication.
    Returns {'U': (20, rank), 'V': (30, rank), 'W': (24, rank)}.
    """
    t0 = time.time()
    DEADLINE = 140  # seconds, safety margin from 150s limit

    n, m, p = 4, 5, 6
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24

    # --- Schoolbook Fallback ---
    # Rank = n*m*p = 120
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    rank_school = n * m * p
    U_school = np.zeros((num_U, rank_school), dtype=float)
    V_school = np.zeros((num_V, rank_school), dtype=float)
    W_school = np.zeros((num_W, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # --- Block Decomposition Strategy ---
    # We partition the indices to exploit block structure.
    # I: 4 rows -> 2 blocks of size 2: I0=[0,1], I1=[2,3]
    # J: 5 cols -> 2 blocks of size 2, 1 block of size 1: J0=[0,1], J1=[2,3], J2=[4]
    # K: 6 cols -> 3 blocks of size 2: K0=[0,1], K1=[2,3], K2=[4,5]
    
    # Block types:
    # 1. 2x2 * 2x2 -> 2x2 result. (I-block size 2, J-block size 2, K-block size 2)
    #    Count: 2 (I) * 2 (J) * 3 (K) = 12 blocks.
    #    Rank of 2x2 mat mult tensor is 7 (Strassen).
    # 2. 2x1 * 1x2 -> 2x2 result. (I-block size 2, J-block size 1, K-block size 2)
    #    Count: 2 (I) * 1 (J) * 3 (K) = 6 blocks.
    #    This is an outer product of a 2x1 vector and 1x2 vector.
    #    C[i,k] = A[i, j0] * B[j0, k].
    #    Tensor T[i, j0, k] = A[i,j0] B[j0,k].
    #    This is rank 4? Let's check.
    #    T is 2x1x2.
    #    We need sum_r U[i, r] V[j0, r] W[k, i] = 1? No, the tensor entries are 1.
    #    Wait, the tensor T[i, j, k] = 1 for all valid i,j,k in the block.
    #    So for a 2x1 * 1x2 block, we have indices i in {0,1}, j in {0}, k in {0,1}.
    #    We need sum_r U[i, r] V[0, r] W[k, i] = 1.
    #    This is equivalent to finding matrices U (2xR), V (1xR), W (2x2xR? No, W is flattened).
    #    Actually, W index is (k, i).
    #    Let's just use schoolbook for these small blocks to be safe, or find rank.
    #    Rank of 2x1x2 tensor (all ones):
    #    T[i, 0, k] = 1.
    #    Can we do it in rank 2?
    #    If r=2:
    #    U = [[1, 0], [0, 1]]
    #    V = [[1, 1]]
    #    W[k, i]:
    #    Term 1: U[i,0] V[0,0] W[k, i] -> W[k, i] must be 1 if i=0, 0 if i=1?
    #    If i=0: 1*1*W[k,0] = 1 => W[k,0]=1.
    #    If i=1: 0*1*W[k,1] = 1 => Impossible.
    #    
    #    Let's try:
    #    U = [[1, 1], [1, 1]]
    #    V = [[1, 1]]
    #    W[k, i]:
    #    i=0: 1*1*W[k,0] = 1 => W[k,0]=1.
    #    i=1: 1*1*W[k,1] = 1 => W[k,1]=1.
    #    So W is all ones.
    #    Check: sum_r U[i,r] V[0,r] W[k,i] = U[i,0]*1*W[k,i] + U[i,1]*1*W[k,i] = (1+1)*1*1 = 2.
    #    We need 1. So divide by 2.
    #    U = [[0.5, 0.5], [0.5, 0.5]], V = [[1, 1]], W = [[1, 1], [1, 1]].
    #    Sum = 0.5*1*1 + 0.5*1*1 = 1.
    #    So Rank 2 works for 2x1 * 1x2!
    #
    #    Let's verify rank 1?
    #    U[i,0] V[0,0] W[k,i] = 1.
    #    U[i,0] W[k,i] = 1/V.
    #    U[0,0] W[k,0] = 1/V
    #    U[1,0] W[k,1] = 1/V
    #    W is 2x2 (k=0,1; i=0,1).
    #    W[0,0], W[1,0] depend on U[0,0].
    #    W[0,1], W[1,1] depend on U[1,0].
    #    If U[0,0]=1, W[:,0]=1/V. If U[1,0]=1, W[:,1]=1/V.
    #    Then W is constant columns.
    #    But we need T[i,0,k]=1 for all i,k.
    #    The term is U[i,0] V[0,0] W[k,i].
    #    This couples i and k via W.
    #    If i=0, k=0: U[0,0]V[0,0]W[0,0] = 1
    #    If i=0, k=1: U[0,0]V[0,0]W[1,0] = 1
    #    => W[0,0] = W[1,0].
    #    If i=1, k=0: U[1,0]V[0,0]W[0,1] = 1
    #    If i=1, k=1: U[1,0]V[0,0]W[1,1] = 1
    #    => W[0,1] = W[1,1].
    #    So W has identical rows? No, identical columns in the 'i' dimension?
    #    W[k,i]. Rows are k. Cols are i.
    #    W[0,0]=W[1,0] means col 0 is constant? No, W[0,0] is row 0 col 0. W[1,0] is row 1 col 0.
    #    So column 0 of W is constant.
    #    Column 1 of W is constant.
    #    Let W = [[a, b], [a, b]].
    #    Then:
    #    i=0: U[0,0] V[0,0] a = 1
    #    i=1: U[1,0] V[0,0] b = 1
    #    This works for any a,b != 0.
    #    So Rank 1 works?
    #    Wait, U is 2x1. V is 1x1. W is 2x2 (flattened to 4 entries in the global W matrix? No, W is (p*n, rank)).
    #    In the block, W has indices (k, i). k in {0,1}, i in {0,1}.
    #    So W_block is 2x2.
    #    If we use Rank 1, we have 1 column in U, V, W.
    #    U_block: 2x1. V_block: 1x1. W_block: 2x2.
    #    The contribution to T[i, 0, k] is U[i, 0] * V[0, 0] * W[k, i].
    #    We established this can be 1 for all i,k with Rank 1.
    #    So Rank of 2x1 * 1x2 is 1.
    #
    #    Let's double check.
    #    U = [1, 1]^T. V = [1]. W = [[1, 1], [1, 1]].
    #    i=0, k=0: 1*1*1 = 1.
    #    i=0, k=1: 1*1*1 = 1.
    #    i=1, k=0: 1*1*1 = 1.
    #    i=1, k=1: 1*1*1 = 1.
    #    Yes! Rank 1.

    # So, block ranks:
    # 2x2 * 2x2: Rank 7 (Strassen)
    # 2x1 * 1x2: Rank 1
    
    # Total Rank Estimate:
    # 12 blocks of type 1: 12 * 7 = 84
    # 6 blocks of type 2: 6 * 1 = 6
    # Total = 90.
    
    # This is an improvement over 120.
    
    # We need to construct the Strassen decomposition for 2x2x2.
    # And the Rank 1 decomposition for 2x1x2.
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W columns.
    # Each M corresponds to 1 column in U, V, W.
    # U column has entries for A11, A22, etc.
    # V column has entries for B11, B22, etc.
    # W column has entries for C11, C22, etc.
    
    # Let's define the local indices for a 2x2 block.
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1:
    # U: A11(1), A22(1) -> [1, 0, 0, 1]
    # V: B11(1), B22(1) -> [1, 0, 0, 1]
    # W: C11(1), C22(1) -> [1, 0, 0, 1]
    
    # M2:
    # U: A21(1), A22(1) -> [0, 0, 1, 1]
    # V: B11(1) -> [1, 0, 0, 0]
    # W: C21(1) -> [0, 0, 1, 0]
    
    # M3:
    # U: A11(1) -> [1, 0, 0, 0]
    # V: B21(1), B22(-1) -> [0, 0, 1, -1]
    # W: C12(1) -> [0, 1, 0, 0]
    
    # M4:
    # U: A22(1) -> [0, 0, 0, 1]
    # V: B12(1), B11(-1) -> [-1, 1, 0, 0]
    # W: C11(1), C21(1) -> [1, 0, 1, 0]
    
    # M5:
    # U: A11(1), A12(1) -> [1, 1, 0, 0]
    # V: B22(1) -> [0, 0, 0, 1]
    # W: C11(-1), C12(1) -> [-1, 1, 0, 0]
    
    # M6:
    # U: A21(1), A11(-1) -> [-1, 0, 1, 0]
    # V: B11(1), B12(1) -> [1, 1, 0, 0]
    # W: C22(1) -> [0, 0, 0, 1]
    
    # M7:
    # U: A12(1), A22(-1) -> [0, 1, 0, -1]
    # V: B21(1), B22(1) -> [0, 0, 1, 1]
    # W: C11(1), C22(1) -> [1, 0, 0, 1]
    
    # Let's verify M4 contribution to C11 and C21.
    # C11 = ... + M4 ...
    # C21 = ... + M4 ...
    # So W for M4 should have 1 at C11 and 1 at C21.
    # My W for M4 was [1, 0, 1, 0]. Correct.
    
    # Let's verify M5 contribution to C11 and C12.
    # C11 = ... - M5 ...
    # C12 = ... + M5 ...
    # So W for M5 should have -1 at C11 and 1 at C12.
    # My W for M5 was [-1, 1, 0, 0]. Correct.
    
    # Let's verify M7 contribution to C11 and C22.
    # C11 = ... + M7 ...
    # C22 = ... + M7 ...
    # So W for M7 should have 1 at C11 and 1 at C22.
    # My W for M7 was [1, 0, 0, 1]. Correct.
    
    # Now construct the full matrices.
    
    U_struct = np.zeros((num_U, 90), dtype=float)
    V_struct = np.zeros((num_V, 90), dtype=float)
    W_struct = np.zeros((num_W, 90), dtype=float)
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i
    
    col_offset = 0
    
    # Define Strassen terms as lists of (index, coeff) for U, V, W
    # Indices are local 0..3
    strassen_terms = [
        # M1
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        # M2
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1)]
        },
        # M3
        {
            'U': [(0, 1)],
            'V': [(2, 1), (3, -1)],
            'W': [(1, 1)]
        },
        # M4
        {
            'U': [(3, 1)],
            'V': [(1, 1), (0, -1)],
            'W': [(0, 1), (2, 1)]
        },
        # M5
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)]
        },
        # M6
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        # M7
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        }
    ]
    
    # Process 2x2 * 2x2 blocks
    # I blocks: [0,1], [2,3]
    # J blocks: [0,1], [2,3]
    # K blocks: [0,1], [2,3], [4,5]
    
    i_blocks = [(0, 2), (2, 4)]
    j_blocks_2x2 = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 6)]
    
    for i_start, i_end in i_blocks:
        for j_start, j_end in j_blocks_2x2:
            for k_start, k_end in k_blocks:
                # Local indices mapping
                # A: i in [i_start, i_end), j in [j_start, j_end)
                # Local 0: (i_start, j_start)
                # Local 1: (i_start, j_start+1)
                # Local 2: (i_start+1, j_start)
                # Local 3: (i_start+1, j_start+1)
                
                # B: j in [j_start, j_end), k in [k_start, k_end)
                # Local 0: (j_start, k_start)
                # Local 1: (j_start, k_start+1)
                # Local 2: (j_start+1, k_start)
                # Local 3: (j_start+1, k_start+1)
                
                # C: i in [i_start, i_end), k in [k_start, k_end)
                # Local 0: (i_start, k_start)
                # Local 1: (i_start, k_start+1)
                # Local 2: (i_start+1, k_start)
                # Local 3: (i_start+1, k_start+1)
                
                for term in strassen_terms:
                    col = col_offset
                    # Fill U
                    for local_idx, coeff in term['U']:
                        li, lj = divmod(local_idx, 2) # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> divmod gives (0,0),(0,1),(1,0),(1,1)
                        gi = i_start + li
                        gj = j_start + lj
                        U_struct[get_u_idx(gi, gj), col] = coeff
                    
                    # Fill V
                    for local_idx, coeff in term['V']:
                        lj, lk = divmod(local_idx, 2)
                        gj = j_start + lj
                        gk = k_start + lk
                        V_struct[get_v_idx(gj, gk), col] = coeff
                    
                    # Fill W
                    for local_idx, coeff in term['W']:
                        li, lk = divmod(local_idx, 2)
                        gi = i_start + li
                        gk = k_start + lk
                        W_struct[get_w_idx(gk, gi), col] = coeff
                    
                    col_offset += 1
    
    # Process 2x1 * 1x2 blocks
    # I blocks: [0,1], [2,3]
    # J blocks: [4] (size 1)
    # K blocks: [0,1], [2,3], [4,5]
    
    # Rank 1 term for this block:
    # U: all 1s (2 entries)
    # V: 1 (1 entry)
    # W: all 1s (4 entries)
    
    j_block_1x1 = [(4, 5)]
    
    for i_start, i_end in i_blocks:
        for j_start, j_end in j_block_1x1:
            for k_start, k_end in k_blocks:
                col = col_offset
                
                # U: i in [i_start, i_end), j in [j_start, j_end)
                for li in range(i_end - i_start):
                    gi = i_start + li
                    gj = j_start # only one j
                    U_struct[get_u_idx(gi, gj), col] = 1.0
                
                # V: j in [j_start, j_end), k in [k_start, k_end)
                gj = j_start
                # V is 1x1 in this block? No, V index is (j, k).
                # The term is U[i,j] V[j,k] W[k,i].
                # For the block, j is fixed.
                # The rank 1 decomposition we found was:
                # U[i, 0] = 1
                # V[0, 0] = 1  (where 0 is the local j index, but V is flattened)
                # Wait, V is (m*p, rank). The entry is V[j*p+k, r].
                # In the block, j is fixed. k varies.
                # My previous derivation:
                # T[i, j, k] = 1.
                # Term: U[i, r] V[j, k, r] W[k, i, r].
                # If we use Rank 1:
                # U[i, 0] = 1 for all i.
                # V[j, k, 0] = 1 for all k?
                # W[k, i, 0] = 1 for all k, i?
                # Then sum = 1 * 1 * 1 = 1.
                # So yes, V should be 1 for all k in the block.
                
                for lk in range(k_end - k_start):
                    gk = k_start + lk
                    V_struct[get_v_idx(gj, gk), col] = 1.0
                
                # W: k in [k_start, k_end), i in [i_start, i_end)
                for lk in range(k_end - k_start):
                    for li in range(i_end - i_start):
                        gk = k_start + lk
                        gi = i_start + li
                        W_struct[get_w_idx(gk, gi), col] = 1.0
                
                col_offset += 1
    
    # Verify the structured solution
    # Check if time allows
    if time.time() - t0 < DEADLINE - 5:
        # Verify: sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
        # We can do a spot check or full check.
        # Full check is O(n*m*p*rank) = 120 * 90 = 10800 ops. Very fast.
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Calculate sum
                    u_val = U_struct[i * m + j, :]
                    v_val = V_struct[j * p + k, :]
                    w_val = W_struct[k * n + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    if not np.isclose(s, 1.0, atol=1e-9):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_U, best_V, best_W = U_struct, V_struct, W_struct
            best_rank = col_offset
    
    # Return the best found
    # Convert to lists for JSON compatibility if needed, but numpy arrays are fine if the evaluator handles it.
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # The return interface in the current program is {"U": U, "V": V, "W": W} where U,V,W are numpy arrays.
    # Inspiration 1 used lists. Inspiration 2 used lists.
    # Current program used numpy arrays.
    # I will stick to numpy arrays as in the current program.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
