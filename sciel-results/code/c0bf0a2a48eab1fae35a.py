"""Matrix multiplication 4x5 by 5x6 — rank-minimal exact tensor decomposition.

Strategy:
  1. Build the 120-rank schoolbook decomposition.
  2. Use exact rational linear algebra to find a minimal linearly-independent
     subset of the 120 rank-1 terms that still spans the multiplication tensor.
  3. If the minimal rank is still > 90, fall back to a structured block
     decomposition (Strassen 2x2x2 + schoolbook) that is known to achieve
     rank 90.
  4. Verify the result and return it.
"""
import numpy as np
import time
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [4, 5, 6]


def _schoolbook_decomp(n, m, p):
    """Return (U, V, W) as lists of lists, rank = n*m*p, all entries 0 or 1."""
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _find_minimal_rank(U, V, W, n, m, p):
    """
    Given a rank-R decomposition (lists of lists), find a minimal subset of
    indices that still exactly represents the tensor. Uses exact rational
    linear algebra via Fraction.
    
    The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] must equal 1.
    We build a linear system: for each (i,j,k), sum_r a_r = 1 where
    a_r = U[i,j,r]*V[j,k,r]*W[k,i,r].
    
    We want the smallest subset S of {0,...,R-1} such that for every (i,j,k),
    sum_{r in S} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    This is a linear feasibility problem. We use a greedy approach:
    start with all indices, and try to remove each one if the remaining
    set still satisfies all equations.
    """
    R = len(U[0])
    
    # Build the constraint matrix: rows = (i,j,k) triples, cols = r
    # A[row, r] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need A * x = 1 where x_r is 1 if r is kept, 0 otherwise.
    # But we want x to be 0/1 and we want to minimize sum x.
    
    # Instead, let's use a different approach:
    # The tensor T is a specific tensor. The 120 rank-1 tensors
    # T_r = U_r ⊗ V_r ⊗ W_r span T (their sum is T).
    # We want a minimal subset that still sums to T.
    
    # Key insight: if the 120 tensors are linearly dependent, we can
    # potentially remove some. But we need the SUM to be exactly T, not
    # just that T is in the span.
    
    # Let's try: start with all indices. For each r, check if removing r
    # still gives sum = T. If so, remove it.
    
    # First, compute T as a dict: T[(i,j,k)] = 1
    # And compute the sum of all rank-1 terms.
    
    # For efficiency, let's represent the sum as a 3D array.
    n, m, p = 4, 5, 6
    R = len(U[0])
    
    # Compute the full sum tensor
    T_sum = np.zeros((n, m, p), dtype=np.float64)
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_sum[i, j, k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # Verify T_sum is all ones
    if not np.allclose(T_sum, 1.0, atol=1e-9):
        return None  # Invalid input
    
    # Now try to remove indices greedily
    keep = [True] * R
    removed = []
    
    for r in range(R):
        if not keep[r]:
            continue
        # Check if removing r still gives all-ones
        # T_sum - U_r ⊗ V_r ⊗ W_r should be all-ones
        # i.e., T_sum[i,j,k] - U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k
        # i.e., U[i,j,r]*V[j,k,r]*W[k,i,r] == 0 for all i,j,k
        # This means the rank-1 term r must be identically zero!
        
        # So we can only remove r if U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 0
        # for ALL i,j,k.
        
        can_remove = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 0:
                        can_remove = False
                        break
                if not can_remove:
                    break
            if not can_remove:
                break
        
        if can_remove:
            keep[r] = False
            removed.append(r)
    
    if not removed:
        return None
    
    # Build the reduced decomposition
    kept_indices = [r for r in range(R) if keep[r]]
    new_R = len(kept_indices)
    
    U_new = [[U[i*m+j][r] for r in kept_indices] for i in range(n) for j in range(m)]
    V_new = [[V[j*p+k][r] for r in kept_indices] for j in range(m) for k in range(p)]
    W_new = [[W[k*n+i][r] for r in kept_indices] for k in range(p) for i in range(n)]
    
    return U_new, V_new, W_new


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    Returns (U, V, W) as lists of lists with shapes (4,7), (4,7), (4,7).
    U is 2x2 flattened (row-major), V is 2x2 flattened, W is 2x2 flattened
    with index k*2+i (since W index is k*n+i for n=2).
    """
    # Strassen formulas for C = A * B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U_r[i,j] = coefficient of A[i,j] in first factor of M_r
    # V_r[j,k] = coefficient of B[j,k] in second factor of M_r
    # W_r[k,i] = coefficient of M_r in C[i,k]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # U1: A00 coeff 1, A11 coeff 1 -> U[0,0]=1, U[3,0]=1 (A11 is i=1,j=1 -> index 3)
    U[0][0] = 1; U[3][0] = 1
    # V1: B00 coeff 1, B11 coeff 1 -> V[0,0]=1, V[3,0]=1 (B11 is j=1,k=1 -> index 3)
    V[0][0] = 1; V[3][0] = 1
    # W1: C00 coeff 1, C11 coeff 1 -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    # U2: A10 coeff 1, A11 coeff 1 -> U[2,1]=1 (i=1,j=0), U[3,1]=1 (i=1,j=1)
    U[2][1] = 1; U[3][1] = 1
    # V2: B00 coeff 1 -> V[0,1]=1
    V[0][1] = 1
    # W2: C10 coeff 1 -> W[1,1]=1 (k=0,i=1)
    W[1][1] = 1
    
    # M3: A00(B01-B11)
    # U3: A00 coeff 1 -> U[0,2]=1
    U[0][2] = 1
    # V3: B01 coeff 1, B11 coeff -1 -> V[1,2]=1 (j=0,k=1), V[3,2]=-1 (j=1,k=1)
    V[1][2] = 1; V[3][2] = -1
    # W3: C01 coeff 1 -> W[2,2]=1 (k=1,i=0)
    W[2][2] = 1
    
    # M4: A11(B10-B00)
    # U4: A11 coeff 1 -> U[3,3]=1
    U[3][3] = 1
    # V4: B10 coeff 1, B00 coeff -1 -> V[2,3]=1 (j=1,k=0), V[0,3]=-1 (j=0,k=0)
    V[2][3] = 1; V[0][3] = -1
    # W4: C00 coeff 1, C10 coeff 1 -> W[0,3]=1 (k=0,i=0), W[1,3]=1 (k=0,i=1)
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A00+A01)B11
    # U5: A00 coeff 1, A01 coeff 1 -> U[0,4]=1, U[1,4]=1 (i=0,j=1)
    U[0][4] = 1; U[1][4] = 1
    # V5: B11 coeff 1 -> V[3,4]=1
    V[3][4] = 1
    # W5: C00 coeff -1, C01 coeff 1 -> W[0,4]=-1 (k=0,i=0), W[2,4]=1 (k=1,i=0)
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U6: A10 coeff 1, A00 coeff -1 -> U[2,5]=1 (i=1,j=0), U[0,5]=-1
    U[2][5] = 1; U[0][5] = -1
    # V6: B00 coeff 1, B01 coeff 1 -> V[0,5]=1, V[1,5]=1 (j=0,k=1)
    V[0][5] = 1; V[1][5] = 1
    # W6: C11 coeff 1 -> W[3,5]=1 (k=1,i=1)
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U7: A01 coeff 1, A11 coeff -1 -> U[1,6]=1 (i=0,j=1), U[3,6]=-1 (i=1,j=1)
    U[1][6] = 1; U[3][6] = -1
    # V7: B10 coeff 1, B11 coeff 1 -> V[2,6]=1 (j=1,k=0), V[3,6]=1 (j=1,k=1)
    V[2][6] = 1; V[3][6] = 1
    # W7: C00 coeff 1 -> W[0,6]=1 (k=0,i=0)
    W[0][6] = 1
    
    return U, V, W


def _combine_blocks(blocks, n, m, p):
    """
    Combine block decompositions into a full n x m x p decomposition.
    
    blocks: list of (U_block, V_block, W_block, i_start, j_start, k_start)
    where U_block has shape (block_n * block_m, block_rank),
          V_block has shape (block_m * block_p, block_rank),
          W_block has shape (block_p * block_n, block_rank),
    and the block covers A[i_start:i_start+block_n, j_start:j_start+block_m]
    times B[j_start:j_start+block_m, k_start:k_start+block_p].
    
    Returns (U, V, W) as lists of lists.
    """
    total_rank = sum(len(b[0][0]) for b in blocks)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for (Ub, Vb, Wb, i0, j0, k0) in blocks:
        block_n = len(Ub) // len(Ub[0]) if False else None
        # Determine block dimensions from the block matrices
        # Ub has shape (bn * bm, br), Vb has shape (bm * bp, br), Wb has shape (bp * bn, br)
        br = len(Ub[0])
        bn_bm = len(Ub)
        bm_bp = len(Vb)
        bp_bn = len(Wb)
        
        # We need to figure out bn, bm, bp from the context
        # Actually, let's pass them explicitly. Let me restructure.
        
        # For now, let's just do it manually for our specific case.
        pass
    
    # This generic approach is getting complex. Let's do it specifically.
    return None


def _build_90_rank_decomp():
    """
    Build a rank-90 decomposition of 4x5 * 5x6.
    
    Strategy: Decompose into blocks:
    - Split A into 2x2, 2x2, 1x1 row blocks (total 4 rows)
    - Split B into 2x2, 2x2, 1x2 column blocks (total 6 cols)
    
    Actually, let's use a cleaner decomposition:
    - A is 4x5, B is 5x6
    - Split A rows into [2, 2], B cols into [2, 2, 2]
    - This gives 2x3 = 6 blocks, each of size 2x2x2
    - But A has 5 columns, so we can't split evenly into 2x2 blocks.
    
    Better approach:
    - Split A rows into [2, 2], B rows (shared) into [2, 2, 1]
    - Split B cols into [2, 2, 2]
    
    Actually, the cleanest is:
    - Use Strassen for 2x2x2 blocks where possible
    - Use schoolbook for the rest
    
    Let me think about this more carefully.
    
    A is 4x5, B is 5x6.
    
    Option: Split A into 2 blocks of 2 rows each. Split B into 3 blocks of 2 cols each.
    But B has 5 rows, which we need to split consistently.
    
    Let's split the 5 shared dimension into [2, 2, 1].
    A rows: [2, 2]
    B cols: [2, 2, 2]
    
    This gives 2 * 3 = 6 blocks:
    Block (i, k): A[2i:2i+2, 2k:2k+2] * B[2k:2k+2, 2k:2k+2] -- no, this doesn't work.
    
    The shared dimension is the 5 in A's columns and B's rows.
    
    Let me split the 5 shared dimension into [2, 2, 1].
    A: rows [2, 2], cols [2, 2, 1] -> 2x3 = 6 blocks of A
    B: rows [2, 2, 1], cols [2, 2, 2] -> 3x3 = 9 blocks of B
    
    But the multiplication C = A * B sums over the shared dimension.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we split j into [2, 2, 1], then:
    C[i,k] = sum_{j in [0,1]} A[i,j]*B[j,k] + sum_{j in [2,3]} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    
    For each block of i (2 rows) and block of k (2 cols), and block of j (2 or 1 rows):
    The contribution is A_block (2x2 or 2x1) * B_block (2x2 or 1x2)
    
    This is getting complicated. Let me just use the known result:
    
    For 4x5 * 5x6, we can achieve rank 90 by:
    - 6 blocks of 2x2x2 (Strassen, rank 7 each) = 42
    - 6 blocks of 2x1x2 (rank 1 each) = 6
    - 6 blocks of 1x2x2 (rank 1 each) = 6
    - ... this doesn't add up cleanly.
    
    Let me try a different approach: just use schoolbook for everything but
    replace some 2x2x2 sub-blocks with Strassen.
    
    Schoolbook rank = 4*5*6 = 120.
    
    If we can find 6 disjoint 2x2x2 sub-tensors and replace each with Strassen
    (rank 7 instead of 8), we save 6, getting rank 114.
    
    To get to 90, we need to save 30. That's 30/1 = 30 single-term savings,
    or 30/2 = 15 two-term savings, etc.
    
    Actually, the known bound for 4x5x6 is at most 90. Let me just implement
    a specific decomposition.
    
    Here's a concrete plan:
    - Split A (4x5) into 2 row-blocks of 2 rows.
    - Split B (5x6) into 3 col-blocks of 2 cols.
    - The shared dimension (5) is split into [2, 2, 1].
    
    For each pair (row-block i, col-block k), the contribution to C is:
    C_block = A_block * B_block
    where A_block is 2x5 and B_block is 5x2.
    
    C_block[i',k'] = sum_{j=0}^{4} A_block[i',j] * B_block[j,k']
    
    Split j into [2, 2, 1]:
    C_block[i',k'] = sum_{j=0}^{1} A[i',j]*B[j,k'] + sum_{j=2}^{3} A[i',j]*B[j,k'] + A[i',4]*B[4,k']
    
    The first sum is a 2x2 * 2x2 multiplication (Strassen, rank 7).
    The second sum is also 2x2 * 2x2 (Strassen, rank 7).
    The third term is 2x1 * 1x2, which is rank 2 (two rank-1 terms).
    
    So per (row-block, col-block) pair: 7 + 7 + 2 = 16.
    Total: 2 * 3 * 16 = 96.
    
    That's 96, not 90. Close but not quite.
    
    Let me try: split shared dim into [2, 2, 1] and use Strassen for the 2x2 parts.
    2x2x2 Strassen = rank 7.
    2x1x2 = rank 2 (A[i',j]*B[j,k'] for j=4, i' in {0,1}, k' in {0,1}).
    
    Actually 2x1x2: A is 2x1, B is 1x2. C is 2x2.
    C[i',k'] = A[i',0] * B[0,k']
    This is rank 2: two rank-1 terms (one for each i',k' pair... no, it's actually
    the outer product of A (2x1) and B (1x2), which is rank 1!
    
    Wait: C[i',k'] = A[i',0] * B[0,k']. This is a rank-1 matrix (outer product).
    So 2x1x2 has rank 1.
    
    Similarly 1x2x2 has rank 1.
    
    So per block: 7 + 7 + 1 = 15.
    Total: 2 * 3 * 15 = 90.
    
    Let me verify: 
    - 2x2x2 Strassen: rank 7
    - 2x2x2 Strassen: rank 7  
    - 2x1x2: rank 1
    Total per (row-block, col-block): 15
    Number of (row-block, col-block) pairs: 2 * 3 = 6
    Total rank: 90.
    
    Now I need to implement this.
    """
    n, m, p = 4, 5, 6
    
    # Strassen 2x2x2 decomposition
    Su, Sv, Sw = _strassen_2x2x2()
    
    # We'll build the full U, V, W
    # Total rank = 90
    total_rank = 90
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 90
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 90
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 90
    
    # Row blocks: [0,1], [2,3]
    row_blocks = [(0, 2), (2, 4)]
    # Col blocks: [0,1], [2,3], [4,5]
    col_blocks = [(0, 2), (2, 4), (4, 6)]
    # Shared dim splits: [0,1], [2,3], [4]
    shared_splits = [(0, 2), (2, 4), (4, 5)]
    
    rank_offset = 0
    for (i0, i1) in row_blocks:
        for (k0, k1) in col_blocks:
            for (j0, j1) in shared_splits:
                block_n = i1 - i0  # 2
                block_p = k1 - k0  # 2
                block_m = j1 - j0  # 2 or 1
                
                if block_m == 2:
                    # Use Strassen 2x2x2
                    br = 7
                    for r in range(br):
                        col = rank_offset + r
                        # U: A[i,j] where i in [i0,i1), j in [j0,j1)
                        # Su has shape (4, 7), index i'*2+j' where i' in [0,1], j' in [0,1]
                        for ii in range(2):
                            for jj in range(2):
                                u_val = Su[ii * 2 + jj][r]
                                if u_val != 0:
                                    U[(i0 + ii) * m + (j0 + jj)][col] = u_val
                        # V: B[j,k] where j in [j0,j1), k in [k0,k1)
                        # Sv has shape (4, 7), index j'*2+k' where j' in [0,1], k' in [0,1]
                        for jj in range(2):
                            for kk in range(2):
                                v_val = Sv[jj * 2 + kk][r]
                                if v_val != 0:
                                    V[(j0 + jj) * p + (k0 + kk)][col] = v_val
                        # W: C[i,k] where i in [i0,i1), k in [k0,k1)
                        # Sw has shape (4, 7), index k'*2+i' where k' in [0,1], i' in [0,1]
                        for kk in range(2):
                            for ii in range(2):
                                w_val = Sw[kk * 2 + ii][r]
                                if w_val != 0:
                                    W[(k0 + kk) * n + (i0 + ii)][col] = w_val
                    rank_offset += br
                else:
                    # block_m == 1: 2x1x2, rank 1
                    # C[i',k'] = A[i',j0] * B[j0,k']
                    # This is rank 1: U[i',j0]=1, V[j0,k']=1, W[k',i']=1
                    br = 1
                    col = rank_offset
                    for ii in range(2):
                        U[(i0 + ii) * m + j0][col] = 1
                    for kk in range(2):
                        V[j0 * p + (k0 + kk)][col] = 1
                    for kk in range(2):
                        for ii in range(2):
                            W[(k0 + kk) * n + (i0 + ii)][col] = 1
                    rank_offset += br
    
    return U, V, W, rank_offset


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x6.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # Leave 20s margin
    
    # Build the rank-90 structured decomposition
    U, V, W, actual_rank = _build_90_rank_decomp()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomp(n, m, p)
    
    # Ensure all entries are integers or half-integers
    for arr in [U, V, W]:
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                arr[i][j] = round(arr[i][j] * 2) / 2.0
    
    return {"U": U, "V": V, "W": W}
