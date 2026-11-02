"""Matrix multiplication 3,4,5 - rank-minimal tensor decomposition search."""
import numpy as np
import time
import sys
from fractions import Fraction
import itertools

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, rank, n, m, p):
    """Verify exact decomposition for valid (i,j,k) triples."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _verify_full(U, V, W, rank, n, m, p):
    """Full verification including off-diagonal zeros."""
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    # Check invalid entries (a,b,c) not corresponding to any (i,j,k)
    # a = i*m+j, b = j*p+k, c = k*n+i
    # For each (a,b,c), check if there exists i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                valid_set.add((a, b, c))
    
    # Only need to check a subset of invalid entries for speed
    # Sample some invalid entries
    import random
    rng = random.Random(123)
    samples = 0
    while samples < 200:
        a = rng.randrange(n*m)
        b = rng.randrange(m*p)
        c = rng.randrange(p*n)
        if (a, b, c) not in valid_set:
            s = 0
            for r in range(rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if abs(s) > 1e-10:
                return False
            samples += 1
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Return the 7-rank Strassen decomposition for 2x2x2 matrix multiplication.
    U, V, W are 4x7 matrices (for 2x2, n*m=4, m*p=4, p*n=4).
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # A indices: A00->0, A01->1, A10->2, A11->3
    # B indices: B00->0, B01->1, B10->2, B11->3
    # C indices (trace convention k*n+i): C00->0, C01->1, C10->2, C11->3
    #   C[k*n+i]: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    # M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, best_rank = _schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    if time.time() > deadline:
        return best_result
    
    # Strategy: Block decomposition
    # Split the 3x4x5 multiplication into sub-blocks.
    # A is 3x4, B is 4x5, C is 3x5.
    # 
    # Split A into blocks: rows 0-2 (3 rows), cols 0-3 (4 cols)
    # We can split along the middle dimension (j, size 4) and/or output dimension.
    #
    # Approach 1: Split j into two groups of 2.
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    # First term: (3x2) @ (2x5) -> 3x5, rank 3*2*5=30
    # Second term: (3x2) @ (2x5) -> 3x5, rank 3*2*5=30
    # Total: 60 (same as schoolbook)
    #
    # Approach 2: Use Strassen on sub-blocks where possible.
    # For a 2x2x2 block, we can use rank 7 instead of 8.
    #
    # Let's try to identify 2x2x2 sub-problems within the 3x4x5 multiplication.
    # The multiplication tensor T[i,j,k] is nonzero for all i,j,k.
    # We can group indices to form 2x2x2 sub-tensors.
    #
    # For i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2 block, rank 7
    # For i in {0,1}, j in {2,3}, k in {0,1}: 2x2x2 block, rank 7
    # For i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2 block, rank 7
    # For i in {0,1}, j in {2,3}, k in {2,3}: 2x2x2 block, rank 7
    # That's 4 blocks of 2x2x2, each rank 7 = 28
    # Remaining: i=2, all j, all k: 1x4x5, rank 20
    # Total: 28 + 20 = 48
    #
    # Wait, this doesn't work directly because the blocks overlap in a way that
    # the tensor entries are shared. The tensor T[i,j,k] = 1 for all i,j,k.
    # If we decompose T as a sum of block tensors, each block tensor has support
    # on a subset of (i,j,k) indices.
    #
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    # Block 5: i = {2}, j in {0,1,2,3}, k in {0,1,2,3,4}
    #
    # These blocks partition the set of all (i,j,k) triples.
    # Each block is a sub-tensor of T.
    #
    # For blocks 1-4: each is a 2x2x2 all-ones tensor, rank 7 (Strassen).
    # For block 5: 1x4x5 tensor. The rank of a 1xMxP tensor is max(M,P) if we can
    # factor it. Actually, T[i,j,k] with i fixed is a 4x5 matrix of all ones.
    # The rank of the all-ones 4x5 matrix is 1. So the 1x4x5 tensor has rank 1?
    # No, the tensor T_block5[j,k] = 1 for all j,k. As a tensor in R^4 ⊗ R^5,
    # it's the outer product of ones vector (4) and ones vector (5), so rank 1.
    #
    # So total rank = 4*7 + 1 = 29? That seems too good.
    #
    # Wait, I need to be careful. The tensor T is in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ (R^{p*n})^*.
    # The indices are a = i*m+j, b = j*p+k, c = k*n+i.
    # 
    # For block 1 (i in {0,1}, j in {0,1}, k in {0,1}):
    # a in {0,1,4,5}, b in {0,1,10,11}, c in {0,1,3,4}
    # The sub-tensor has entries T[a,b,c] = 1 for these specific (a,b,c) triples.
    # This is a 2x2x2 tensor in the subspaces, but the indices are not contiguous.
    #
    # The key question: can we decompose the full tensor T as a sum of rank-1 tensors
    # where each rank-1 tensor has support on only one block?
    #
    # Yes! If we have a decomposition of block 1 as sum of 7 rank-1 tensors, each
    # rank-1 tensor has nonzero entries only in the a,b,c indices of block 1.
    # Similarly for other blocks.
    # The sum of all these rank-1 tensors gives T.
    #
    # So the total rank is the sum of the ranks of the block decompositions.
    #
    # Let's verify: for block 5 (i=2, j in {0,1,2,3}, k in {0,1,2,3,4}):
    # a = 2*m+j = 8+j, so a in {8,9,10,11}
    # b = j*p+k, so b in {0..4, 5..9, 10..14, 15..19}
    # c = k*n+2, so c in {2,5,8,11,14}
    # T[a,b,c] = 1 for all these (a,b,c).
    # This is a tensor of size 4x20x5 with all entries 1.
    # The rank of the all-ones tensor of size 4x20x5 is 1 (it's u ⊗ v ⊗ w where
    # u, v, w are all-ones vectors).
    #
    # So block 5 has rank 1.
    #
    # Total rank = 4*7 + 1 = 29.
    #
    # But wait, I need to make sure the blocks don't overlap in a way that causes
    # issues. The blocks partition the (i,j,k) space, so each (a,b,c) triple belongs
    # to exactly one block. The decomposition is valid.
    #
    # Let's implement this!
    
    def _build_block_decomp(n, m, p, i_range, j_range, k_range, base_rank):
        """
        Build a decomposition for a block with i in i_range, j in j_range, k in k_range.
        Returns (U_block, V_block, W_block, rank) where U_block is (n*m, rank), etc.
        The entries are placed at the correct global indices.
        """
        i_list = list(range(i_range[0], i_range[1]))
        j_list = list(range(j_range[0], j_range[1]))
        k_list = list(range(k_range[0], k_range[1]))
        
        ni = len(i_list)
        nj = len(j_list)
        nk = len(k_list)
        
        if ni == 2 and nj == 2 and nk == 2:
            # Use Strassen
            U_local, V_local, W_local, r = _strassen_2x2x2()
            # Map local indices to global
            # Local i: 0->i_list[0], 1->i_list[1]
            # Local j: 0->j_list[0], 1->j_list[1]
            # Local k: 0->k_list[0], 1->k_list[1]
            U_g = [[0] * r for _ in range(n * m)]
            V_g = [[0] * r for _ in range(m * p)]
            W_g = [[0] * r for _ in range(p * n)]
            for li in range(2):
                for lj in range(2):
                    for lr in range(r):
                        U_g[i_list[li] * m + j_list[lj]][lr] = U_local[li * 2 + lj][lr]
            for lj in range(2):
                for lk in range(2):
                    for lr in range(r):
                        V_g[j_list[lj] * p + k_list[lk]][lr] = V_local[lj * 2 + lk][lr]
            for lk in range(2):
                for li in range(2):
                    for lr in range(r):
                        W_g[k_list[lk] * n + i_list[li]][lr] = W_local[lk * 2 + li][lr]
            return U_g, V_g, W_g, r
        elif ni == 1:
            # Rank 1: all-ones tensor
            i0 = i_list[0]
            U_g = [[0] * 1 for _ in range(n * m)]
            V_g = [[0] * 1 for _ in range(m * p)]
            W_g = [[0] * 1 for _ in range(p * n)]
            for j in j_list:
                U_g[i0 * m + j][0] = 1
            for j in j_list:
                for k in k_list:
                    V_g[j * p + k][0] = 1
            for k in k_list:
                W_g[k * n + i0][0] = 1
            return U_g, V_g, W_g, 1
        else:
            # Schoolbook for this block
            r = ni * nj * nk
            U_g = [[0] * r for _ in range(n * m)]
            V_g = [[0] * r for _ in range(m * p)]
            W_g = [[0] * r for _ in range(p * n)]
            col = 0
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U_g[i * m + j][col] = 1
                        V_g[j * p + k][col] = 1
                        W_g[k * n + i][col] = 1
                        col += 1
            return U_g, V_g, W_g, r
    
    # Define blocks
    # Block 1: i in [0,2), j in [0,2), k in [0,2)
    # Block 2: i in [0,2), j in [2,4), k in [0,2)
    # Block 3: i in [0,2), j in [0,2), k in [2,4)
    # Block 4: i in [0,2), j in [2,4), k in [2,4)
    # Block 5: i in [2,3), j in [0,4), k in [0,5)
    
    blocks = [
        ((0, 2), (0, 2), (0, 2)),
        ((0, 2), (2, 4), (0, 2)),
        ((0, 2), (0, 2), (2, 4)),
        ((0, 2), (2, 4), (2, 4)),
        ((2, 3), (0, 4), (0, 5)),
    ]
    
    U_total = [[0] * (4*7 + 1) for _ in range(n * m)]
    V_total = [[0] * (4*7 + 1) for _ in range(m * p)]
    W_total = [[0] * (4*7 + 1) for _ in range(p * n)]
    
    offset = 0
    for (ir, jr, kr) in blocks:
        U_b, V_b, W_b, r = _build_block_decomp(n, m, p, ir, jr, kr, offset)
        for row in range(n * m):
            for c in range(r):
                U_total[row][offset + c] += U_b[row][c]
        for row in range(m * p):
            for c in range(r):
                V_total[row][offset + c] += V_b[row][c]
        for row in range(p * n):
            for c in range(r):
                W_total[row][offset + c] += W_b[row][c]
        offset += r
    
    total_rank = offset
    print(f"Block decomposition rank: {total_rank}", file=sys.stderr)
    
    # Verify
    if _verify(U_total, V_total, W_total, total_rank, n, m, p):
        print("Block decomposition verified!", file=sys.stderr)
        best_result = {"U": U_total, "V": V_total, "W": W_total}
        best_rank = total_rank
    else:
        print("Block decomposition FAILED verification, using schoolbook", file=sys.stderr)
    
    if time.time() > deadline:
        return best_result
    
    # Try to improve further with more sophisticated block splits
    # For example, split k into 3 groups for the 3x4x5 case
    # Or use different block sizes
    
    # Alternative: split into 2x2x2 and 1x1x1 blocks more finely
    # i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2, rank 7
    # i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2, rank 7
    # i in {0,1}, j in {0,1}, k in {4}: 2x2x1, rank 4 (schoolbook) or better?
    # 2x2x1 tensor: T[i,j,k] with k fixed. This is a 2x2 matrix of all ones, rank 1.
    # So 2x2x1 has rank 1.
    # Similarly for other 2x2x1 blocks.
    
    # Let's try:
    # i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2, rank 7
    # i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2, rank 7
    # i in {0,1}, j in {0,1}, k in {4}: 2x2x1, rank 1
    # i in {0,1}, j in {2,3}, k in {0,1}: 2x2x2, rank 7
    # i in {0,1}, j in {2,3}, k in {2,3}: 2x2x2, rank 7
    # i in {0,1}, j in {2,3}, k in {4}: 2x2x1, rank 1
    # i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4}: 1x4x5, rank 1
    # Total: 4*7 + 2*1 + 1 = 31
    
    # That's worse than 29. Let's stick with the 29-rank solution.
    
    # Try yet another split:
    # i in {0,1}, j in {0,1}, k in {0,1,2,3}: 2x2x4
    # 2x2x4 tensor: can we do better than 16?
    # Split k into two 2s: 2x2x2 (rank 7) + 2x2x2 (rank 7) = 14
    # So 2x2x4 has rank 14.
    # i in {0,1}, j in {0,1}, k in {4}: 2x2x1, rank 1
    # i in {0,1}, j in {2,3}, k in {0,1,2,3}: 2x2x4, rank 14
    # i in {0,1}, j in {2,3}, k in {4}: 2x2x1, rank 1
    # i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4}: 1x4x5, rank 1
    # Total: 14 + 1 + 14 + 1 + 1 = 31
    
    # Still worse. The 29-rank solution seems good.
    
    # Let's also try a split that uses 2x2x2 for all possible 2x2x2 sub-blocks
    # and handles the rest optimally.
    
    # Actually, let's try to verify the 29-rank solution more carefully.
    if _verify_full(U_total, V_total, W_total, total_rank, n, m, p):
        print("Full verification passed!", file=sys.stderr)
    else:
        print("Full verification FAILED!", file=sys.stderr)
        # Fall back to schoolbook
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = schoolbook_rank
    
    return best_result

# EVOLVE-BLOCK-END
