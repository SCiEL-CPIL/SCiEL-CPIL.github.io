"""
Tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
Strategy: Hierarchical Strassen decomposition exploiting the rectangular p=8 dimension.
Group the 8 columns into 4 pairs of 2. Apply Strassen-style decomposition to the
column pairing, reducing to a 3x3x4 problem, then apply 2x2x2 Strassen blocks
in the row/column dimensions.

The key insight: for 3x3x8, we can pair columns (k, k+1) for k in {0,2,4,6}.
Each pair can be handled with a Strassen-like 2-product decomposition in the
column dimension, combined with the 3x3 structure.

Actually, let's think more carefully. The tensor is T[(i,j),(j',k),(k',i')] 
= delta_{i,i'} delta_{j,j'} delta_{k,k'}.

For 2x2x2 Strassen, we replace 8 scalar multiplications with 7.
For our 3x3x8 problem, schoolbook rank is 3*3*8 = 72.

A valid approach: use the fact that p=8 is even, so we can group columns into pairs.
For each pair of columns (k, k+1), and for each (i,j), the tensor element involves
delta_{k,k'} which couples to W[k*n+i].

Let me try a different structural approach: use a 2x2x4 Strassen-like block.
Actually, let me just try to find a valid decomposition by exploiting the
rectangular structure more carefully.

For the 3x3x8 tensor, consider splitting into blocks. The key is that the
tensor factors as a sum of rank-1 terms. We need to find the minimum number.

Let me try: for each i, j, k we need U[i*3+j, c] * V[j*8+k, c] * W[k*3+i, c] 
to sum to delta_{i,i'}delta_{j,j'}delta_{k,k'}.

A valid decomposition of rank 72 is the schoolbook. To go lower, we need
cancellations.

Let me try a structured approach: group (i,j) pairs and (k) pairs.
For 2x2x2 Strassen embedded in specific positions, we can save 1 rank per block.

Let me implement a more systematic search: try all possible 2x2x2 Strassen
embeddings simultaneously (non-overlapping) and verify.
"""
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _verify_full(U, V, W):
    """Full verification of the tensor decomposition."""
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = len(U[0])
    
    # Check all diagonal elements
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check all off-diagonal elements (full check)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                continue  # already checked
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j][c] * V[jp*p+k][c] * W[kp*n+ip][c]
                            if abs(val) > 1e-9:
                                return False
    
    return True


def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2x2.
    Returns U, V, W each of shape (4, 7).
    U maps 2x2 input A, V maps 2x2 input B, W maps 2x2 output C.
    """
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1... 
    
    # Wait, let me be more careful with Strassen.
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M6
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11
    
    # M2 = (A10+A11)(B00)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1   # C10
    W[3][1] = 1   # C11
    
    # M3 = (A00)(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1   # C01
    W[3][2] = -1  # C11
    
    # M4 = (A11)(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1  # C00
    W[2][3] = 1   # C10
    
    # M5 = (A00+A01)(B11)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1   # C00
    W[1][4] = 1   # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C00
    
    return U, V, W


def _try_single_strassen_block(i0, j0, k0, n=3, m=3, p=8):
    """
    Try embedding a single 2x2x2 Strassen block at position (i0, j0, k0).
    Returns (U, V, W, rank) if valid, else None.
    """
    sU, sV, sW = _strassen_2x2x2()
    
    # Schoolbook rank for the full problem
    full_rank = n * m * p
    
    # Create schoolbook decomposition
    U = [[0.0]*full_rank for _ in range(n*m)]
    V = [[0.0]*full_rank for _ in range(m*p)]
    W = [[0.0]*full_rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    # Identify the 8 schoolbook columns to replace
    remove_cols = set()
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                c = i * m + j * p + k
                remove_cols.add(c)
    
    # Build new decomposition
    new_rank = full_rank - 8 + 7
    new_U = [[0.0]*new_rank for _ in range(n*m)]
    new_V = [[0.0]*new_rank for _ in range(m*p)]
    new_W = [[0.0]*new_rank for _ in range(p*n)]
    
    # Copy non-removed columns
    new_col = 0
    for c in range(full_rank):
        if c not in remove_cols:
            for r in range(n*m):
                new_U[r][new_col] = U[r][c]
            for r in range(m*p):
                new_V[r][new_col] = V[r][c]
            for r in range(p*n):
                new_W[r][new_col] = W[r][c]
            new_col += 1
    
    # Add Strassen block
    base = new_col
    for sc in range(7):
        for di in range(2):
            for dj in range(2):
                idx = di*2 + dj
                i = i0 + di
                j = j0 + dj
                new_U[i*m+j][base+sc] = sU[idx][sc]
        for dj in range(2):
            for dk in range(2):
                idx = dj*2 + dk
                j = j0 + dj
                k = k0 + dk
                new_V[j*p+k][base+sc] = sV[idx][sc]
        for dk in range(2):
            for di in range(2):
                idx = dk*2 + di
                k = k0 + dk
                i = i0 + di
                new_W[k*n+i][base+sc] = sW[idx][sc]
    
    # Verify
    if _verify_full(new_U, new_V, new_W):
        return new_U, new_V, new_W, new_rank
    return None


def _try_multiple_strassen_blocks(blocks, n=3, m=3, p=8):
    """
    Try embedding multiple non-overlapping 2x2x2 Strassen blocks.
    blocks: list of (i0, j0, k0)
    """
    full_rank = n * m * p
    num_blocks = len(blocks)
    new_rank = full_rank - 8*num_blocks + 7*num_blocks
    
    # Check for overlaps
    covered = set()
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    key = (i0+di, j0+dj, k0+dk)
                    if key in covered:
                        return None  # Overlapping blocks
                    covered.add(key)
    
    sU, sV, sW = _strassen_2x2x2()
    
    U = [[0.0]*full_rank for _ in range(n*m)]
    V = [[0.0]*full_rank for _ in range(m*p)]
    W = [[0.0]*full_rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    # Build new decomposition
    new_U = [[0.0]*new_rank for _ in range(n*m)]
    new_V = [[0.0]*new_rank for _ in range(m*p)]
    new_W = [[0.0]*new_rank for _ in range(p*n)]
    
    remove_cols = set()
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    remove_cols.add(i * m + j * p + k)
    
    new_col = 0
    for c in range(full_rank):
        if c not in remove_cols:
            for r in range(n*m):
                new_U[r][new_col] = U[r][c]
            for r in range(m*p):
                new_V[r][new_col] = V[r][c]
            for r in range(p*n):
                new_W[r][new_col] = W[r][c]
            new_col += 1
    
    # Add each Strassen block
    for i0, j0, k0 in blocks:
        for sc in range(7):
            for di in range(2):
                for dj in range(2):
                    idx = di*2 + dj
                    i = i0 + di
                    j = j0 + dj
                    new_U[i*m+j][new_col+sc] = sU[idx][sc]
            for dj in range(2):
                for dk in range(2):
                    idx = dj*2 + dk
                    j = j0 + dj
                    k = k0 + dk
                    new_V[j*p+k][new_col+sc] = sV[idx][sc]
            for dk in range(2):
                for di in range(2):
                    idx = dk*2 + di
                    k = k0 + dk
                    i = i0 + di
                    new_W[k*n+i][new_col+sc] = sW[idx][sc]
        new_col += 7
    
    if _verify_full(new_U, new_V, new_W):
        return new_U, new_V, new_W, new_rank
    return None


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Start with schoolbook
    full_rank = n * m * p
    U = [[0.0]*full_rank for _ in range(n*m)]
    V = [[0.0]*full_rank for _ in range(m*p)]
    W = [[0.0]*full_rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    best_rank = full_rank
    best_U, best_V, best_W = U, V, W
    
    # Try single Strassen blocks
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                if time.time() > deadline - 10:
                    break
                result = _try_single_strassen_block(i0, j0, k0, n, m, p)
                if result is not None:
                    rU, rV, rW, rrank = result
                    if rrank < best_rank:
                        best_rank = rrank
                        best_U, best_V, best_W = rU, rV, rW
    
    # Try pairs of non-overlapping Strassen blocks
    all_blocks = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    for idx1 in range(len(all_blocks)):
        for idx2 in range(idx1+1, len(all_blocks)):
            if time.time() > deadline - 5:
                break
            b1 = all_blocks[idx1]
            b2 = all_blocks[idx2]
            # Check overlap
            covered1 = set()
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered1.add((b1[0]+di, b1[1]+dj, b1[2]+dk))
            overlap = False
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (b2[0]+di, b2[1]+dj, b2[2]+dk) in covered1:
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    break
            if overlap:
                continue
            
            result = _try_multiple_strassen_blocks([b1, b2], n, m, p)
            if result is not None:
                rU, rV, rW, rrank = result
                if rrank < best_rank:
                    best_rank = rrank
                    best_U, best_V, best_W = rU, rV, rW
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
