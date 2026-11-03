"""
Tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
Strategy: Global combinatorial search for the maximum number of disjoint Strassen blocks.
The current program tries single blocks or simple disjoint sets. This program
systematically searches for the maximum set of non-overlapping 2x2x2 Strassen blocks
that can be embedded in the 3x3x8 tensor, maximizing the rank reduction.
"""
import time
import sys
import itertools

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """
    Verify that U, V, W correctly decompose the 3x3x8 matrix multiplication tensor.
    U: (9, rank), V: (24, rank), W: (24, rank)
    Check: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] == 1 if consistent, else 0.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Check all (i,j,k) triples - for the diagonal elements
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > 1e-9:
                    return False, (i, j, k, val, "should be 1")
    
    # Check off-diagonal: T[i*m+j, j'*p+k, k'*n+i'] should be 0 if not consistent
    import random
    random.seed(42)
    for _ in range(500):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        jp = random.randint(0, m-1)
        k = random.randint(0, p-1)
        kp = random.randint(0, p-1)
        ip = random.randint(0, n-1)
        
        expected = 1 if (i==ip and j==jp and k==kp) else 0
        if expected == 0:
            val = 0.0
            for c in range(rank):
                val += U[i*m+j][c] * V[jp*p+k][c] * W[kp*n+ip][c]
            if abs(val) > 1e-9:
                return False, (i, j, jp, k, kp, ip, val, "should be 0")
    
    return True, None


def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C11 += M1
    
    # M2: (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C21 += M2
    W[3][1] = 1  # C22 += M2
    
    # M3: A11(B21-B22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C12 += M3
    W[3][2] = -1 # C22 -= M3
    
    # M4: A22(B12-B11)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1 # C11 -= M4
    W[2][3] = 1  # C21 += M4
    
    # M5: (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1  # C11 += M5
    W[1][4] = 1  # C12 += M5
    
    # M6: (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += M6
    
    # M7: (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += M7
    
    return U, V, W


def _build_decomposition(blocks):
    """
    Build a decomposition using the given list of Strassen blocks.
    blocks: list of (i0, j0, k0) tuples.
    Returns U, V, W lists or None if invalid.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Mark which schoolbook terms are covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    col = i * m + j * p + k
                    covered.add(col)
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2()
    
    # First, add all uncovered schoolbook terms
    current_rank = 0
    for c in range(rank):
        if c not in covered:
            i = c // (m * p)
            rem = c % (m * p)
            j = rem // p
            k = rem % p
            
            U[i * m + j][current_rank] = 1
            V[j * p + k][current_rank] = 1
            W[k * n + i][current_rank] = 1
            current_rank += 1
    
    # Then, add Strassen blocks
    for (i0, j0, k0) in blocks:
        for sc in range(7):
            # U part: 2x2 block at (i0, j0)
            for si in range(2):
                for sj in range(2):
                    idx = si * 2 + sj
                    i = i0 + si
                    j = j0 + sj
                    U[i * m + j][current_rank] = strassen_U[idx][sc]
            
            # V part: 2x2 block at (j0, k0)
            for sj in range(2):
                for sk in range(2):
                    idx = sj * 2 + sk
                    j = j0 + sj
                    k = k0 + sk
                    V[j * p + k][current_rank] = strassen_V[idx][sc]
            
            # W part: 2x2 block at (k0, i0)
            for sk in range(2):
                for si in range(2):
                    idx = sk * 2 + si
                    k = k0 + sk
                    i = i0 + si
                    W[k * n + i][current_rank] = strassen_W[idx][sc]
            
            current_rank += 1
    
    # Trim to actual rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return U, V, W


def construct():
    """
    Construct the tensor decomposition for 3x3x8 matrix multiplication.
    Returns {'U': (9, rank), 'V': (24, rank), 'W': (24, rank)}
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Generate all possible Strassen block positions
    # i0 in {0, 1}, j0 in {0, 1}, k0 in {0, ..., 6}
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
    
    # Check for conflicts between blocks
    # Two blocks (i0, j0, k0) and (i1, j1, k1) conflict if their index sets overlap
    # in any of the three dimensions (i, j, k)
    def blocks_conflict(b1, b2):
        i0, j0, k0 = b1
        i1, j1, k1 = b2
        # i overlap: {i0, i0+1} and {i1, i1+1}
        if i0 == i1 or i0 + 1 == i1 or i1 == i0 or i1 + 1 == i0:
            # j overlap
            if j0 == j1 or j0 + 1 == j1 or j1 == j0 or j1 + 1 == j0:
                # k overlap
                if k0 == k1 or k0 + 1 == k1 or k1 == k0 or k1 + 1 == k0:
                    return True
        return False
    
    # Find the maximum set of non-conflicting blocks
    # Since the number of blocks is small (2*2*7 = 28), we can use a simple search
    # or even brute force for small sets
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try adding blocks greedily and then verify
    # First, try all single blocks
    for b in all_blocks:
        if time.time() > deadline:
            break
        blocks = [b]
        U_new, V_new, W_new = _build_decomposition(blocks)
        rank_new = len(U_new[0])
        ok, err = _verify(U_new, V_new, W_new)
        if ok and rank_new < best_rank:
            best_rank = rank_new
            best_U, best_V, best_W = U_new, V_new, W_new
    
    # Try pairs of non-conflicting blocks
    for i1 in range(len(all_blocks)):
        if time.time() > deadline:
            break
        for i2 in range(i1 + 1, len(all_blocks)):
            if blocks_conflict(all_blocks[i1], all_blocks[i2]):
                continue
            blocks = [all_blocks[i1], all_blocks[i2]]
            U_new, V_new, W_new = _build_decomposition(blocks)
            rank_new = len(U_new[0])
            ok, err = _verify(U_new, V_new, W_new)
            if ok and rank_new < best_rank:
                best_rank = rank_new
                best_U, best_V, best_W = U_new, V_new, W_new
    
    # Try triples of non-conflicting blocks
    for i1 in range(len(all_blocks)):
        if time.time() > deadline:
            break
        for i2 in range(i1 + 1, len(all_blocks)):
            if blocks_conflict(all_blocks[i1], all_blocks[i2]):
                continue
            for i3 in range(i2 + 1, len(all_blocks)):
                if blocks_conflict(all_blocks[i1], all_blocks[i3]) or blocks_conflict(all_blocks[i2], all_blocks[i3]):
                    continue
                blocks = [all_blocks[i1], all_blocks[i2], all_blocks[i3]]
                U_new, V_new, W_new = _build_decomposition(blocks)
                rank_new = len(U_new[0])
                ok, err = _verify(U_new, V_new, W_new)
                if ok and rank_new < best_rank:
                    best_rank = rank_new
                    best_U, best_V, best_W = U_new, V_new, W_new
    
    # Try quadruples of non-conflicting blocks
    for i1 in range(len(all_blocks)):
        if time.time() > deadline:
            break
        for i2 in range(i1 + 1, len(all_blocks)):
            if blocks_conflict(all_blocks[i1], all_blocks[i2]):
                continue
            for i3 in range(i2 + 1, len(all_blocks)):
                if blocks_conflict(all_blocks[i1], all_blocks[i3]) or blocks_conflict(all_blocks[i2], all_blocks[i3]):
                    continue
                for i4 in range(i3 + 1, len(all_blocks)):
                    if (blocks_conflict(all_blocks[i1], all_blocks[i4]) or 
                        blocks_conflict(all_blocks[i2], all_blocks[i4]) or 
                        blocks_conflict(all_blocks[i3], all_blocks[i4])):
                        continue
                    blocks = [all_blocks[i1], all_blocks[i2], all_blocks[i3], all_blocks[i4]]
                    U_new, V_new, W_new = _build_decomposition(blocks)
                    rank_new = len(U_new[0])
                    ok, err = _verify(U_new, V_new, W_new)
                    if ok and rank_new < best_rank:
                        best_rank = rank_new
                        best_U, best_V, best_W = U_new, V_new, W_new
    
    # Try quintuples of non-conflicting blocks
    for i1 in range(len(all_blocks)):
        if time.time() > deadline:
            break
        for i2 in range(i1 + 1, len(all_blocks)):
            if blocks_conflict(all_blocks[i1], all_blocks[i2]):
                continue
            for i3 in range(i2 + 1, len(all_blocks)):
                if blocks_conflict(all_blocks[i1], all_blocks[i3]) or blocks_conflict(all_blocks[i2], all_blocks[i3]):
                    continue
                for i4 in range(i3 + 1, len(all_blocks)):
                    if (blocks_conflict(all_blocks[i1], all_blocks[i4]) or 
                        blocks_conflict(all_blocks[i2], all_blocks[i4]) or 
                        blocks_conflict(all_blocks[i3], all_blocks[i4])):
                        continue
                    for i5 in range(i4 + 1, len(all_blocks)):
                        if (blocks_conflict(all_blocks[i1], all_blocks[i5]) or 
                            blocks_conflict(all_blocks[i2], all_blocks[i5]) or 
                            blocks_conflict(all_blocks[i3], all_blocks[i5]) or 
                            blocks_conflict(all_blocks[i4], all_blocks[i5])):
                            continue
                        blocks = [all_blocks[i1], all_blocks[i2], all_blocks[i3], all_blocks[i4], all_blocks[i5]]
                        U_new, V_new, W_new = _build_decomposition(blocks)
                        rank_new = len(U_new[0])
                        ok, err = _verify(U_new, V_new, W_new)
                        if ok and rank_new < best_rank:
                            best_rank = rank_new
                            best_U, best_V, best_W = U_new, V_new, W_new
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
