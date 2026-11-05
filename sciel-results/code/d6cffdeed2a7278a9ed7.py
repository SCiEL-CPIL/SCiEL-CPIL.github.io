import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-8):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

# Strassen rank-7 product for 2x2 matrices
# U[i*2+j, c] = a[i,j] for i,j in 0..1
# V[j*2+k, c] = b[j,k]
# W[k*2+i, c] = c[k,i]
# The 7 Strassen products:
# M1 = (A11+A22)(B11+B22)
# M2 = (A21+A22)B11
# M3 = A11(B21-B22)
# M4 = A22(B21-B22)  -- actually A22*(B12-B11) in standard, let me use standard Strassen
# Standard Strassen:
# M1 = (A11+A22)(B11+B22)
# M2 = (A21+A22)B11
# M3 = A11(B21-B22)
# M4 = A22(B12-B11)
# M5 = (A11+A12)B22
# M6 = (A21-A11)(B11+B12)
# M7 = (A12-A22)(B21+B22)
# C11 = M1 + M4 - M5 - M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M7

def _strassen_2x2_block(i0, j0, k0, U, V, W, col):
    """Add rank-7 Strassen decomposition for block at (i0,j0,k0).
    U rows: (i0+i, j0+j) for i,j in 0,1 -> row index (i0+i)*2 + (j0+j) in local, 
    but we need global row indices. The block covers rows i0..i0+1 of A (size 2x2 local).
    Actually for a 2x2 block of A at rows i0:i0+2, cols j0:j0+2, and B at rows j0:j0+2, cols k0:k0+2,
    and C at rows k0:k0+2, cols i0:i0+2.
    """
    # For each of the 7 Strassen products, add one rank-1 term
    # We need to map local (i,j) to global row index in U: (i0+i)*m + (j0+j)
    # But m is the full width. Let's pass m.
    
    # Actually, let me restructure. The block is 2x2 in each dimension.
    # A block: rows i0..i0+1, cols j0..j0+1 (local 2x2)
    # B block: rows j0..j0+1, cols k0..k0+1
    # C block: rows k0..k0+1, cols i0..i0+1
    
    # Strassen formulas for C = A*B where A,B are 2x2:
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # The 7 products and their contributions:
    # Each product Mx is a scalar that multiplies a combination of A entries and B entries.
    # For the tensor decomposition, each rank-1 term is U_vec * V_vec * W_vec.
    # For Strassen, each of the 7 terms corresponds to a specific linear combination.
    
    # Let me define the 7 terms explicitly:
    # Term 1: M1 = (a11+a22)(b11+b22)
    #   U: a11 at (i0,j0), a22 at (i0+1,j0+1)
    #   V: b11 at (j0,k0), b22 at (j0+1,k0+1)
    #   W: contributes to c11 (+1), c12 (0), c21 (0), c22 (-1)
    #     c11 is at (k0, i0), c22 is at (k0+1, i0+1)
    
    # This is getting complex. Let me just enumerate all 7 terms.
    
    terms = []
    
    # M1 = (a11+a22)(b11+b22)
    # U: {(i0,j0):1, (i0+1,j0+1):1}
    # V: {(j0,k0):1, (j0+1,k0+1):1}
    # W: {(k0,i0):1, (k0+1,i0+1):-1}
    terms.append((
        [(i0, j0, 1.0), (i0+1, j0+1, 1.0)],
        [(j0, k0, 1.0), (j0+1, k0+1, 1.0)],
        [(k0, i0, 1.0), (k0+1, i0+1, -1.0)]
    ))
    
    # M2 = (a21+a22)(b11)
    # U: {(i0+1,j0):1, (i0+1,j0+1):1}
    # V: {(j0,k0):1}
    # W: contributes to c21 (+1)
    #   c21 is at (k0, i0+1)
    terms.append((
        [(i0+1, j0, 1.0), (i0+1, j0+1, 1.0)],
        [(j0, k0, 1.0)],
        [(k0, i0+1, 1.0)]
    ))
    
    # M3 = a11(b21-b22)
    # U: {(i0,j0):1}
    # V: {(j0+1,k0):1, (j0+1,k0+1):-1}
    # W: contributes to c12 (+1)
    #   c12 is at (k0+1, i0)
    terms.append((
        [(i0, j0, 1.0)],
        [(j0+1, k0, 1.0), (j0+1, k0+1, -1.0)],
        [(k0+1, i0, 1.0)]
    ))
    
    # M4 = a22(b12-b11)
    # U: {(i0+1,j0+1):1}
    # V: {(j0,k0+1):1, (j0,k0):-1}
    # W: contributes to c11 (+1)
    #   c11 is at (k0, i0)
    terms.append((
        [(i0+1, j0+1, 1.0)],
        [(j0, k0+1, 1.0), (j0, k0, -1.0)],
        [(k0, i0, 1.0)]
    ))
    
    # M5 = (a11+a12)(b22)
    # U: {(i0,j0):1, (i0,j0+1):1}
    # V: {(j0+1,k0+1):1}
    # W: contributes to c11 (-1)
    #   c11 is at (k0, i0)
    terms.append((
        [(i0, j0, 1.0), (i0, j0+1, 1.0)],
        [(j0+1, k0+1, 1.0)],
        [(k0, i0, -1.0)]
    ))
    
    # M6 = (a21-a11)(b11+b12)
    # U: {(i0+1,j0):1, (i0,j0):-1}
    # V: {(j0,k0):1, (j0,k0+1):1}
    # W: contributes to c22 (-1)
    #   c22 is at (k0+1, i0+1)
    terms.append((
        [(i0+1, j0, 1.0), (i0, j0, -1.0)],
        [(j0, k0, 1.0), (j0, k0+1, 1.0)],
        [(k0+1, i0+1, -1.0)]
    ))
    
    # M7 = (a12-a22)(b21+b22)
    # U: {(i0,j0+1):1, (i0+1,j0+1):-1}
    # V: {(j0+1,k0):1, (j0+1,k0+1):1}
    # W: contributes to c11 (+1)
    #   c11 is at (k0, i0)
    terms.append((
        [(i0, j0+1, 1.0), (i0+1, j0+1, -1.0)],
        [(j0+1, k0, 1.0), (j0+1, k0+1, 1.0)],
        [(k0, i0, 1.0)]
    ))
    
    # Now add these to U, V, W
    for u_terms, v_terms, w_terms in terms:
        for (i, j, val) in u_terms:
            U[i * 2 + j, col] += val  # local 2x2 indexing
        for (j, k, val) in v_terms:
            V[j * 2 + k, col] += val
        for (k, i, val) in w_terms:
            W[k * 2 + i, col] += val
        col += 1
    
    return col

def _block_strassen(n, m, p):
    """
    Block Strassen for 3x5 x 5x6.
    Split into blocks of size 2x2 where possible.
    
    A is 3x5, B is 5x6, C is 3x6.
    
    Strategy:
    - For the 2x2 blocks where all three dimensions have at least 2 elements, use Strassen (rank 7 instead of 8).
    - For remaining 1x* or *x1 blocks, use schoolbook.
    
    Let's partition:
    A: rows 0-1 (2 rows), row 2 (1 row); cols 0-3 (4 cols -> two 2x2 blocks), col 4 (1 col)
    B: rows 0-3 (4 rows -> two 2x2 blocks), row 4 (1 row); cols 0-5 (6 cols -> three 2x2 blocks)
    C: rows 0-1 (2 rows), row 2 (1 row); cols 0-5 (6 cols -> three 2x2 blocks)
    
    Actually, let me think about this more carefully.
    
    The tensor T[i,j,k] = 1 for all i in 0..2, j in 0..4, k in 0..5.
    
    We can partition the indices into blocks:
    - Block type (2,2,2): use Strassen, rank 7
    - Block type (1,2,2) or (2,1,2) or (2,2,1) or (1,1,2) etc: use schoolbook
    
    Let me partition:
    i: {0,1} (size 2), {2} (size 1)
    j: {0,1} (size 2), {2,3} (size 2), {4} (size 1)
    k: {0,1} (size 2), {2,3} (size 2), {4,5} (size 2)
    
    Blocks:
    - (i={0,1}, j={0,1}, k={0,1}): 2x2x2, Strassen, rank 7
    - (i={0,1}, j={0,1}, k={2,3}): 2x2x2, Strassen, rank 7
    - (i={0,1}, j={0,1}, k={4,5}): 2x2x2, Strassen, rank 7
    - (i={0,1}, j={2,3}, k={0,1}): 2x2x2, Strassen, rank 7
    - (i={0,1}, j={2,3}, k={2,3}): 2x2x2, Strassen, rank 7
    - (i={0,1}, j={2,3}, k={4,5}): 2x2x2, Strassen, rank 7
    - (i={0,1}, j={4}, k={0,1}): 2x1x2, schoolbook, rank 4
    - (i={0,1}, j={4}, k={2,3}): 2x1x2, schoolbook, rank 4
    - (i={0,1}, j={4}, k={4,5}): 2x1x2, schoolbook, rank 4
    - (i={2}, j={0,1}, k={0,1}): 1x2x2, schoolbook, rank 4
    - (i={2}, j={0,1}, k={2,3}): 1x2x2, schoolbook, rank 4
    - (i={2}, j={0,1}, k={4,5}): 1x2x2, schoolbook, rank 4
    - (i={2}, j={2,3}, k={0,1}): 1x2x2, schoolbook, rank 4
    - (i={2}, j={2,3}, k={2,3}): 1x2x2, schoolbook, rank 4
    - (i={2}, j={2,3}, k={4,5}): 1x2x2, schoolbook, rank 4
    - (i={2}, j={4}, k={0,1}): 1x1x2, schoolbook, rank 2
    - (i={2}, j={4}, k={2,3}): 1x1x2, schoolbook, rank 2
    - (i={2}, j={4}, k={4,5}): 1x1x2, schoolbook, rank 2
    
    Total rank: 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
    
    Schoolbook is 3*5*6 = 90. So we save 6.
    """
    n, m, p = 3, 5, 6
    total_rank = 84
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Define the blocks
    # i_blocks: [0,1] and [2]
    # j_blocks: [0,1], [2,3], [4]
    # k_blocks: [0,1], [2,3], [4,5]
    
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4, 5]]
    
    for i_block in i_blocks:
        for j_block in j_blocks:
            for k_block in k_blocks:
                ni = len(i_block)
                nj = len(j_block)
                nk = len(k_block)
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen
                    i0 = i_block[0]
                    j0 = j_block[0]
                    k0 = k_block[0]
                    
                    # Local 2x2x2 Strassen
                    # We need to add 7 rank-1 terms
                    # U local: 2x2, V local: 2x2, W local: 2x2
                    # Map local (i,j) to global: (i0+i, j0+j) -> row (i0+i)*m + (j0+j)
                    # Map local (j,k) to global: (j0+j, k0+k) -> row (j0+j)*p + (k0+k)
                    # Map local (k,i) to global: (k0+k, i0+i) -> row (k0+k)*n + (i0+i)
                    
                    terms = [
                        # M1
                        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,-1.0)]),
                        # M2
                        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(0,1,1.0)]),
                        # M3
                        ([(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)], [(1,0,1.0)]),
                        # M4
                        ([(1,1,1.0)], [(0,1,1.0),(0,0,-1.0)], [(0,0,1.0)]),
                        # M5
                        ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0)]),
                        # M6
                        ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,-1.0)]),
                        # M7
                        ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(0,0,1.0)]),
                    ]
                    
                    for u_terms, v_terms, w_terms in terms:
                        for (li, lj, val) in u_terms:
                            gi = i0 + li
                            gj = j0 + lj
                            U[gi * m + gj, col] += val
                        for (lj, lk, val) in v_terms:
                            gj = j0 + lj
                            gk = k0 + lk
                            V[gj * p + gk, col] += val
                        for (lk, li, val) in w_terms:
                            gk = k0 + lk
                            gi = i0 + li
                            W[gk * n + gi, col] += val
                        col += 1
                else:
                    # Schoolbook for this block
                    for ii in i_block:
                        for jj in j_block:
                            for kk in k_block:
                                U[ii * m + jj, col] = 1.0
                                V[jj * p + kk, col] = 1.0
                                W[kk * n + ii, col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Expected rank {total_rank}, got {col}"
    
    # Verify
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If verification fails, fall back to schoolbook
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    # Try block Strassen
    result = _block_strassen(n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
