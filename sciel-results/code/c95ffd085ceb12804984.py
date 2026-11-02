import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that U, V, W form a valid tensor decomposition of the n x m x p matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Trace convention: W[k*n + i, r]
    """
    T = np.ones((n, m, p))
    T_decomp = np.zeros((n, m, p))
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    return np.allclose(T, T_decomp, atol=1e-9)

def _schoolbook():
    """Generate the standard schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _strassen_block(U, V, W, base_col, i_off, j_off, k_off, n, m, p):
    """
    Add a 2x2x2 Strassen block at the specified offset.
    Returns the next column index.
    
    U index: (i_off + di) * m + (j_off + dj)
    V index: (j_off + dj) * p + (k_off + dk)
    W index: (k_off + dk) * n + (i_off + di)
    
    Strassen 7 terms:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # We need to add 7 columns
    col = base_col
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11 and A22 -> U[i_off*m + j_off, col] = 1, U[(i_off+1)*m + (j_off+1), col] = 1
    # V: B11 and B12... wait, B11 is B[j1,k1], B22 is B[j2,k2]
    # Actually for Strassen on C = A*B:
    # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
    # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1 = (A11+A22)(B11+B22)
    U[(i_off) * m + (j_off), col] = 1
    U[(i_off+1) * m + (j_off+1), col] = 1
    V[(j_off) * p + (k_off), col] = 1
    V[(j_off+1) * p + (k_off+1), col] = 1
    # W contribution: C11 += M1, C22 += M1
    # C11 -> W[k_off*n + i_off, col] += 1
    # C22 -> W[(k_off+1)*n + (i_off+1), col] += 1
    W[k_off * n + i_off, col] = 1
    W[(k_off+1) * n + (i_off+1), col] = 1
    col += 1
    
    # M2 = (A21+A22)B11
    U[(i_off+1) * m + (j_off), col] = 1
    U[(i_off+1) * m + (j_off+1), col] = 1
    V[(j_off) * p + (k_off), col] = 1
    # W: C21 += M2, C22 -= M2
    W[k_off * n + (i_off+1), col] = 1
    W[(k_off+1) * n + (i_off+1), col] -= 1
    col += 1
    
    # M3 = A11(B21-B22)
    U[(i_off) * m + (j_off), col] = 1
    V[(j_off+1) * p + (k_off), col] = 1
    V[(j_off+1) * p + (k_off+1), col] = -1
    # W: C12 += M3, C22 += M3
    W[(k_off) * n + i_off, col] = 1
    W[(k_off+1) * n + (i_off+1), col] = 1
    col += 1
    
    # M4 = A22(B12-B11)
    U[(i_off+1) * m + (j_off+1), col] = 1
    V[(j_off) * p + (k_off+1), col] = 1
    V[(j_off) * p + (k_off), col] = -1
    # W: C11 += M4, C21 += M4
    W[k_off * n + i_off, col] = 1
    W[k_off * n + (i_off+1), col] = 1
    col += 1
    
    # M5 = (A11+A12)B22
    U[(i_off) * m + (j_off), col] = 1
    U[(i_off) * m + (j_off+1), col] = 1
    V[(j_off+1) * p + (k_off+1), col] = 1
    # W: C12 += M5, C11 -= M5
    W[(k_off) * n + i_off, col] = 1
    W[k_off * n + i_off, col] -= 1
    col += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[(i_off+1) * m + (j_off), col] = 1
    U[(i_off) * m + (j_off), col] = -1
    V[(j_off) * p + (k_off), col] = 1
    V[(j_off) * p + (k_off+1), col] = 1
    # W: C22 += M6
    W[(k_off+1) * n + (i_off+1), col] = 1
    col += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[(i_off) * m + (j_off+1), col] = 1
    U[(i_off+1) * m + (j_off+1), col] = -1
    V[(j_off+1) * p + (k_off), col] = 1
    V[(j_off+1) * p + (k_off+1), col] = 1
    # W: C11 += M7
    W[k_off * n + i_off, col] = 1
    col += 1
    
    return col

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the tensor decomposition of 3x5x6 matrix multiplication.
    Uses Strassen for 2x2 blocks where possible to reduce rank below schoolbook.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank 90
    # Try to replace 2x2x2 blocks with Strassen (rank 7 instead of 8)
    
    # Identify possible 2x2x2 blocks:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # i in {0,1}, j in {2,3}, k in {2,3}
    # i in {1,2}, j in {0,1}, k in {0,1}
    # etc.
    
    # Let's be systematic. We'll build the decomposition by processing
    # each (i,j,k) triple. If a 2x2x2 block is available, use Strassen.
    # Otherwise, use schoolbook for that triple.
    
    # Strategy: greedily place Strassen blocks where they fit.
    # A Strassen block at (i0, j0, k0) covers:
    # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # Requires i0+1 < n, j0+1 < m, k0+1 < p
    
    # For n=3, m=5, p=6:
    # Possible i0: 0, 1 (since i0+1 < 3)
    # Possible j0: 0, 1, 2, 3 (since j0+1 < 5)
    # Possible k0: 0, 1, 2, 3, 4 (since k0+1 < 6)
    
    # We want to maximize the number of non-overlapping Strassen blocks.
    # Each Strassen block saves 1 rank (8 - 7 = 1).
    
    # Let's try to pack as many as possible.
    # Block at (i0, j0, k0) uses indices:
    # U: (i0*m+j0), (i0*m+j0+1), ((i0+1)*m+j0), ((i0+1)*m+j0+1)
    # V: (j0*p+k0), (j0*p+k0+1), ((j0+1)*p+k0), ((j0+1)*p+k0+1)
    # W: (k0*n+i0), (k0*n+i0+1), ((k0+1)*n+i0), ((k0+1)*n+i0+1)
    
    # Two blocks don't overlap if they don't share any U, V, or W indices.
    
    # Let's try a simple greedy approach:
    # Sort all possible blocks and pick non-overlapping ones.
    
    possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                possible_blocks.append((i0, j0, k0))
    
    # Try to find a set of non-overlapping blocks
    # For simplicity, let's try a few specific configurations
    
    # Configuration 1: 2 blocks
    # Block A: i0=0, j0=0, k0=0
    # Block B: i0=0, j0=2, k0=2
    # Check overlap:
    # A U: 0,1,5,6
    # B U: 2,3,7,8
    # No U overlap.
    # A V: 0,1,6,7
    # B V: 14,15,20,21
    # No V overlap.
    # A W: 0,1,3,4
    # B W: 6,7,9,10
    # No W overlap.
    # This works!
    
    # Can we add more blocks?
    # Block C: i0=1, j0=0, k0=0
    # C U: 5,6,10,11
    # Overlaps with A U: 5,6. No good.
    
    # Block C: i0=1, j0=2, k0=2
    # C U: 7,8,12,13
    # Overlaps with B U: 7,8. No good.
    
    # Block C: i0=0, j0=0, k0=2
    # C U: 0,1,5,6
    # Overlaps with A U. No good.
    
    # Block C: i0=1, j0=0, k0=2
    # C U: 5,6,10,11
    # C V: 12,13,18,19
    # C W: 6,7,9,10
    # A U: 0,1,5,6 - overlaps with C U at 5,6. No good.
    
    # Let's try a different configuration.
    # Block A: i0=0, j0=0, k0=0
    # Block B: i0=1, j0=1, k0=1
    # A U: 0,1,5,6
    # B U: 6,7,11,12
    # Overlap at 6. No good.
    
    # Block A: i0=0, j0=0, k0=0
    # Block B: i0=1, j0=3, k0=3
    # A U: 0,1,5,6
    # B U: 8,9,13,14
    # No U overlap.
    # A V: 0,1,6,7
    # B V: 21,22,27,28
    # No V overlap.
    # A W: 0,1,3,4
    # B W: 9,10,12,13
    # No W overlap.
    # This also works!
    
    # Can we add a third block?
    # Block C: i0=0, j0=3, k0=0
    # C U: 3,4,8,9
    # A U: 0,1,5,6 - no overlap
    # B U: 8,9,13,14 - overlaps at 8,9. No good.
    
    # Block C: i0=0, j0=1, k0=3
    # C U: 1,2,6,7
    # A U: 0,1,5,6 - overlaps at 1,6. No good.
    
    # Let's try 3 blocks with different placement.
    # Block A: i0=0, j0=0, k0=0
    # Block B: i0=0, j0=3, k0=2
    # Block C: i0=1, j0=1, k0=3
    # A U: 0,1,5,6
    # B U: 3,4,8,9
    # C U: 6,7,11,12
    # A-C overlap at 6. No good.
    
    # It seems hard to get more than 2 non-overlapping blocks.
    # Let's go with 2 Strassen blocks, giving rank 88.
    
    rank = 90 - 2  # 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First, place the two Strassen blocks
    col = _strassen_block(U, V, W, 0, 0, 0, 0, n, m, p)  # Block at (0,0,0), uses cols 0-6
    col = _strassen_block(U, V, W, col, 0, 2, 2, n, m, p)  # Block at (0,2,2), uses cols 7-13
    
    # Now fill the rest with schoolbook
    # We need to track which (i,j,k) triples are covered by Strassen blocks
    # Block 1 covers: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2 covers: i in {0,1}, j in {2,3}, k in {2,3}
    
    covered = set()
    # Block 1
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                covered.add((0 + di, 0 + dj, 0 + dk))
    # Block 2
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                covered.add((0 + di, 2 + dj, 2 + dk))
    
    # Fill remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
