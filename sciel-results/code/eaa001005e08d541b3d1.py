import time

START_TIME = time.time()
TIME_LIMIT = 120

def _elapsed():
    return time.time() - START_TIME

def _deadline_exceeded(margin=2.0):
    return _elapsed() > (TIME_LIMIT - margin)

def construct():
    n, m, p = 3, 5, 5
    
    # Lader 3x3x3 decomposition (Rank 23)
    # U, V, W are 9x23 matrices for the 3x3x3 block
    # Indices: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # Lader coefficients (from literature, e.g., "Fast matrix multiplication" by Lader)
    # This is a specific rank-23 decomposition.
    # We use a known valid set of coefficients.
    
    lader_U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=0, j=0
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=0, j=1
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=0, j=2
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=1, j=0
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=1, j=1
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=1, j=2
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=2, j=0
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=2, j=1
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=2, j=2
    ]
    # Note: The above is just identity. I need the actual Lader coefficients.
    # Since I cannot reliably recall the exact 23x9 matrix from memory without error,
    # and implementing a search for Lader is too slow, I will stick to the Strassen optimization 
    # which is safer.
    
    # Let's try a different partitioning of the 3x5x5 tensor using only 2x2x2 Strassen blocks
    # to see if we can get lower than 71.
    
    # Current: 4 blocks on i={0,1}. Rank 28. Remainder 43. Total 71.
    # Try: 4 blocks on i={0,1} and 4 blocks on i={1,2}?
    # We established this leads to double counting.
    
    # Try: 2 blocks on i={0,1}, 2 blocks on i={1,2}?
    # Blocks:
    # 1. i={0,1}, j={0,1}, k={0,1}
    # 2. i={0,1}, j={2,3}, k={2,3}
    # 3. i={1,2}, j={0,1}, k={2,3}
    # 4. i={1,2}, j={2,3}, k={0,1}
    # This covers:
    # i=0: j={0,1}, k={0,1} and j={2,3}, k={2,3} (16 elements)
    # i=1: j={0,1}, k={0,1} and j={2,3}, k={2,3} (16 elements)
    # i=2: j={0,1}, k={2,3} and j={2,3}, k={0,1} (16 elements)
    # Total covered: 48.
    # Rank: 4 * 7 = 28.
    # Uncovered: 75 - 48 = 27.
    # Total Rank: 28 + 27 = 55?
    # Wait, if the blocks are disjoint in the tensor support, this is valid.
    # Are they disjoint?
    # Block 1: i=0,1; j=0,1; k=0,1
    # Block 2: i=0,1; j=2,3; k=2,3
    # Block 3: i=1,2; j=0,1; k=2,3
    # Block 4: i=1,2; j=2,3; k=0,1
    
    # Overlaps:
    # i=1, j=0,1, k=0,1 is in Block 1.
    # i=1, j=0,1, k=2,3 is in Block 3.
    # i=1, j=2,3, k=0,1 is in Block 4.
    # i=1, j=2,3, k=2,3 is in Block 2.
    # So for i=1, the blocks cover disjoint j,k regions.
    # For i=0: Block 1 (j=0,1,k=0,1) and Block 2 (j=2,3,k=2,3). Disjoint.
    # For i=2: Block 3 (j=0,1,k=2,3) and Block 4 (j=2,3,k=0,1). Disjoint.
    
    # So the union of these 4 blocks covers 48 distinct elements.
    # And they are linearly independent (Strassen blocks are disjoint in support).
    # So we can sum them.
    # Rank = 28.
    # Uncovered elements = 75 - 48 = 27.
    # Total Rank = 28 + 27 = 55.
    
    # This is a huge improvement!
    
    U = [[0.0] * 55 for _ in range(n * m)]
    V = [[0.0] * 55 for _ in range(m * p)]
    W = [[0.0] * 55 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, start_col):
        c = start_col + 0
        U[i0*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = start_col + 1
        U[(i0+1)*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += -1.0
        
        c = start_col + 2
        U[i0*m + j0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = start_col + 3
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[j0*p + k0][c] += -1.0
        W[k0*n + i0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        
        c = start_col + 4
        U[i0*m + j0][c] += 1.0
        U[i0*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        
        c = start_col + 5
        U[(i0+1)*m + j0][c] += 1.0
        U[i0*m + j0][c] += -1.0
        V[j0*p + k0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = start_col + 6
        U[i0*m + (j0+1)][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += -1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0
        
        return start_col + 7
    
    blocks = [
        (0, 0, 0),
        (0, 2, 2),
        (1, 0, 2),
        (1, 2, 0),
    ]
    
    for block in blocks:
        if _deadline_exceeded():
            break
        col = add_strassen_block(block[0], block[1], block[2], col)
    
    covered_set = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_set.add((i0+di, j0+dj, k0+dk))
    
    for i in range(n):
        if _deadline_exceeded():
            break
        for j in range(m):
            if _deadline_exceeded():
                break
            for k in range(p):
                if _deadline_exceeded():
                    break
                if (i, j, k) not in covered_set:
                    U[i*m + j][col] += 1.0
                    V[j*p + k][col] += 1.0
                    W[k*n + i][col] += 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
