"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U,V,W (each 4x7) for 2x2 matrix multiplication using Strassen (rank 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    # Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: A00->0, A01->1, A10->2, A11->3
    # B indices: B00->0, B01->1, B10->2, B11->3
    # C indices: C00->0, C01->1, C10->2, C11->3
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return U,V,W for n x m times m x p schoolbook multiplication.
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p)."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _concat_blocks(blocks):
    """Concatenate a list of (U,V,W) blocks horizontally.
    Each block has U of shape (rows_u, r), V of shape (rows_v, r), W of shape (rows_w, r)."""
    total_rank = sum(b[0][0].__len__() if b[0] else 0 for b in blocks)
    # Actually let me compute properly
    total_rank = 0
    for U, V, W in blocks:
        if U:
            total_rank += len(U[0])
    
    n_rows_u = blocks[0][0][0].__len__() if blocks[0][0] else 0
    # Get shapes from first block
    rows_u = len(blocks[0][0])
    rows_v = len(blocks[0][1])
    rows_w = len(blocks[0][2])
    
    U = [[0]*total_rank for _ in range(rows_u)]
    V = [[0]*total_rank for _ in range(rows_v)]
    W = [[0]*total_rank for _ in range(rows_w)]
    
    col = 0
    for U_b, V_b, W_b in blocks:
        r = len(U_b[0]) if U_b else 0
        for i in range(rows_u):
            for c in range(r):
                U[i][col+c] = U_b[i][c]
        for i in range(rows_v):
            for c in range(r):
                V[i][col+c] = V_b[i][c]
        for i in range(rows_w):
            for c in range(r):
                W[i][col+c] = W_b[i][c]
        col += r
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes matrix multiplication.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta(i,j,k)
    where delta = 1 if this is a valid term, 0 otherwise.
    Actually the condition is: the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    should equal 1 for all (i,j,k) in range since matrix multiplication has T[i,j,k]=1 for all.
    Wait, let me reconsider. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    (since C[i,k] = sum_j A[i,j]*B[j,k], each term contributes 1).
    So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Strategy: Use Strassen (rank 7) for 2x2 sub-blocks where possible,
    # and schoolbook for the rest.
    #
    # We have a 3x4 * 4x4 multiplication.
    # Think of it as:
    #   A is 3x4, B is 4x4, C is 3x4.
    #
    # Partition A into rows: row 0 (1x4), rows 1-2 (2x4)
    # Partition B into cols: col 0 (4x1), cols 1-3 (4x3)
    # Actually, let's think in terms of output C[i,k] = sum_j A[i,j]*B[j,k].
    #
    # Approach: For output rows i=1,2 (0-indexed), and output cols k=0,1 (0-indexed),
    # we have a 2x2 sub-matrix multiplication if we consider A[1:3, 0:2] and B[0:2, 0:2].
    # But the dimensions don't align neatly for full Strassen usage.
    #
    # Let's use a hybrid:
    # - For the top-left 2x2 output block (i in {0,1}, k in {0,1}): 
    #   C[0:2, 0:2] involves A[0:2, :] * B[:, 0:2]
    #   This is a 2x4 * 4x2 = 2x2 multiplication. Not directly Strassen.
    #
    # Better approach: Partition the middle dimension (j) into chunks of 2.
    # j ranges over 0,1,2,3. Split into j0={0,1} and j1={2,3}.
    #
    # C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    #
    # The first sum is a 3x2 * 2x4 multiplication (3 outputs, 2 middle, 4 cols)
    # The second sum is a 3x2 * 2x4 multiplication.
    #
    # Each 3x2*2x4 can be decomposed. For a 3x2*2x4:
    #   rank = 3*2*4 = 24 schoolbook.
    #   Can we do better? For 2x2x2 blocks within...
    #   Actually 3x2*2x4: think of it as for each pair (i,k), sum over j=0,1.
    #   This is like 12 independent dot products of length 2.
    #   We can group: for i in {0,1} and k in {0,1}, we have a 2x2x2 sub-problem.
    #   
    # Let me try a different partitioning.
    #
    # Partition i into {0} and {1,2}, k into {0,1} and {2,3}.
    #
    # Block (i={1,2}, k={0,1}): 2x4 * 4x2 -> 2x2 output
    #   Middle dim j has 4 elements. Split j into {0,1} and {2,3}.
    #   For j={0,1}: 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
    #   For j={2,3}: 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
    #   Total for this block: 14.
    #
    # Block (i={1,2}, k={2,3}): 2x4 * 4x2 -> 2x2 output
    #   Similarly: 7 + 7 = 14.
    #
    # Block (i={0}, k={0,1}): 1x4 * 4x2 -> 1x2 output
    #   Middle dim j has 4 elements. Split j into {0,1} and {2,3}.
    #   For j={0,1}: 1x2 * 2x2 -> 1x2. Schoolbook rank = 1*2*2 = 4.
    #   For j={2,3}: 1x2 * 2x2 -> 1x2. Schoolbook rank = 4.
    #   Total: 8.
    #
    # Block (i={0}, k={2,3}): 1x4 * 4x2 -> 1x2 output
    #   Similarly: 4 + 4 = 8.
    #
    # Total rank: 14 + 14 + 8 + 8 = 44.
    
    # Let me implement this carefully.
    # 
    # The full problem: n=3, m=4, p=4.
    # U has shape (12, rank), V has shape (16, rank), W has shape (12, rank).
    # U[i*4+j, c], V[j*4+k, c], W[k*3+i, c].
    
    # I'll build each block and then map indices.
    
    # Block A: i in {1,2}, k in {0,1}, j in {0,1} -> Strassen 2x2x2
    # Local: A_local is 2x2 (rows i=1,2; cols j=0,1)
    #         B_local is 2x2 (rows j=0,1; cols k=0,1)
    #         C_local is 2x2 (rows i=1,2; cols k=0,1)
    # Strassen gives U_s (2x2 x 7), V_s (2x2 x 7), W_s (2x2 x 7)
    # Mapping: local row r (0->i=1, 1->i=2), local col c (0->j=0, 1->j=1)
    # U_global[i*4+j] = U_s[r*2+c]
    # V_global[j*4+k] = V_s[c*2+d] where d is local col of B
    # W_global[k*3+i] = W_s[d*2+r] where d is local row of C (k), r is local col of C (i)
    
    U_total = [[0]*44 for _ in range(12)]
    V_total = [[0]*44 for _ in range(16)]
    W_total = [[0]*44 for _ in range(12)]
    
    col = 0
    
    # Block A: i={1,2}, k={0,1}, j={0,1}
    U_s, V_s, W_s = _strassen_2x2x2()
    for c in range(7):
        for r in range(2):  # local row of A -> i
            i = 1 + r
            for cc in range(2):  # local col of A -> j
                j = 0 + cc
                U_total[i*4+j][col+c] += U_s[r*2+cc][c]
        for cc in range(2):  # local row of B -> j
            j = 0 + cc
            for dd in range(2):  # local col of B -> k
                k = 0 + dd
                V_total[j*4+k][col+c] += V_s[cc*2+dd][c]
        for dd in range(2):  # local row of C -> k
            k = 0 + dd
            for r in range(2):  # local col of C -> i
                i = 1 + r
                W_total[k*3+i][col+c] += W_s[dd*2+r][c]
    col += 7
    
    # Block B: i={1,2}, k={0,1}, j={2,3}
    U_s, V_s, W_s = _strassen_2x2x2()
    for c in range(7):
        for r in range(2):
            i = 1 + r
            for cc in range(2):
                j = 2 + cc
                U_total[i*4+j][col+c] += U_s[r*2+cc][c]
        for cc in range(2):
            j = 2 + cc
            for dd in range(2):
                k = 0 + dd
                V_total[j*4+k][col+c] += V_s[cc*2+dd][c]
        for dd in range(2):
            k = 0 + dd
            for r in range(2):
                i = 1 + r
                W_total[k*3+i][col+c] += W_s[dd*2+r][c]
    col += 7
    
    # Block C: i={1,2}, k={2,3}, j={0,1}
    U_s, V_s, W_s = _strassen_2x2x2()
    for c in range(7):
        for r in range(2):
            i = 1 + r
            for cc in range(2):
                j = 0 + cc
                U_total[i*4+j][col+c] += U_s[r*2+cc][c]
        for cc in range(2):
            j = 0 + cc
            for dd in range(2):
                k = 2 + dd
                V_total[j*4+k][col+c] += V_s[cc*2+dd][c]
        for dd in range(2):
            k = 2 + dd
            for r in range(2):
                i = 1 + r
                W_total[k*3+i][col+c] += W_s[dd*2+r][c]
    col += 7
    
    # Block D: i={1,2}, k={2,3}, j={2,3}
    U_s, V_s, W_s = _strassen_2x2x2()
    for c in range(7):
        for r in range(2):
            i = 1 + r
            for cc in range(2):
                j = 2 + cc
                U_total[i*4+j][col+c] += U_s[r*2+cc][c]
        for cc in range(2):
            j = 2 + cc
            for dd in range(2):
                k = 2 + dd
                V_total[j*4+k][col+c] += V_s[cc*2+dd][c]
        for dd in range(2):
            k = 2 + dd
            for r in range(2):
                i = 1 + r
                W_total[k*3+i][col+c] += W_s[dd*2+r][c]
    col += 7
    
    # Block E: i={0}, k={0,1}, j={0,1} -> 1x2*2x2, schoolbook rank 4
    U_e, V_e, W_e = _schoolbook_block(1, 2, 2)
    for c in range(4):
        for r in range(1):
            i = 0
            for cc in range(2):
                j = 0 + cc
                U_total[i*4+j][col+c] += U_e[r*2+cc][c]
        for cc in range(2):
            j = 0 + cc
            for dd in range(2):
                k = 0 + dd
                V_total[j*4+k][col+c] += V_e[cc*2+dd][c]
        for dd in range(2):
            k = 0 + dd
            for r in range(1):
                i = 0
                W_total[k*3+i][col+c] += W_e[dd*1+r][c]
    col += 4
    
    # Block F: i={0}, k={0,1}, j={2,3} -> 1x2*2x2, schoolbook rank 4
    U_e, V_e, W_e = _schoolbook_block(1, 2, 2)
    for c in range(4):
        for r in range(1):
            i = 0
            for cc in range(2):
                j = 2 + cc
                U_total[i*4+j][col+c] += U_e[r*2+cc][c]
        for cc in range(2):
            j = 2 + cc
            for dd in range(2):
                k = 0 + dd
                V_total[j*4+k][col+c] += V_e[cc*2+dd][c]
        for dd in range(2):
            k = 0 + dd
            for r in range(1):
                i = 0
                W_total[k*3+i][col+c] += W_e[dd*1+r][c]
    col += 4
    
    # Block G: i={0}, k={2,3}, j={0,1} -> 1x2*2x2, schoolbook rank 4
    U_e, V_e, W_e = _schoolbook_block(1, 2, 2)
    for c in range(4):
        for r in range(1):
            i = 0
            for cc in range(2):
                j = 0 + cc
                U_total[i*4+j][col+c] += U_e[r*2+cc][c]
        for cc in range(2):
            j = 0 + cc
            for dd in range(2):
                k = 2 + dd
                V_total[j*4+k][col+c] += V_e[cc*2+dd][c]
        for dd in range(2):
            k = 2 + dd
            for r in range(1):
                i = 0
                W_total[k*3+i][col+c] += W_e[dd*1+r][c]
    col += 4
    
    # Block H: i={0}, k={2,3}, j={2,3} -> 1x2*2x2, schoolbook rank 4
    U_e, V_e, W_e = _schoolbook_block(1, 2, 2)
    for c in range(4):
        for r in range(1):
            i = 0
            for cc in range(2):
                j = 2 + cc
                U_total[i*4+j][col+c] += U_e[r*2+cc][c]
        for cc in range(2):
            j = 2 + cc
            for dd in range(2):
                k = 2 + dd
                V_total[j*4+k][col+c] += V_e[cc*2+dd][c]
        for dd in range(2):
            k = 2 + dd
            for r in range(1):
                i = 0
                W_total[k*3+i][col+c] += W_e[dd*1+r][c]
    col += 4
    
    # Verify
    if _verify(U_total, V_total, W_total, n, m, p):
        return {"U": U_total, "V": V_total, "W": W_total}
    
    # Fallback: schoolbook
    U, V, W = _schoolbook_block(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
