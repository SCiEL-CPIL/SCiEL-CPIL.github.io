import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _time_left(deadline):
    return deadline - time.time()

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), the coefficient should be sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if k == (i*j) mod p... no wait.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor for matmul)
    # Actually, T[i,j,k] = 1 for all valid i,j,k in the standard matmul tensor.
    
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matmul tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where C[i,k] = sum_j A[i,j] B[j,k]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return factors for 2x2x2 Strassen decomposition (rank 7).
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U factors (A side): rows indexed by i*2+j
    # V factors (B side): rows indexed by j*2+k
    # W factors (C side): rows indexed by k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    # C00 += M1, C11 += M1
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)
    
    # M2 = (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    # C10 += M2, C11 -= M2
    W[2][1] = 1  # C10 (k=0,i=1)
    W[3][1] = -1 # C11 (k=1,i=1)
    
    # M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[3][2] = -1 # B11
    # C01 += M3, C11 += M3
    W[1][2] = 1  # C01 (k=1,i=0)
    W[3][2] = 1  # C11 (k=1,i=1)
    
    # M4 = A11(B01-B10)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[2][3] = -1 # B10
    # C00 += M4, C10 += M4
    W[0][3] = 1  # C00 (k=0,i=0)
    W[2][3] = 1  # C10 (k=0,i=1)
    
    # M5 = (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    # C00 -= M5, C01 += M5
    W[0][4] = -1 # C00 (k=0,i=0)
    W[1][4] = 1  # C01 (k=1,i=0)
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    # (M6 not used in standard Strassen output formulas for 2x2... let me recheck)
    # Actually in standard Strassen: C00 = M1 - M5 + M6 - M7
    # Let me redo this properly.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let me restart with correct formulas
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[2][3] = 1  # C10
    W[0][3] = 1  # C00... wait, C00 doesn't use M4 in standard form
    # Let me recheck: C00 = M1 - M5 + M6 - M7, C10 = M2 + M4
    # So M4 only contributes to C10
    W[2][3] = 1  # C10
    # Remove the C00 contribution
    W[0][3] = 0
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1  # C00
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1 # C00
    W[3][6] = 1  # C11
    
    return U, V, W

def _build_strassen_embedding(n, m, p, row_block, col_block, out_block):
    """
    Build a decomposition that uses Strassen for a 2x2x2 sub-block
    and schoolbook for the rest.
    
    row_block: which 2 of the 3 rows of A to use in Strassen (e.g., (0,1), (0,2), (1,2))
    col_block: which 2 of the 3 cols of A / rows of B to use (e.g., (0,1), (0,2), (1,2))  
    out_block: which 2 of the 8 output cols to use (e.g., (0,1), (0,2), ...)
    
    Returns U, V, W for the full 3x3x8 problem.
    """
    rank = 7 + (n*m*p - 4*2)  # 7 Strassen + schoolbook for remaining
    # Actually, let's think about this more carefully.
    # Total entries: n*m*p = 3*3*8 = 72
    # Strassen handles 2*2*2 = 8 entries with rank 7
    # Remaining: 72 - 8 = 64 entries with rank 64 (schoolbook)
    # Total rank: 7 + 64 = 71
    # That's only 1 better than schoolbook 72. Not great.
    
    # Better approach: use multiple Strassen blocks.
    # We can partition the 3x3 A matrix into overlapping 2x2 blocks.
    # But they must be non-overlapping in terms of the tensor entries they cover.
    
    # Actually, let's try a different approach:
    # Partition the 8 output columns into groups and use Strassen where possible.
    
    # For now, let's just do one Strassen block + schoolbook for the rest.
    rank = 7 + (n*m*p - 8)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen block for rows row_block, cols col_block, outputs out_block
    r0, r1 = row_block
    c0, c1 = col_block
    o0, o1 = out_block
    
    sU, sV, sW = _strassen_2x2x2()
    
    for s in range(7):
        # A side: map (s_row, s_col) -> global (row, col)
        for sr in range(2):
            for sc in range(2):
                if sU[sr*2+sc][s] != 0:
                    gr = row_block[sr]
                    gc = col_block[sc]
                    U[gr*m + gc][s] = sU[sr*2+sc][s]
        
        # B side: map (s_row, s_col) -> global (row, col)
        for sr in range(2):
            for sc in range(2):
                if sV[sr*2+sc][s] != 0:
                    gr = col_block[sr]  # B row index = A col index
                    gc = out_block[sc]
                    V[gr*p + gc][s] = sV[sr*2+sc][s]
        
        # C side: map (s_row, s_col) -> global (out_row, out_col)
        for sr in range(2):
            for sc in range(2):
                if sW[sr*2+sc][s] != 0:
                    go = out_block[sr]  # C col index
                    gi = row_block[sc]  # C row index
                    W[go*n + gi][s] = sW[sr*2+sc][s]
    
    # Schoolbook for all other entries
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this entry is covered by Strassen
                if i in row_block and j in col_block and k in out_block:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, get schoolbook as baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_sb, V_sb, W_sb)
    
    # Try single Strassen embeddings
    row_blocks = [(0,1), (0,2), (1,2)]
    col_blocks = [(0,1), (0,2), (1,2)]
    out_blocks = [(a,b) for a in range(p) for b in range(p) if a < b]
    
    for rb in row_blocks:
        for cb in col_blocks:
            for ob in out_blocks:
                if _time_left(deadline) < 5.0:
                    break
                U, V, W = _build_strassen_embedding(n, m, p, rb, cb, ob)
                if _verify(U, V, W, n, m, p):
                    rank = len(U[0])
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
    
    if _time_left(deadline) < 5.0:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Try multiple non-overlapping Strassen blocks
    # We need to find sets of (row_block, col_block, out_block) that don't overlap
    # in their tensor entries.
    
    def entries_covered(rb, cb, ob):
        """Return set of (i,j,k) entries covered by a Strassen block."""
        s = set()
        for i in rb:
            for j in cb:
                for k in ob:
                    s.add((i,j,k))
        return s
    
    # Find all maximal sets of non-overlapping Strassen blocks
    all_blocks = []
    for rb in row_blocks:
        for cb in col_blocks:
            for ob in out_blocks:
                all_blocks.append((rb, cb, ob))
    
    # Greedy: try to add blocks that don't overlap
    # For simplicity, try pairs of blocks
    for i in range(len(all_blocks)):
        if _time_left(deadline) < 3.0:
            break
        rb1, cb1, ob1 = all_blocks[i]
        covered1 = entries_covered(rb1, cb1, ob1)
        
        for j in range(i+1, len(all_blocks)):
            if _time_left(deadline) < 2.0:
                break
            rb2, cb2, ob2 = all_blocks[j]
            covered2 = entries_covered(rb2, cb2, ob2)
            
            if covered1 & covered2:
                continue
            
            # Build decomposition with two Strassen blocks
            rank = 14 + (n*m*p - 16)
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            sU, sV, sW = _strassen_2x2x2()
            
            # First Strassen block
            for s in range(7):
                for sr in range(2):
                    for sc in range(2):
                        if sU[sr*2+sc][s] != 0:
                            U[rb1[sr]*m + cb1[sc]][s] = sU[sr*2+sc][s]
                        if sV[sr*2+sc][s] != 0:
                            V[cb1[sr]*p + ob1[sc]][s] = sV[sr*2+sc][s]
                        if sW[sr*2+sc][s] != 0:
                            W[ob1[sr]*n + rb1[sc]][s] = sW[sr*2+sc][s]
            
            # Second Strassen block (offset by 7)
            for s in range(7):
                for sr in range(2):
                    for sc in range(2):
                        if sU[sr*2+sc][s] != 0:
                            U[rb2[sr]*m + cb2[sc]][s+7] = sU[sr*2+sc][s]
                        if sV[sr*2+sc][s] != 0:
                            V[cb2[sr]*p + ob2[sc]][s+7] = sV[sr*2+sc][s]
                        if sW[sr*2+sc][s] != 0:
                            W[ob2[sr]*n + rb2[sc]][s+7] = sW[sr*2+sc][s]
            
            # Schoolbook for remaining
            col = 14
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        if (i2,j2,k2) in covered1 or (i2,j2,k2) in covered2:
                            continue
                        U[i2*m+j2][col] = 1
                        V[j2*p+k2][col] = 1
                        W[k2*n+i2][col] = 1
                        col += 1
            
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
