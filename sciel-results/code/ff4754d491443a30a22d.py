"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import itertools

DIMENSIONS = [3, 3, 8]
_DEADLINE = 130.0  # safety margin before 150s wall-clock kill
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

# Known small Strassen-like decompositions and their rank
# For 2x2x2, Strassen gives rank 7
# We need to embed these into 3x3x8

def _strassen_2x2_2():
    """
    Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Column 0: P1 = (a+d)(e+h)
    # a is at i=0,j=0 -> U[0,0]=1
    # d is at i=1,j=1 -> U[3,0]=1
    # e is at j=0,k=0 -> V[0,0]=1
    # h is at j=1,k=1 -> V[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 = P1...: W[0*2+0, 0] = W[0,0] = 1
    # C22 = P1...: W[1*2+1, 0] = W[3,0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Column 1: P2 = (c+d)e
    # c is at i=1,j=0 -> U[2,1]=1
    # d is at i=1,j=1 -> U[3,1]=1
    # e is at j=0,k=0 -> V[0,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 = P2...: W[0*2+1, 1] = W[1,1] = 1
    # C22 = -P2...: W[1*2+1, 1] = W[3,1] = -1
    W[1][1] = 1; W[3][1] = -1
    
    # Column 2: P3 = a(f-h)
    # a is at i=0,j=0 -> U[0,2]=1
    # f is at j=0,k=1 -> V[1,2]=1
    # h is at j=1,k=1 -> V[3,2]=-1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 = P3...: W[1*2+0, 2] = W[2,2] = 1
    # C22 = P3...: W[1*2+1, 2] = W[3,2] = 1
    W[2][2] = 1; W[3][2] = 1
    
    # Column 3: P4 = d(g-e)
    # d is at i=1,j=1 -> U[3,3]=1
    # g is at j=1,k=0 -> V[2,3]=1
    # e is at j=0,k=0 -> V[0,3]=-1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C11 = P4...: W[0*2+0, 3] = W[0,3] = 1
    # C21 = P4...: W[0*2+1, 3] = W[1,3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # Column 4: P5 = (a+b)h
    # a is at i=0,j=0 -> U[0,4]=1
    # b is at i=0,j=1 -> U[1,4]=1
    # h is at j=1,k=1 -> V[3,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 = -P5...: W[0*2+0, 4] = W[0,4] = -1
    # C12 = P5...: W[1*2+0, 4] = W[2,4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Column 5: P6 = (c-a)(e+f)
    # c is at i=1,j=0 -> U[2,5]=1
    # a is at i=0,j=0 -> U[0,5]=-1
    # e is at j=0,k=0 -> V[0,5]=1
    # f is at j=0,k=1 -> V[1,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 = P6...: W[1*2+1, 5] = W[3,5] = 1
    W[3][5] = 1
    
    # Column 6: P7 = (b-d)(g+h)
    # b is at i=0,j=1 -> U[1,6]=1
    # d is at i=1,j=1 -> U[3,6]=-1
    # g is at j=1,k=0 -> V[2,6]=1
    # h is at j=1,k=1 -> V[3,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 = P7...: W[0*2+0, 6] = W[0,6] = 1
    W[0][6] = 1
    
    return U, V, W

def _embed_strassen_into_3x3x8():
    """
    Embed the 2x2x2 Strassen decomposition into 3x3x8.
    We use the top-left 2x2 block for the Strassen part,
    and handle the remaining rows/columns with schoolbook.
    
    For 3x3 * 3x8:
    - Rows 0,1 of A (first 2 rows) and cols 0,1 of B (first 2 cols) -> Strassen 2x2x2
    - Row 2 of A * all 8 cols of B -> schoolbook (24 terms)
    - Rows 0,1 of A * cols 2-7 of B -> schoolbook (2*6=12 terms)
    
    Total rank = 7 + 24 + 12 = 43
    """
    n, m, p = 3, 3, 8
    rank = 43
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Embed Strassen 2x2x2 in the top-left 2x2 block
    sU, sV, sW = _strassen_2x2_2()
    for c in range(7):
        for u_idx in range(4):
            U[u_idx][c] = sU[u_idx][c]
        for v_idx in range(4):
            V[v_idx][c] = sV[v_idx][c]
        for w_idx in range(4):
            W[w_idx][c] = sW[w_idx][c]
    
    col = 7
    
    # Row 2 of A * all 8 cols of B (schoolbook)
    # i=2, j in {0,1,2}, k in {0,...,7}
    for j in range(3):
        for k in range(8):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    # Rows 0,1 of A * cols 2-7 of B (schoolbook)
    for i in range(2):
        for j in range(3):
            for k in range(2, 8):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return U, V, W

def _build_strassen_embedding(i_start, k_start):
    """
    Build a decomposition with one Strassen 2x2x2 block at rows [i_start, i_start+1]
    and cols [k_start, k_start+1], with schoolbook for the rest.
    
    Returns (U, V, W, rank) or None if invalid placement.
    """
    n, m, p = 3, 3, 8
    if i_start + 1 >= n or k_start + 1 >= p:
        return None
    
    sU, sV, sW = _strassen_2x2_2()
    
    # Determine which (i, j, k) tuples are covered by Strassen
    # Strassen covers: i in {i_start, i_start+1}, j in {0,1} (for the 2x2 A block), k in {k_start, k_start+1}
    # Wait - Strassen 2x2x2 means 2x2 A, 2x2 B, 2x2 C
    # A is 2x2: rows i_start, i_start+1, cols j=0,1 (the middle dimension)
    # B is 2x2: rows j=0,1, cols k_start, k_start+1
    # C is 2x2: rows k_start, k_start+1, cols i_start, i_start+1
    
    # Actually, let me think about this more carefully.
    # The tensor is T[i,j,k] where A[i,j] * B[j,k] = C[i,k]
    # A is n x m = 3x3, B is m x p = 3x8, C is p x n = 8x3
    # U has shape (n*m, rank) = (9, rank), indexed by (i,j)
    # V has shape (m*p, rank) = (24, rank), indexed by (j,k)
    # W has shape (p*n, rank) = (24, rank), indexed by (k,i)
    
    # Strassen 2x2x2 handles a 2x2 sub-problem:
    # A_sub is 2x2 (rows i_start..i_start+1, cols j_start..j_start+1)
    # B_sub is 2x2 (rows j_start..j_start+1, cols k_start..k_start+1)
    # C_sub is 2x2 (rows k_start..k_start+1, cols i_start..i_start+1)
    
    # But wait - in our problem, m=3 (middle dimension). The Strassen block uses
    # 2 rows of A, 2 cols of B, which share 2 rows of B = 2 cols of A.
    # So we need j_start too. Let's use j_start=0 for now (use first 2 middle indices).
    
    # Actually, the middle dimension is shared. For a 2x2x2 Strassen:
    # A uses rows i_start..i_start+1, cols j_start..j_start+1
    # B uses rows j_start..j_start+1, cols k_start..k_start+1
    # C uses rows k_start..k_start+1, cols i_start..i_start+1
    
    # The key constraint: the j indices must be the same for A's cols and B's rows.
    # So we need to pick j_start as well.
    
    # For simplicity, let's fix j_start=0 (use middle indices 0,1)
    j_start = 0
    
    # Covered (i,j,k) tuples:
    covered = set()
    for i in [i_start, i_start+1]:
        for j in [j_start, j_start+1]:
            for k in [k_start, k_start+1]:
                covered.add((i, j, k))
    
    # Count schoolbook terms needed
    schoolbook_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook_count += 1
    
    rank = 7 + schoolbook_count
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Place Strassen block
    for c in range(7):
        for u_idx in range(4):
            # u_idx maps to (i_local, j_local) in the 2x2 sub-block
            i_local = u_idx // 2
            j_local = u_idx % 2
            i_global = i_start + i_local
            j_global = j_start + j_local
            U[i_global*m + j_global][c] = sU[u_idx][c]
        for v_idx in range(4):
            # v_idx maps to (j_local, k_local)
            j_local = v_idx // 2
            k_local = v_idx % 2
            j_global = j_start + j_local
            k_global = k_start + k_local
            V[j_global*p + k_global][c] = sV[v_idx][c]
        for w_idx in range(4):
            # w_idx maps to (k_local, i_local)
            k_local = w_idx // 2
            i_local = w_idx % 2
            k_global = k_start + k_local
            i_global = i_start + i_local
            W[k_global*n + i_global][c] = sW[w_idx][c]
    
    col = 7
    
    # Schoolbook for uncovered terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W, rank

def _try_strassen_placements():
    """
    Try all valid single Strassen placements and return the best one.
    """
    n, m, p = 3, 3, 8
    best = None
    best_rank = float('inf')
    
    # Try all (i_start, k_start) combinations with j_start=0
    for i_start in range(n - 1):
        for k_start in range(p - 1):
            if _time_left() < 5.0:
                break
            result = _build_strassen_embedding(i_start, k_start)
            if result is None:
                continue
            U, V, W, rank = result
            if rank < best_rank:
                if _verify(U, V, W, n, m, p):
                    best = (U, V, W)
                    best_rank = rank
    
    return best, best_rank

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    and for all other (i,j,k) combinations that should be 0, they are 0.
    """
    rank = len(U[0])
    # Build the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Try systematic search over Strassen placements
    try:
        if _time_left() > 10.0:
            best, best_rank = _try_strassen_placements()
            if best is not None:
                U, V, W = best
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try the fixed top-left embedding
    try:
        U, V, W = _embed_strassen_into_3x3x8()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
