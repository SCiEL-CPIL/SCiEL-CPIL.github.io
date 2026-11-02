"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def _rank1_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m x p."""
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

def _direct_sum(U1, V1, W1, U2, V2, W2):
    """Direct sum of two tensor decompositions (disjoint index sets handled by caller)."""
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    n1m1 = len(U1)
    n2m2 = len(U2)
    m1p1 = len(V1)
    m2p2 = len(V2)
    p1n1 = len(W1)
    p2n2 = len(W2)
    
    # Pad to same sizes
    nmax = max(n1m1, n2m2)
    mmax_p1 = max(m1p1, m2p2)
    pmax_n1 = max(p1n1, p2n2)
    
    U = [[0] * (r1 + r2) for _ in range(nmax)]
    V = [[0] * (r1 + r2) for _ in range(mmax_p1)]
    W = [[0] * (r1 + r2) for _ in range(pmax_n1)]
    
    for i in range(n1m1):
        for c in range(r1):
            U[i][c] = U1[i][c]
    for i in range(n2m2):
        for c in range(r2):
            U[i][r1 + c] = U2[i][c]
    for i in range(m1p1):
        for c in range(r1):
            V[i][c] = V1[i][c]
    for i in range(m2p2):
        for c in range(r2):
            V[i][r1 + c] = V2[i][c]
    for i in range(p1n1):
        for c in range(r1):
            W[i][c] = W1[i][c]
    for i in range(p2n2):
        for c in range(r2):
            W[i][r1 + c] = W2[i][c]
    
    return U, V, W

def _construct_2x3x9():
    """
    Construct rank-minimal decomposition for 2x3 * 3x9.
    Split 2x3*3x9 into (2x3*3x3) + (2x3*3x6).
    For 2x3*3x3: use schoolbook rank 18.
    For 2x3*3x6: use schoolbook rank 36.
    Total: 54 (schoolbook).
    
    Actually, let's try a better approach.
    2x3 * 3x9 can be decomposed as:
    Split the 3 middle dimension.
    
    Let me try: split the 9 columns into 3+3+3.
    2x3 * 3x3 (rank 18) + 2x3 * 3x3 (rank 18) + 2x3 * 3x3 (rank 18) = 54.
    
    Alternatively, split the 2 rows:
    1x3 * 3x9 (rank 27) + 1x3 * 3x9 (rank 27) = 54.
    
    The schoolbook rank is 54. Let's try to find a better one.
    
    For M_{2,3,9}, the known lower bound is max(2*9, 3*9, 2*3) = 27, but that's not tight.
    Actually, for M_{m,n,p}, the rank is at most m*n*p and at least max(mn, np, mp).
    
    Let me just return schoolbook for now, but with a time guard.
    """
    n, m, p = 2, 3, 9
    
    # Try to see if we can do better than 54.
    # One approach: use the fact that 9 = 3*3, and 3 = 3.
    # Decompose 2x3*3x9 as a block matrix.
    
    # Actually, let's try: 2x3 * 3x9
    # = [A1; A2] where A1 is 1x3, A2 is 1x3
    # B is 3x9.
    # C1 = A1 * B (1x9)
    # C2 = A2 * B (1x9)
    
    # C1 and C2 can be computed with shared intermediate products.
    # Let B = [b1, b2, b3] where bj is 1x9.
    # C1 = a11*b1 + a12*b2 + a13*b3
    # C2 = a21*b1 + a22*b2 + a23*b3
    
    # Shared: b1, b2, b3 (each 1x9, so 9 scalars each)
    # But we need matrix multiplication, not just scalar.
    
    # Let's stick with schoolbook: rank 54.
    U, V, W = _rank1_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication 2x3 * 3x9.
    Rank is 2*3*9 = 54.
    """
    start = time.time()
    deadline = start + 140  # 140s safety margin
    
    try:
        return _construct_2x3x9()
    except Exception as e:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
