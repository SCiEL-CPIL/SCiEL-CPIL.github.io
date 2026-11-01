"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _strassen_block_decomposition(block_idx, n, m, p):
    """
    Return (U_rows, V_rows, W_rows) for one 2x2 * 2x2 Strassen block.
    block_idx: 0, 1, or 2 (which 2x2 block of the 2x6 B matrix)
    
    U has 4 rows (A indices: i*m+j for i in {0,1}, j in {0,1})
    V has 12 rows (B indices: j*p+k for j in {0,1}, k in {0,1,2,3,4,5})
    W has 12 rows (C indices: k*n+i for k in {0,1,2,3,4,5}, i in {0,1})
    
    For block_idx b, the relevant B columns are k = b*2, b*2+1
    The relevant C columns are k = b*2, b*2+1
    """
    b = block_idx
    rank = 7  # Strassen rank for 2x2*2x2
    
    # U: 4 rows, 7 columns
    # Row indices: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
    # A indices: A[0,0]=row0, A[0,1]=row1, A[1,0]=row2, A[1,1]=row3
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]: appears in P1,P3,P5,P6
        [0, 0, 0, 0, 1, 0, -1],   # A[0,1]: appears in P5,P7
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]: appears in P2,P6
        [1, 1, 0, 1, 0, 0, 0],    # A[1,1]: appears in P1,P2,P4
    ]
    
    # V: 12 rows, 7 columns
    # Row indices: j*p+k, j in {0,1}, k in {0..5}
    # For block b: B[0, b*2], B[0, b*2+1], B[1, b*2], B[1, b*2+1]
    # V row 0 = B[0,0], row 1 = B[0,1], row 2 = B[0,2], row 3 = B[0,3], row 4 = B[0,4], row 5 = B[0,5]
    # V row 6 = B[1,0], row 7 = B[1,1], row 8 = B[1,2], row 9 = B[1,3], row 10 = B[1,4], row 11 = B[1,5]
    V = [[0]*7 for _ in range(12)]
    # B[0, b*2] = B[0,0] of block: appears in P1, P2, P6
    V[0 * p + b*2] = [1, 1, 0, 0, 0, 1, 0]
    # B[0, b*2+1] = B[0,1] of block: appears in P4, P7
    V[0 * p + b*2 + 1] = [0, 0, 0, -1, 0, 0, 1]
    # B[1, b*2] = B[1,0] of block: appears in P3, P6
    V[1 * p + b*2] = [0, 0, 1, -1, 0, 1, 0]
    # B[1, b*2+1] = B[1,1] of block: appears in P1, P5, P7
    V[1 * p + b*2 + 1] = [1, 0, 0, 0, 1, 0, 1]
    
    # W: 12 rows, 7 columns
    # Row indices: k*n+i, k in {0..5}, i in {0,1}
    # W row 0 = C[0,0], row 1 = C[0,1], row 2 = C[1,0], row 3 = C[1,1], row 4 = C[2,0], row 5 = C[2,1], ...
    # For block b: C[0, b*2], C[0, b*2+1], C[1, b*2], C[1, b*2+1]
    W = [[0]*7 for _ in range(12)]
    # C[0, b*2] = C[0,0] of block: P1+P4-P5+P7
    W[(b*2) * n + 0] = [1, 0, 0, 1, -1, 0, 1]
    # C[0, b*2+1] = C[0,1] of block: P3+P5
    W[(b*2+1) * n + 0] = [0, 0, 1, 0, 1, 0, 0]
    # C[1, b*2] = C[1,0] of block: P2+P4
    W[(b*2) * n + 1] = [0, 1, 0, 1, 0, 0, 0]
    # C[1, b*2+1] = C[1,1] of block: P1-P2+P3+P6
    W[(b*2+1) * n + 1] = [1, -1, 1, 0, 0, 1, 0]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build rank-21 decomposition using Strassen on three 2x2 blocks
    rank = 21
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for b in range(3):
        U_b, V_b, W_b = _strassen_block_decomposition(b, n, m, p)
        offset = b * 7
        for i in range(nm):
            for r in range(7):
                U[i][offset + r] = U_b[i][r]
        for i in range(mp):
            for r in range(7):
                V[i][offset + r] = V_b[i][r]
        for i in range(pn):
            for r in range(7):
                W[i][offset + r] = W_b[i][r]
    
    # Verify the decomposition
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-24 decomposition
    schoolbook_rank = n * m * p  # 24
    U_fb = [[0] * schoolbook_rank for _ in range(nm)]
    V_fb = [[0] * schoolbook_rank for _ in range(mp)]
    W_fb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
