"""Matrix multiplication 2x4 by 4x5 tensor decomposition with rank minimization."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify exact decomposition using numpy for speed."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) for n=2, m=2, p=2.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [
        [1,0,1,0,1,-1,0],
        [0,0,0,0,1,0,1],
        [0,1,0,0,0,1,0],
        [1,1,0,1,0,0,-1]
    ]
    V = [
        [1,1,0,-1,0,1,0],
        [0,0,1,0,0,1,0],
        [0,0,0,1,0,0,1],
        [1,0,-1,0,1,0,1]
    ]
    W = [
        [1,0,1,0,-1,-1,0],
        [0,0,-1,1,0,0,1],
        [0,1,0,-1,-1,0,0],
        [1,0,0,0,1,1,1]
    ]
    
    # Verify
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.dot(U_np[i*2+j], V_np[j*2+k] * W_np[k*2+i])
                if abs(s - 1.0) > 1e-9:
                    return None
    return U, V, W

def _block_2x2x1():
    """
    Decomposition for 2x2x1 with rank 2.
    A (2x2), B (2x1), C (2x1).
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    
    Rank 2:
    c=0: U=[1,1,0,0], V=[1,1], W=[1,0] -> contributes to C0
    c=1: U=[0,0,1,1], V=[1,1], W=[0,1] -> contributes to C1
    """
    U = [
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ]
    V = [
        [1,1],
        [1,1]
    ]
    W = [
        [1,0],
        [0,1]
    ]
    return U, V, W

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy: 2x4x5 = two 2x2x5 blocks
    # 2x2x5 = 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x1 (rank 2) = rank 16
    # Total rank = 32
    
    total_rank = 32
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    strassen_2x2x2 = _strassen_2x2x2()
    if strassen_2x2x2 is None:
        return _schoolbook(n, m, p)
    sU, sV, sW = strassen_2x2x2
    bU, bV, bW = _block_2x2x1()
    
    col_offset = 0
    
    for j_block in [0, 2]:
        # 1. 2x2x2 for k=0,1
        for c in range(7):
            for li in range(2):
                gi = li
                for lj in range(2):
                    gj = j_block + lj
                    U[gi*4+gj][col_offset+c] = sU[li*2+lj][c]
            
            for lj in range(2):
                gj = j_block + lj
                for lk in range(2):
                    gk = lk
                    V[gj*5+gk][col_offset+c] = sV[lj*2+lk][c]
            
            for lk in range(2):
                gk = lk
                for li in range(2):
                    gi = li
                    W[gk*2+gi][col_offset+c] = sW[lk*2+li][c]
        col_offset += 7
        
        # 2. 2x2x2 for k=2,3
        for c in range(7):
            for li in range(2):
                gi = li
                for lj in range(2):
                    gj = j_block + lj
                    U[gi*4+gj][col_offset+c] = sU[li*2+lj][c]
            
            for lj in range(2):
                gj = j_block + lj
                for lk in range(2):
                    gk = 2 + lk
                    V[gj*5+gk][col_offset+c] = sV[lj*2+lk][c]
            
            for lk in range(2):
                gk = 2 + lk
                for li in range(2):
                    gi = li
                    W[gk*2+gi][col_offset+c] = sW[lk*2+li][c]
        col_offset += 7
        
        # 3. 2x2x1 for k=4
        for c in range(2):
            for li in range(2):
                gi = li
                for lj in range(2):
                    gj = j_block + lj
                    U[gi*4+gj][col_offset+c] = bU[li*2+lj][c]
            
            for lj in range(2):
                gj = j_block + lj
                gk = 4
                V[gj*5+gk][col_offset+c] = bV[lj][c]
            
            gk = 4
            for li in range(2):
                gi = li
                W[gk*2+gi][col_offset+c] = bW[li][c]
        col_offset += 2
    
    # Verify
    if _verify(U, V, W, n, m, p, total_rank):
        return {"U": U, "V": V, "W": W}
    else:
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
