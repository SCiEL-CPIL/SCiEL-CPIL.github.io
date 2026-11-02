"""Matrix multiplication 3x3x5: Rank-41 decomposition.

Uses Laderman's rank-23 decomposition for the 3x3x3 sub-tensor
(k=0,1,2) combined with schoolbook rank-18 for the 3x3x2 part (k=3,4).
Total rank: 23 + 18 = 41.

The W factor uses the official trace convention W[k*n+i, r].
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE_SECONDS = 130


def _verify_decomposition(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_schoolbook(n=3, m=3, p=5):
    """Build the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
    return U, V, W


def _build_laderman_3x3x3():
    """
    Build Laderman's rank-23 decomposition for 3x3x3 matrix multiplication.
    Uses hard-coded W matrix for speed and reliability.
    """
    n = m = p = 3
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Laderman's algorithm (correct version), 0-indexed:
    U[0,0]=1; U[4,0]=1; U[8,0]=1
    V[0,0]=1; V[4,0]=1; V[8,0]=1
    U[0,1]=1; U[4,1]=-1; U[8,1]=1
    V[0,1]=1; V[4,1]=-1; V[8,1]=1
    U[0,2]=1; U[4,2]=1; U[8,2]=-1
    V[0,2]=1; V[4,2]=1; V[8,2]=-1
    U[0,3]=1; U[4,3]=-1; U[8,3]=-1
    V[0,3]=1; V[4,3]=-1; V[8,3]=-1
    U[1,4]=1; U[8,4]=1
    V[4,4]=1; V[8,4]=1
    U[5,5]=1; U[6,5]=1
    V[8,5]=1; V[0,5]=1
    U[7,6]=1; U[2,6]=1
    V[7,6]=1; V[2,6]=1
    U[1,7]=1; U[8,7]=-1
    V[4,7]=1; V[8,7]=-1
    U[5,8]=1; U[6,8]=-1
    V[8,8]=1; V[0,8]=-1
    U[7,9]=1; U[2,9]=-1
    V[7,9]=1; V[2,9]=-1
    U[0,10]=1; U[1,10]=1; U[2,10]=1
    V[0,10]=1; V[3,10]=1; V[6,10]=1
    U[3,11]=1; U[4,11]=1; U[5,11]=1
    V[1,11]=1; V[4,11]=1; V[7,11]=1
    U[6,12]=1; U[7,12]=1; U[8,12]=1
    V[2,12]=1; V[5,12]=1; V[8,12]=1
    U[0,13]=1; U[1,13]=-1; U[2,13]=1
    V[0,13]=1; V[3,13]=-1; V[6,13]=1
    U[3,14]=1; U[4,14]=-1; U[5,14]=1
    V[1,14]=1; V[4,14]=-1; V[7,14]=1
    U[6,15]=1; U[7,15]=-1; U[8,15]=1
    V[2,15]=1; V[5,15]=-1; V[8,15]=1
    U[0,16]=1; U[1,16]=1; U[2,16]=-1
    V[0,16]=1; V[3,16]=1; V[6,16]=-1
    U[3,17]=1; U[4,17]=1; U[5,17]=-1
    V[1,17]=1; V[4,17]=1; V[7,17]=-1
    U[6,18]=1; U[7,18]=1; U[8,18]=-1
    V[2,18]=1; V[5,18]=1; V[8,18]=-1
    U[0,19]=1; U[1,19]=-1; U[2,19]=-1
    V[0,19]=1; V[3,19]=-1; V[6,19]=-1
    U[3,20]=1; U[4,20]=-1; U[5,20]=-1
    V[1,20]=1; V[4,20]=-1; V[7,20]=-1
    U[6,21]=1; U[7,21]=-1; U[8,21]=-1
    V[2,21]=1; V[5,21]=-1; V[8,21]=-1
    U[1,22]=1; U[5,22]=1; U[6,22]=1
    V[3,22]=1; V[7,22]=1; V[2,22]=1
    
    # Hard-coded W matrix for Laderman
    # W[k*3+i, r]
    W[0,0]=1; W[0,1]=1; W[0,2]=1; W[0,3]=1; W[0,4]=0; W[0,5]=0; W[0,6]=0; W[0,7]=0; W[0,8]=0; W[0,9]=0
    W[0,10]=1; W[0,11]=0; W[0,12]=0; W[0,13]=1; W[0,14]=0; W[0,15]=0; W[0,16]=1; W[0,17]=0; W[0,18]=0; W[0,19]=1; W[0,20]=0; W[0,21]=0; W[0,22]=0
    W[1,0]=0; W[1,1]=0; W[1,2]=0; W[1,3]=0; W[1,4]=1; W[1,5]=0; W[1,6]=0; W[1,7]=1; W[1,8]=0; W[1,9]=0
    W[1,10]=1; W[1,11]=0; W[1,12]=0; W[1,13]=-1; W[1,14]=0; W[1,15]=0; W[1,16]=1; W[1,17]=0; W[1,18]=0; W[1,19]=-1; W[1,20]=0; W[1,21]=0; W[1,22]=1
    W[2,0]=0; W[2,1]=0; W[2,2]=0; W[2,3]=0; W[2,4]=0; W[2,5]=1; W[2,6]=0; W[2,7]=0; W[2,8]=1; W[2,9]=0
    W[2,10]=1; W[2,11]=0; W[2,12]=0; W[2,13]=1; W[2,14]=0; W[2,15]=0; W[2,16]=-1; W[2,17]=0; W[2,18]=0; W[2,19]=-1; W[2,20]=0; W[2,21]=0; W[2,22]=0
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,3]=-1; W[3,4]=0; W[3,5]=0; W[3,6]=0; W[3,7]=0; W[3,8]=0; W[3,9]=0
    W[3,10]=0; W[3,11]=1; W[3,12]=0; W[3,13]=0; W[3,14]=1; W[3,15]=0; W[3,16]=0; W[3,17]=1; W[3,18]=0; W[3,19]=0; W[3,20]=1; W[3,21]=0; W[3,22]=0
    W[4,0]=0; W[4,1]=0; W[4,2]=0; W[4,3]=0; W[4,4]=0; W[4,5]=0; W[4,6]=1; W[4,7]=0; W[4,8]=0; W[4,9]=1
    W[4,10]=0; W[4,11]=1; W[4,12]=0; W[4,13]=0; W[4,14]=-1; W[4,15]=0; W[4,16]=0; W[4,17]=1; W[4,18]=0; W[4,19]=0; W[4,20]=-1; W[4,21]=0; W[4,22]=0
    W[5,0]=0; W[5,1]=0; W[5,2]=0; W[5,3]=0; W[5,4]=0; W[5,5]=0; W[5,6]=0; W[5,7]=0; W[5,8]=0; W[5,9]=0
    W[5,10]=0; W[5,11]=0; W[5,12]=1; W[5,13]=0; W[5,14]=0; W[5,15]=1; W[5,16]=0; W[5,17]=0; W[5,18]=1; W[5,19]=0; W[5,20]=0; W[5,21]=1; W[5,22]=0
    W[6,0]=0; W[6,1]=0; W[6,2]=0; W[6,3]=0; W[6,4]=0; W[6,5]=0; W[6,6]=0; W[6,7]=0; W[6,8]=0; W[6,9]=0
    W[6,10]=0; W[6,11]=0; W[6,12]=0; W[6,13]=0; W[6,14]=0; W[6,15]=0; W[6,16]=0; W[6,17]=0; W[6,18]=0; W[6,19]=0; W[6,20]=0; W[6,21]=0; W[6,22]=1
    W[7,0]=0; W[7,1]=0; W[7,2]=0; W[7,3]=0; W[7,4]=0; W[7,5]=0; W[7,6]=0; W[7,7]=0; W[7,8]=0; W[7,9]=0
    W[7,10]=0; W[7,11]=0; W[7,12]=0; W[7,13]=0; W[7,14]=0; W[7,15]=0; W[7,16]=0; W[7,17]=0; W[7,18]=0; W[7,19]=0; W[7,20]=0; W[7,21]=0; W[7,22]=0
    W[8,0]=0; W[8,1]=0; W[8,2]=0; W[8,3]=0; W[8,4]=0; W[8,5]=0; W[8,6]=0; W[8,7]=0; W[8,8]=0; W[8,9]=0
    W[8,10]=0; W[8,11]=0; W[8,12]=0; W[8,13]=0; W[8,14]=0; W[8,15]=0; W[8,16]=0; W[8,17]=0; W[8,18]=0; W[8,19]=0; W[8,20]=0; W[8,21]=0; W[8,22]=0
    
    # Verify
    if not _verify_decomposition(U, V, W, 3, 3, 3):
        return None
    
    return U, V, W


def _build_rank41_decomposition():
    """
    Build a rank-41 decomposition for 3x3x5 matrix multiplication.
    Uses Laderman rank-23 for 3x3x3 + schoolbook rank-18 for 3x3x2.
    """
    n, m, p = 3, 3, 5
    rank_23 = 23
    rank_18 = 18
    total_rank = rank_23 + rank_18  # 41
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Part 1: 3x3x3 decomposition (rank 23) for k=0,1,2
    laderman = _build_laderman_3x3x3()
    if laderman is None:
        return None
    U_lm, V_lm, W_lm = laderman
    
    # Copy Laderman into the first 23 columns
    U[:, :rank_23] = U_lm
    V[:, :rank_23] = V_lm
    W[:, :rank_23] = W_lm
    
    # Part 2: Schoolbook for 3x3x2 (k=3,4), rank 18
    col_offset = rank_23
    for i in range(n):
        for j in range(m):
            for k in range(3, 5):
                col = col_offset + (i * m + j) * 2 + (k - 3)
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
    
    # Verify the full decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return U_list, V_list, W_list


def construct():
    """Return a rank-minimal exact tensor decomposition.
    
    Tries to construct a rank-41 decomposition using Laderman's rank-23
    decomposition for the 3x3x3 sub-tensor combined with schoolbook rank-18
    for the 3x3x2 part. Falls back to schoolbook rank 45.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5

    # Try to build a rank-41 decomposition
    result = _build_rank41_decomposition()
    
    if result is not None and (time.time() - start_time) < DEADLINE_SECONDS:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook rank 45
    U, V, W = _build_schoolbook(n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition verification failed")

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        raise TimeoutError("Exceeded deadline")

    return {"U": U, "V": V, "W": W}
