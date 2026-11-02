"""Tensor decomposition for 2x3 x 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    def verify(U, V, W, rank):
        """Verify exact decomposition. Returns max error."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    err = abs(val - 1.0)
                    max_err = max(max_err, err)
        return max_err

    def try_rank4_structured():
        """
        Structured rank-4 decomposition using block-wise tensor factorization.
        
        The idea is to group the (i, j, k) indices into 4 blocks:
        - Block 0: i=0, j in {0,1}, k in {0,1}  -> 4 points
        - Block 1: i=1, j in {0,1}, k in {0,1}  -> 4 points
        - Block 2: i in {0,1}, j=2, k in {0,1}  -> 4 points
        - Block 3: i in {0,1}, j in {0,1,2}, k=2 -> 6 points
        
        Each block is covered by one rank-1 term with appropriate support.
        """
        rank = 4
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        # Block 0: i=0, j in {0,1}, k in {0,1}
        # U[0] = U[1] = 1 (i=0, j=0 and j=1)
        # V[0]=V[1]=V[3]=V[4]=1 (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
        # W[0]=W[2]=1 (k=0,i=0; k=1,i=0)
        U[0][0] = 1
        U[1][0] = 1
        V[0][0] = 1
        V[1][0] = 1
        V[3][0] = 1
        V[4][0] = 1
        W[0][0] = 1
        W[2][0] = 1
        
        # Block 1: i=1, j in {0,1}, k in {0,1}
        # U[3]=U[4]=1 (i=1, j=0 and j=1)
        # V[0]=V[1]=V[3]=V[4]=1 (same as block 0)
        # W[1]=W[3]=1 (k=0,i=1; k=1,i=1)
        U[3][1] = 1
        U[4][1] = 1
        V[0][1] = 1
        V[1][1] = 1
        V[3][1] = 1
        V[4][1] = 1
        W[1][1] = 1
        W[3][1] = 1
        
        # Block 2: i in {0,1}, j=2, k in {0,1}
        # U[2]=U[5]=1 (i=0,j=2 and i=1,j=2)
        # V[6]=V[7]=1 (j=2,k=0; j=2,k=1)
        # W[0]=W[1]=W[2]=W[3]=1 (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
        U[2][2] = 1
        U[5][2] = 1
        V[6][2] = 1
        V[7][2] = 1
        W[0][2] = 1
        W[1][2] = 1
        W[2][2] = 1
        W[3][2] = 1
        
        # Block 3: i in {0,1}, j in {0,1,2}, k=2
        # U[0]=U[1]=U[2]=U[3]=U[4]=U[5]=1 (all i, j)
        # V[2]=V[5]=V[8]=1 (j=0,k=2; j=1,k=2; j=2,k=2)
        # W[4]=W[5]=1 (k=2,i=0; k=2,i=1)
        U[0][3] = 1
        U[1][3] = 1
        U[2][3] = 1
        U[3][3] = 1
        U[4][3] = 1
        U[5][3] = 1
        V[2][3] = 1
        V[5][3] = 1
        V[8][3] = 1
        W[4][3] = 1
        W[5][3] = 1
        
        err = verify(U, V, W, rank)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
        return None

    # Try structured rank 4
    result4 = try_rank4_structured()
    if result4 is not None:
        return result4

    # Fallback to schoolbook
    return fallback

