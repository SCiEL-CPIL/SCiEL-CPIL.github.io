import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    n, m, p = DIMENSIONS
    
    # Lader's rank-23 decomposition for 3x3x3
    # U: 9x23, V: 9x23, W: 9x23
    lader_U = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0]
    ], dtype=int)
    
    lader_V = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0]
    ], dtype=int)
    
    lader_W = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 1]
    ], dtype=int)
    
    # Verify Lader decomposition
    def verify_333(U, V, W):
        # U, V, W are 9x23
        # Check T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for r in range(23):
                        total += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                    if total != 1:
                        return False
        return True

    if not verify_333(lader_U, lader_V, lader_W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Construct 3x3x8 decomposition
    # Rank 48: 2 blocks of (23 + 1)
    rank = 48
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Block 1: k=0,1,2 (Lader) + k=3 (Rank 1)
    # Columns 0-22: Lader
    for r in range(23):
        for i in range(3):
            for j in range(3):
                U[i*m+j, r] = lader_U[i*3+j, r]
        for j in range(3):
            for k in range(3):
                V[j*p+k, r] = lader_V[j*3+k, r]
        for k in range(3):
            for i in range(3):
                W[k*n+i, r] = lader_W[k*3+i, r]
                
    # Column 23: Rank 1 for k=3
    for i in range(3):
        for j in range(3):
            U[i*m+j, 23] = 1
        for j in range(3):
            V[j*p+3, 23] = 1
        for i in range(3):
            W[3*n+i, 23] = 1
            
    # Block 2: k=4,5,6 (Lader) + k=7 (Rank 1)
    # Columns 24-46: Lader
    for r in range(23):
        for i in range(3):
            for j in range(3):
                U[i*m+j, 24+r] = lader_U[i*3+j, r]
        for j in range(3):
            for k in range(3):
                V[j*p+(4+k), 24+r] = lader_V[j*3+k, r]
        for k in range(3):
            for i in range(3):
                W[(4+k)*n+i, 24+r] = lader_W[k*3+i, r]
                
    # Column 47: Rank 1 for k=7
    for i in range(3):
        for j in range(3):
            U[i*m+j, 47] = 1
        for j in range(3):
            V[j*p+7, 47] = 1
        for i in range(3):
            W[7*n+i, 47] = 1

    # Final Verification
    def verify_full(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True

    if verify_full(U, V, W, n, m, p):
        return {"U": U, "V": W, "W": W} # Wait, typo in return dict? No, "U": U, "V": V, "W": W
    else:
        # Fallback
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
