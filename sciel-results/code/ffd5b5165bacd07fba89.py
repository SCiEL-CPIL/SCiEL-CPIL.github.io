import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit

    # Rank 7 Strassen decomposition (optimal for 2x2 matrix multiplication)
    rank = 7
    
    # Index conventions:
    # U[i*2+j][r] corresponds to A[i,j]
    # V[j*2+k][r] corresponds to B[j,k]
    # W[k*2+i][r] corresponds to C[i,k] output (Note: W index is k*n+i)
    
    # U shape (4, 7)
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0] (i=0, j=0)
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1] (i=0, j=1)
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0] (i=1, j=0)
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1] (i=1, j=1)
    ]
    
    # V shape (4, 7)
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0] (j=0, k=0)
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1] (j=0, k=1)
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0] (j=1, k=0)
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1] (j=1, k=1)
    ]
    
    # W shape (4, 7)
    # W index = k*2 + i
    # Row 0: k=0, i=0 (C[0,0])
    # Row 1: k=0, i=1 (C[1,0])
    # Row 2: k=1, i=0 (C[0,1])
    # Row 3: k=1, i=1 (C[1,1])
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]
    ]
    
    # Verification
    # T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] == 1
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(rank):
                    s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to rank 8 (should not happen)
        rank = 8
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
