"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition (Rank 8)
    # U[i*m+j] -> A[i,j]
    # V[j*p+k] -> B[j,k]
    # W[k*n+i] -> C[i,k]
    # W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11
    
    # 1. A00*B00 -> C00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # 2. A01*B10 -> C00
    U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
    # 3. A00*B01 -> C01
    U[0][2] = 1; V[1][2] = 1; W[2][2] = 1
    # 4. A01*B11 -> C01
    U[1][3] = 1; V[3][3] = 1; W[2][3] = 1
    # 5. A10*B00 -> C10
    U[2][4] = 1; V[0][4] = 1; W[1][4] = 1
    # 6. A11*B10 -> C10
    U[3][5] = 1; V[2][5] = 1; W[1][5] = 1
    # 7. A10*B01 -> C11
    U[2][6] = 1; V[1][6] = 1; W[3][6] = 1
    # 8. A11*B11 -> C11
    U[3][7] = 1; V[3][7] = 1; W[3][7] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
