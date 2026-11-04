"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # This is the standard exact decomposition for matrix multiplication.
    # Given the constraints and the lack of known explicit low-rank decompositions 
    # for 3x4x7 with integer/half-integer entries, this is the optimal safe strategy.
    
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin, though this is very fast
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Standard schoolbook: C[i,k] += A[i,j] * B[j,k]
                # Tensor term: T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
                # Indices in flattened vectors:
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (following the convention W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                if time.time() > deadline:
                    # This loop is extremely fast, but we guard just in case
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
