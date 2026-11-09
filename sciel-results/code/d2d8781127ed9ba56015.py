import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Constructs the schoolbook rank decomposition for matrix multiplication 3x4 * 4x7.
    The rank is n*m*p = 3*4*7 = 84.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s limit
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Schoolbook term: A[i,j] * B[j,k] contributes to C[i,k]
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (as per trace convention W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                
                column += 1
                
                # Check time
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
