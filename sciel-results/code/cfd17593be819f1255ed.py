"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using random search
    # The tensor is T[i,j,k] = delta_{i,j,k} for matrix multiplication
    # We need U, V, W such that T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    
    # For 2x2 * 2x16, schoolbook rank is 2*2*16 = 64
    # Let's try to find if there's a rank < 64 decomposition
    
    # Strategy: Use the fact that this is a matrix multiplication tensor
    # The tensor T_{i,j,k} for (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # T is a 3-tensor of size (n*m) x (m*p) x (p*n)
    # T[(i,j), (j,k), (k,i)] = 1 if indices are consistent, else 0
    
    # Actually, let me reconsider the tensor structure.
    # The matrix multiplication tensor T has entries T_{(i,j),(j,k),(k,i)} = 1
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # For our case: n=2, m=2, p=16
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (32, rank)  
    # W has shape (p*n, rank) = (32, rank)
    
    # The decomposition must satisfy:
    # For all (a,b) in [0,4)x[0,32), (c,d) in [0,32)x[0,32):
    # sum_c U[a,c] * V[b,c] * W[c,d] = T[a,b,c]
    
    # Where T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # else 0
    
    # Let's verify the schoolbook solution is correct and see if we can do better
    # with a clever structure.
    
    # For now, return the schoolbook solution.
    # A more advanced search could try random rank r < 64 and solve the linear system.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
