"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    # Use Strassen-like decomposition: split into blocks and apply 2x2 Strassen
    # Split n=3 as 2+1, m=4 as 2+2, p=6 as 2+2+2
    # This gives a structured decomposition
    
    # For 2x2x2 Strassen: rank 7
    # For 2x2x1: rank 4 (schoolbook)
    # For 1x2x2: rank 4 (schoolbook)  
    # For 1x2x1: rank 2 (schoolbook)
    
    # Let's build the full decomposition using Kronecker products of known decompositions
    # M_{2,2,2} has rank 7 (Strassen)
    # M_{2,2,1} has rank 4
    # M_{1,2,2} has rank 4
    # M_{1,2,1} has rank 2
    
    # Split:
    # n=3 = 2+1
    # m=4 = 2+2
    # p=6 = 2+2+2
    
    # Blocks of C (3x6):
    # C[0:2, 0:2] = A[0:2,0:2] * B[0:2,0:2] -> M_{2,2,2}, rank 7
    # C[0:2, 2:4] = A[0:2,0:2] * B[0:2,2:4] -> M_{2,2,2}, rank 7
    # C[0:2, 4:6] = A[0:2,0:2] * B[0:2,4:6] -> M_{2,2,2}, rank 7
    # C[0:2, :] done
    
    # C[2:3, 0:2] = A[2:3,0:2] * B[0:2,0:2] -> M_{1,2,2}, rank 4
    # C[2:3, 2:4] = A[2:3,0:2] * B[0:2,2:4] -> M_{1,2,2}, rank 4
    # C[2:3, 4:6] = A[2:3,0:2] * B[0:2,4:6] -> M_{1,2,2}, rank 4
    
    # Total: 3*7 + 3*4 = 21 + 12 = 33? Wait, that's not right.
    # Actually we need to be more careful.
    
    # Let me just use the schoolbook decomposition which is known to work:
    rank = n * m * p  # 72
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
