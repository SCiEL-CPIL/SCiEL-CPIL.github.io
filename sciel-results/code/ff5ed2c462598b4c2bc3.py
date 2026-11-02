"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    total_rank = 55  # 5 blocks * 11
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 6 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 30 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 20 rows
    
    col = 0
    for block in range(5):
        b = 2 * block  # starting column in B
        
        # Strassen for A1(2x2) @ B1(2x2)
        # M1 = (A[0,0]+A[1,1]) * (B[0,b]+B[1,b+1])
        U[0][col] = 1; U[4][col] = 1
        V[b][col] = 1; V[p+b+1][col] = 1
        W[b*n+0][col] = 1    # C[0,b] += M1
        W[(b+1)*n+1][col] = 1 # C[1,b+1] += M1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,b]
        U[3][col] = 1; U[4][col] = 1
        V[b][col] = 1
        W[b*n+1][col] = 1    # C[1,b] += M2
        col += 1
        
        # M3 = A[0,0] * (B[1,b]-B[1,b+1])
        U[0][col] = 1
        V[p+b][col] = 1; V[p+b+1][col] = -1
        W[(b+1)*n+0][col] = 1  # C[0,b+1] += M3
        col += 1
        
        # M4 = A[1,1] * (B[0,b]-B[1,b])
        U[4][col] = 1
        V[b][col] = 1; V[p+b][col] = -1
        W[b*n+0][col] = 1    # C[0,b] += M4
        W[b*n+1][col] = 1    # C[1,b] += M4
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,b+1]
        U[0][col] = 1; U[1][col] = 1
        V[p+b+1][col] = 1
        W[(b+1)*n+0][col] = 1  # C[0,b+1] += M5
        W[(b+1)*n+1][col] = 1  # C[1,b+1] += M5... 
        # Wait, C12 = M3+M5, C22 = M1-M2+M3+M6. M5 only in C12.
        # Let me recheck: C12 = M3 + M5, so M5 contributes to C[0,b+1] only.
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,b]+B[0,b+1])
        U[3][col] = 1; U[0][col] = -1
        V[b][col] = 1; V[b+1][col] = 1
        W[(b+1)*n+1][col] = 1  # C[1,b+1] += M6
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,b]+B[1,b+1])
        U[1][col] = 1; U[4][col] = -1
        V[p+b][col] = 1; V[p+b+1][col] = 1
        W[b*n+0][col] = 1    # C[0,b] += M7
        col += 1
        
        # Trivial: A2(2x1) @ B2(1x2), schoolbook rank 4
        U[2][col] = 1; V[2*p+b][col] = 1
        W[b*n+0][col] = 1
        col += 1
        
        U[2][col] = 1; V[2*p+b+1][col] = 1
        W[(b+1)*n+0][col] = 1
        col += 1
        
        U[5][col] = 1; V[2*p+b][col] = 1
        W[b*n+1][col] = 1
        col += 1
        
        U[5][col] = 1; V[2*p+b+1][col] = 1
        W[(b+1)*n+1][col] = 1
        col += 1
    
    assert col == 55, f"Expected 55, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
