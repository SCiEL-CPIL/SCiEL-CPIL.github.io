"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _verify_full(U, V, W, n=2, m=2, p=4):
    """Full verification including zero entries."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build expected tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1.0
    
    # Compute decomposition
    for r in range(rank):
        T_comp = np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
    
    # Actually, let's compute it properly
    T_comp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    if np.max(np.abs(T_comp - T)) > 1e-9:
        return False
    return True

def _construct_rank14():
    """Return the known rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def _search_rank12(deadline):
    """Search for a rank-12 decomposition using structured forms."""
    n, m, p = 2, 2, 4
    rank = 12
    
    # Strategy: The 2x2 x 2x4 problem can be viewed as computing C = A * B
    # where A is 2x2, B is 2x4, C is 2x4.
    # 
    # Key idea: We can write B = [B1 | B2] where B1, B2 are 2x2.
    # C = A*B1 | A*B2
    # 
    # For each 2x2x2 multiplication, Strassen uses 7 multiplications.
    # If we can share some multiplications between the two halves, we reduce rank.
    #
    # Let me try a direct construction approach.
    # 
    # Let's parameterize the U, V, W matrices with sparse patterns and solve.
    # 
    # Approach: Use the fact that the problem has a natural block structure.
    # U has 4 rows (indexed by (i,k) for i=0,1 and k=0,1)
    # V has 8 rows (indexed by (k,j) for k=0,1 and j=0,1,2,3)
    # W has 8 rows (indexed by (j,i) for j=0,1,2,3 and i=0,1)
    #
    # Let me try to find a solution where the first 7 rank terms handle B1 (j=0,1)
    # and the last 5 handle B2 (j=2,3), with some sharing.
    #
    # Actually, let me try a more systematic approach:
    # Assume specific sparsity patterns and solve the resulting system.
    
    # Let me try to construct it explicitly using known algebraic identities.
    # 
    # For 2x2 x 2x4, one approach:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    #
    # Total 16 bilinear terms. Rank 12 means 12 multiplications.
    #
    # Let me try to find 12 bilinear forms. I'll use a structured search.
    
    # Try specific structured forms based on Strassen-like patterns
    # with sharing between the two 2x2 blocks.
    
    # Pattern 1: Share the diagonal terms
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])  -- for block 1
    # M2 = (A[0,0]+A[1,1]) * (B[0,2]+B[1,3])  -- for block 2 (shared A factor)
    # M3 = (A[1,0]+A[1,1]) * B[0,0]
    # M4 = (A[1,0]+A[1,1]) * B[0,2]
    # M5 = A[0,0] * (B[1,0]+B[1,1])
    # M6 = A[0,0] * (B[1,2]+B[1,3])
    # M7 = A[1,1] * (B[0,1]-B[1,1])
    # M8 = A[1,1] * (B[0,3]-B[1,3])
    # M9 = (A[0,0]+A[0,1]) * (B[0,0]-B[1,0])
    # M10 = (A[0,0]+A[0,1]) * (B[0,2]-B[1,2])
    # M11 = (A[1,0]+A[1,1]) * (B[0,1]-B[1,1])
    # M12 = (A[1,0]+A[1,1]) * (B[0,3]-B[1,3])
    
    # Let me verify this construction...
    # Actually, let me just try to verify if this works.
    
    U = [[0]*12 for _ in range(4)]
    V = [[0]*12 for _ in range(8)]
    W = [[0]*12 for _ in range(8)]
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U indices: A[0,0]=0, A[1,1]=3; V indices: B[0,0]=0, B[1,1]=5
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    # W: C[0,0] gets +1, C[1,1] gets +1, C[0,1] gets -1, C[1,0] gets -1
    # W indices: C[0,0]=0, C[1,0]=2, C[0,1]=1, C[1,1]=3
    W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = -1
    
    # M2 = (A[0,0]+A[1,1]) * (B[0,2]+B[1,3])
    U[0][1] = 1; U[3][1] = 1
    V[2][1] = 1; V[7][1] = 1
    W[4][1] = 1; W[7][1] = 1; W[5][1] = -1; W[6][1] = -1
    
    # M3 = (A[1,0]+A[1,1]) * B[0,0]
    U[2][2] = 1; U[3][2] = 1
    V[0][2] = 1
    W[2][2] = 1; W[3][2] = -1  # C[1,0] += 1, C[1,1] -= 1
    
    # M4 = (A[1,0]+A[1,1]) * B[0,2]
    U[2][3] = 1; U[3][3] = 1
    V[2][3] = 1
    W[6][3] = 1; W[7][3] = -1
    
    # M5 = A[0,0] * (B[1,0]+B[1,1])
    U[0][4] = 1
    V[4][4] = 1; V[5][4] = 1
    W[0][4] = 1; W[1][4] = -1  # C[0,0] += 1, C[0,1] -= 1
    
    # M6 = A[0,0] * (B[1,2]+B[1,3])
    U[0][5] = 1
    V[6][5] = 1; V[7][5] = 1
    W[4][5] = 1; W[5][5] = -1
    
    # M7 = A[1,1] * (B[0,1]-B[1,1])
    U[3][6] = 1
    V[1][6] = 1; V[5][6] = -1
    W[3][6] = 1; W[2][6] = -1  # C[1,1] += 1, C[1,0] -= 1
    
    # M8 = A[1,1] * (B[0,3]-B[1,3])
    U[3][7] = 1
    V[3][7] = 1; V[7][7] = -1
    W[7][7] = 1; W[6][7] = -1
    
    # M9 = (A[0,0]+A[0,1]) * (B[0,0]-B[1,0])
    U[0][8] = 1; U[1][8] = 1
    V[0][8] = 1; V[4][8] = -1
    W[0][8] = 1; W[2][8] = -1  # C[0,0] += 1, C[1,0] -= 1
    
    # M10 = (A[0,0]+A[0,1]) * (B[0,2]-B[1,2])
    U[0][9] = 1; U[1][9] = 1
    V[2][9] = 1; V[6][9] = -1
    W[4][9] = 1; W[6][9] = -1
    
    # M11 = (A[1,0]+A[1,1]) * (B[0,1]-B[1,1])
    U[2][10] = 1; U[3][10] = 1
    V[1][10] = 1; V[5][10] = -1
    W[3][10] = 1; W[2][10] = -1
    
    # M12 = (A[1,0]+A[1,1]) * (B[0,3]-B[1,3])
    U[2][11] = 1; U[3][11] = 1
    V[3][11] = 1; V[7][11] = -1
    W[7][11] = 1; W[6][11] = -1
    
    # Verify
    if _verify(U, V, W):
        # Full verification
        if _verify_full(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Try another pattern
    # Let me try a different sharing strategy
    
    U = [[0]*12 for _ in range(4)]
    V = [[0]*12 for _ in range(8)]
    W = [[0]*12 for _ in range(8)]
    
    # Different Strassen-like pattern with more sharing
    # M1 = A[0,0]*(B[0,0]+B[1,0]+B[0,2]+B[1,2])
    U[0][0] = 1
    V[0][0] = 1; V[4][0] = 1; V[2][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[2][0] = -1; W[4][0] = 1; W[6][0] = -1
    
    # M2 = A[0,1]*(B[1,0]+B[0,1]+B[1,2]+B[0,3])
    U[1][1] = 1
    V[4][1] = 1; V[1][1] = 1; V[6][1] = 1; V[3][1] = 1
    W[1][1] = 1; W[2][1] = -1; W[5][1] = 1; W[6][1] = -1
    
    # M3 = A[1,0]*(B[0,0]+B[1,1]+B[0,2]+B[1,3])
    U[2][2] = 1
    V[0][2] = 1; V[5][2] = 1; V[2][2] = 1; V[7][2] = 1
    W[2][2] = 1; W[3][2] = -1; W[6][2] = 1; W[7][2] = -1
    
    # M4 = A[1,1]*(B[1,0]+B[0,1]+B[1,2]+B[0,3])
    U[3][3] = 1
    V[4][3] = 1; V[1][3] = 1; V[6][3] = 1; V[3][3] = 1
    W[3][3] = 1; W[2][3] = -1; W[7][3] = 1; W[6][3] = -1
    
    # M5 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[5][4] = 1
    W[0][4] = 1; W[3][4] = 1; W[1][4] = -1; W[2][4] = -1
    
    # M6 = (A[0,0]+A[1,1])*(B[0,2]+B[1,3])
    U[0][5] = 1; U[3][5] = 1
    V[2][5] = 1; V[7][5] = 1
    W[4][5] = 1; W[7][5] = 1; W[5][5] = -1; W[6][5] = -1
    
    # M7 = (A[0,0]+A[0,1])*(B[0,0]-B[1,0])
    U[0][6] = 1; U[1][6] = 1
    V[0][6] = 1; V[4][6] = -1
    W[0][6] = 1; W[2][6] = -1
    
    # M8 = (A[0,0]+A[0,1])*(B[0,2]-B[1,2])
    U[0][7] = 1; U[1][7] = 1
    V[2][7] = 1; V[6][7] = -1
    W[4][7] = 1; W[6][7] = -1
    
    # M9 = (A[1,0]+A[1,1])*(B[0,1]-B[1,1])
    U[2][8] = 1; U[3][8] = 1
    V[1][8] = 1; V[5][8] = -1
    W[3][8] = 1; W[2][8] = -1
    
    # M10 = (A[1,0]+A[1,1])*(B[0,3]-B[1,3])
    U[2][9] = 1; U[3][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[7][9] = 1; W[6][9] = -1
    
    # M11 = (A[0,1]+A[1,1])*(B[1,0]-B[0,0])
    U[1][10] = 1; U[3][10] = 1
    V[4][10] = 1; V[0][10] = -1
    W[1][10] = 1; W[2][10] = -1
    
    # M12 = (A[0,1]+A[1,1])*(B[1,2]-B[0,2])
    U[1][11] = 1; U[3][11] = 1
    V[6][11] = 1; V[2][11] = -1
    W[5][11] = 1; W[6][11] = -1
    
    if _verify(U, V, W):
        if _verify_full(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find rank-12 solution
    result = _search_rank12(deadline)
    if result is not None:
        return result
    
    # Fall back to rank 14
    return _construct_rank14()
# EVOLVE-BLOCK-END
