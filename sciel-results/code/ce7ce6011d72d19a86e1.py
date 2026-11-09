"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    # Use a structured low-rank decomposition
    # Strategy: decompose 2x4x6 as sum of simpler tensors
    # Rank-32 decomposition via block structure
    
    # We'll build a rank-32 decomposition:
    # Split m=4 into two groups of 2, p=6 into three groups of 2
    # For each (i_block, k_block) pair, use a 2x2x2 Strassen-like decomposition
    
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index mapping:
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # i in [0,n), j in [0,m), k in [0,p)
    
    # Decomposition: split j into {0,1} and {2,3}, k into {0,1}, {2,3}, {4,5}
    # For each pair (j_group, k_group), use a 2x2x2 tensor decomposition
    
    # 2x2x2 Strassen has rank 7, but we need exact half-integer entries
    # Use the standard rank-8 decomposition for 2x2x2 (schoolbook) but optimized
    
    # Actually, let's use a simpler approach: 
    # For 2x2x2, the minimal rank is 7 (Strassen)
    # For 2x2x3, we can use rank 12 (two 2x2x2 blocks with sharing)
    # For 2x4x6, we can combine these
    
    # Let me use a direct construction with rank 32:
    # 4 j-groups of size 1, 6 k-groups of size 1, but share across i
    
    # Simpler: use the fact that 2x4x6 = 2*(2x2x6) + 2*(2x2x6)
    # Each 2x2x6 can be done in rank 12 (using 3 blocks of 2x2x2 with rank 4 each)
    # Total: 2*12 + 2*12 = 48... that's not better.
    
    # Let me try a different approach: use rank 24
    # 2x4x6: fix i, we have 4x6 matrices. The tensor is T[i,j,k] = delta
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) valid
    
    # For exact decomposition with small entries, let's use:
    # rank = 24, with entries in {-1, 0, 1}
    
    # Construct via: for each i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # We need exactly one term to fire
    
    # Use a structured approach: 
    # Term c = i*12 + j*3 + k//2  (for even k)
    # Term c' = i*12 + j*3 + k//2 + 6  (for odd k)
    # This gives rank 24
    
    rank = 24
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * (p // 2)) + j * (p // 2) + (k // 2)
                if k % 2 == 0:
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                else:
                    c = i * (m * (p // 2)) + j * (p // 2) + (k // 2) + m * (p // 2)
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = np.zeros((n * m, rank), dtype=float)
                    V = np.zeros((m * p, rank), dtype=float)
                    W = np.zeros((p * n, rank), dtype=float)
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj, col] = 1.0
                                V[jj * p + kk, col] = 1.0
                                W[kk * n + ii, col] = 1.0
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
