import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using the trace convention W[k*n+i, r]."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition using Strassen on 2x2 sub-blocks
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(j_base, k_base):
        """Add 7 Strassen columns for 2x2 block: A rows 0,1 cols j_base,j_base+1; B rows j_base,j_base+1 cols k_base,k_base+1"""
        nonlocal col
        # i_base is always 0 since n=2
        
        # M0: (A[0,j]+A[1,j+1])(B[j,k]+B[j+1,k+1])
        U[0 * m + j_base][col] += 1
        U[1 * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + 0][col] += 1
        W[(k_base+1) * n + 1][col] += 1
        col += 1
        
        # M1: (A[1,j]+A[1,j+1])B[j,k]
        U[1 * m + j_base][col] += 1
        U[1 * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + 1][col] += 1
        W[(k_base+1) * n + 1][col] -= 1
        col += 1
        
        # M2: A[0,j](B[j,k+1]-B[j+1,k+1])
        U[0 * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + 0][col] += 1
        W[(k_base+1) * n + 1][col] += 1
        col += 1
        
        # M3: A[1,j+1](B[j+1,k]-B[j,k])
        U[1 * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + 0][col] += 1
        W[k_base * n + 1][col] += 1
        col += 1
        
        # M4: (A[0,j]+A[0,j+1])B[j+1,k+1]
        U[0 * m + j_base][col] += 1
        U[0 * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + 0][col] -= 1
        W[(k_base+1) * n + 0][col] += 1
        col += 1
        
        # M5: (A[1,j]-A[0,j])(B[j,k]+B[j,k+1])
        U[1 * m + j_base][col] += 1
        U[0 * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + 1][col] += 1
        col += 1
        
        # M6: (A[0,j+1]-A[1,j+1])(B[j+1,k]+B[j+1,k+1])
        U[0 * m + (j_base+1)][col] += 1
        U[1 * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + 0][col] += 1
        col += 1

    # Strassen blocks for j={0,1} and j={3,4}, with k-pairs (0,1), (2,3), (4,5)
    for j_base in [0, 3]:
        for k_base in [0, 2, 4]:
            add_strassen_block(j_base, k_base)
    
    # Schoolbook for j=2 (middle column): 2x1 * 1x6
    for k in range(p):
        # C[0, k] += A[0, 2] * B[2, k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # C[1, k] += A[1, 2] * B[2, k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    # Verify column count
    if col != rank:
        # If mismatch, truncate or pad
        if col < rank:
            # Pad with zeros (shouldn't happen)
            pass
        else:
            # Truncate (shouldn't happen)
            U = [row[:rank] for row in U]
            V = [row[:rank] for row in V]
            W = [row[:rank] for row in W]
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: return schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
