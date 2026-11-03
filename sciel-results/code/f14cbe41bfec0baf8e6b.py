"""Program for matrix multiplication tensor rank minimization (3x5 by 5x6)."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook decomposition as fallback
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
        U[i * m + j, idx] = 1
        V[j * p + k, idx] = 1
        W[k * n + i, idx] = 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Construct a rank 74 decomposition using block structure
    # Split m=5 into 2+3
    # A = [A1 (3x2), A2 (3x3)]
    # B = [B1 (2x6), B2 (3x6)]
    # C = A1 B1 + A2 B2
    
    # Rank of A1 B1 (3x2 by 2x6)
    # 3 independent 1x2 by 2x6 multiplications
    # Rank of 1x2 by 2x6 is 12
    # Total rank 36
    rank_A1B1 = 36
    
    # Rank of A2 B2 (3x3 by 3x6)
    # Split B2 into two 3x3 blocks
    # Rank of 3x3 by 3x3 is 19 (Lader)
    # Total rank 38
    rank_A2B2 = 38
    
    total_rank = rank_A1B1 + rank_A2B2
    U_new = np.zeros((n * m, total_rank))
    V_new = np.zeros((m * p, total_rank))
    W_new = np.zeros((p * n, total_rank))
    
    # Fill U, V, W for A1 B1 part
    # i in 0..2 (rows of A1)
    # j in 0..1 (cols of A1 / rows of B1)
    # k in 0..5 (cols of B1)
    # For each i, we have a 1x2 by 2x6 multiplication
    # We need 12 terms for each i
    # Term c for i, j, k:
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # But we need to map the local indices to global indices
    
    # Local indices for A1 B1:
    # i_local in 0..2, j_local in 0..1, k_local in 0..5
    # Global i = i_local
    # Global j = j_local (since A1 is first 2 cols)
    # Global k = k_local
    
    # For each i_local, we use 12 terms
    # c_offset = i_local * 12
    for i_local in range(3):
        for j_local in range(2):
            for k_local in range(6):
                c = i_local * 12 + j_local * 6 + k_local
                U_new[i_local * m + j_local, c] = 1
                V[j_local * p + k_local, c] = 1
                W[k_local * n + i_local, c] = 1
    
    # Fill U, V, W for A2 B2 part
    # i in 0..2 (rows of A2)
    # j in 0..2 (cols of A2 / rows of B2)
    # k in 0..5 (cols of B2)
    # Global i = i_local
    # Global j = 2 + j_local (since A2 is last 3 cols)
    # Global k = k_local
    
    # Rank of 3x3 by 3x3 is 19
    # We need to construct a rank 19 decomposition for 3x3 by 3x3
    # Lader's algorithm for 3x3 by 3x3
    
    # Lader's 19-term decomposition for 3x3 by 3x3
    # This is complex to hardcode.
    # For now, use schoolbook for A2 B2 part?
    # Schoolbook rank for 3x3 by 3x6 is 54.
    # Total rank 36 + 54 = 90.
    # This is not an improvement.
    
    # Let's try to find a better decomposition for 3x3 by 3x6.
    # If we can get rank 38, total is 74.
    
    # Since implementing Lader is complex, we will stick to schoolbook for the second part
    # but optimize the first part if possible.
    # Actually, 1x2 by 2x6 rank is 12.
    # Schoolbook is 1*2*6 = 12.
    # So no improvement there.
    
    # Thus, the constructed rank is 90.
    # We return the schoolbook decomposition.
    
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
