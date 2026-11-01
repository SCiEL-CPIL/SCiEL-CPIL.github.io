import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Use rank-7 decomposition for 2x3x3 blocks
    # 2x3x9 is 3 blocks of 2x3x3
    rank_small = 7
    n_small, m_small, p_small = 2, 3, 3
    
    U_small = np.zeros((n_small * m_small, rank_small), dtype=np.int64)
    V_small = np.zeros((m_small * p_small, rank_small), dtype=np.int64)
    W_small = np.zeros((p_small * n_small, rank_small), dtype=np.int64)
    
    # Define the 7 terms for 2x3x3
    # Indices: A: 0..5 (00,01,02,10,11,12)
    # B: 0..8 (00,01,02,10,11,12,20,21,22)
    # C: 0..5 (00,01,02,10,11,12) -> k*n+i -> 0,1,2,3,4,5
    
    # M1: (a00+a10)(b00+b10)
    U_small[0, 0] = 1; U_small[3, 0] = 1
    V_small[0, 0] = 1; V_small[3, 0] = 1
    W_small[0, 0] = 1/4; W_small[1, 0] = 1/4; W_small[3, 0] = 1/4; W_small[4, 0] = 1/4 # This is not integer.
    
    # The task requires integer/half-integer entries.
    # The rank-7 algorithm uses fractions 1/4.
    # We can multiply by 4 to make them integers?
    # If we multiply U, V, W by 4, the result is scaled by 64.
    # We need the result to be exactly 1.
    # So we must use fractions or ensure the sum works out.
    # The task says "integer/half-integer entries".
    # 1/4 is not half-integer.
    # So this specific algorithm might not be allowed if it requires 1/4.
    # However, we can scale the whole decomposition?
    # No, the entries must be integer or half-integer.
    # So we need a decomposition with entries in {0, 1, -1, 1/2, -1/2}.
    
    # Does the rank-7 algorithm use 1/4?
    # Yes, Strassen uses 1/4.
    # But maybe there is a different rank-7 algorithm with integer/half-integer?
    # Or maybe we can use a rank-8 algorithm with integers?
    # Rank of 2x3x3 is 7.
    # If rank 7 requires 1/4, we cannot use it directly.
    # We can try rank 8 or 9.
    
    # Let's try to find a rank-9 algorithm with integer entries.
    # Or just use schoolbook rank 54.
    # But 54 is the current score.
    # If I use rank 27 (3 * 9), score is 2.0.
    # Can I find a rank-9 algorithm for 2x3x3 with integer entries?
    # Schoolbook for 2x3x3 is rank 18.
    # Can we do 9?
    # Yes, if we use Strassen-like on 2x2x2 parts?
    # 2x3x3 can be seen as 2x2x2 + 2x1x2 + 2x2x1 + 2x1x1?
    # This is getting complicated.
    
    # I will stick to schoolbook to be safe, but I will try to optimize the construction time.
    # Actually, the current program already uses schoolbook if the search fails.
    # The search is slow and likely fails.
    # I will remove the search and just return schoolbook.
    # But that gives score 1.0.
    
    # Is there a simple rank-9 decomposition for 2x3x3?
    # Let's try to use the rank-7 algorithm but scale by 4?
    # If I use 1, -1, 0 in U, V, W, the result is integer.
    # But the reconstruction of C requires dividing by 4.
    # The tensor decomposition is T = sum U_c V_c W_c.
    # The entries of T are 0 or 1.
    # If I use integer entries, the sum is integer.
    # So I need integer entries that sum to 1.
    # Strassen's 1/4 is a problem.
    
    # I will use the schoolbook decomposition.
    return schoolbook()
