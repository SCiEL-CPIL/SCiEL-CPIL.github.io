import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition."""
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx][col] = 1
                V_sb[v_idx][col] = 1
                W_sb[w_idx][col] = 1
                col += 1
    
    # Attempt Rank 41 (23 + 18)
    # 23 terms for 3x3x3
    # 18 terms for 3x3x2 (schoolbook for k=3,4)
    # This requires the specific 23 terms.
    # If I don't have them, I return schoolbook.
    
    # Placeholder for 23 terms (will fail verification if incorrect)
    # Using a dummy list to ensure fallback to schoolbook if not implemented correctly
    terms_23 = [] # Empty means we can't form the 23-rank part correctly without the data
    
    # Since I cannot reliably recall the 23 terms, I will return the schoolbook solution.
    # This is a safe bet.
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
