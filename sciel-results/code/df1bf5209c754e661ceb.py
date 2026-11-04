"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    deadline = 120.0  # must finish well before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Use the Bini construction for M_{n,m,p} with m <= n, p
    # Rank = n*p*(m-1) + (n + p - 1)
    # For n=4, m=4, p=5: 4*5*3 + (4+5-1) = 60 + 8 = 68
    
    rank_j0 = n + p - 1  # 8
    rank_total = n * p * (m - 1) + rank_j0  # 68
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Terms for j = 1 to m-1
    # Each term isolates one (i, j, k) triple
    for j in range(1, m):
        for i in range(n):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 2: Terms for j = 0
    # Decomposition of the "j=0 slice" with rank n+p-1
    # The tensor slice for j=0 has entries T[i, 0, k] = 1 for all i, k
    # We need: sum_r U_r[i,0] * V_r[0,k] * W_r[k,i] = 1 for all i,k
    
    # Construction:
    # For r = 0 to n-2:
    #   U_r[i,0] = 1 if i == r, else 0
    #   V_r[0,k] = 1 if k == 0, else 0  
    #   W_r[k,i] = 1 if k == 0 and i == r, else 0
    # For r = n-1 to n+p-2:
    #   s = r - (n-1), s in 0..p-1
    #   U_r[i,0] = 1 for all i
    #   V_r[0,k] = 1 if k == s, else 0
    #   W_r[k,i] = 1 if k == s, else 0
    
    # Let's verify: sum over r of U_r[i,0]*V_r[0,k]*W_r[k,i]
    # = sum_{r=0}^{n-2} [i==r]*[k==0]*[k==0 and i==r] + sum_{s=0}^{p-1} [1]*[k==s]*[k==s]
    # = [i < n-1 and k == 0] + [k == k]  (second sum: when s=k, we get 1*1*1=1, else 0)
    # Wait, that gives [i<n-1 and k==0] + 1, which is not always 1.
    
    # This construction is wrong. Let me use a different approach.
    
    # Actually, the correct Bini construction uses a different decomposition.
    # Let me use the known valid construction for rank 77 instead, which is safer.
    
    # Reset and use schoolbook with a small optimization
    # For safety, use the straightforward construction that we know is correct
    rank = n * m * p  # 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
