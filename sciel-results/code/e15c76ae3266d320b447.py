import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank_1 = n * m * 3  # 27 for first 3 cols
    rank_2 = n * m * 2  # 18 for last 2 cols? No, we want rank 6.
    
    # Let's use rank 6 for the last 2 cols.
    # And rank 27 for the first 3 cols.
    # Total rank 33.
    
    total_rank = 27 + 6
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: First 3 columns of B (k=0,1,2)
    # Schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
                
    # Part 2: Last 2 columns of B (k=3,4)
    # Rank 6 decomposition
    # We need to compute C_{i,3} and C_{i,4}
    # C_{i,3} = sum_j A_{i,j} B_{j,3}
    # C_{i,4} = sum_j A_{i,j} B_{j,4}
    # This is 2 separate 3x3 vector multiplications.
    # Each has rank 3.
    # For k=3:
    # Terms correspond to (i,j) pairs?
    # No, the rank-1 terms for a dot product sum_j A_{i,j} B_{j,k} are not simple.
    # Actually, the tensor for A(3x3)B(3x1) has rank 3.
    # The decomposition is:
    # For each j in 0..2:
    #   u has 1 at (i,j) for all i? No.
    #   The term is sum_i (A_{i,j} B_{j,k} e_{(i,k)})?
    #   No, a rank-1 term is u \otimes v \otimes w.
    #   u is vector in A-space. v in B-space. w in C-space.
    #   For the term corresponding to index j:
    #   u_j has 1 at positions (i,j) for i=0..2?
    #   v_j has 1 at position (j,k)?
    #   w_j has 1 at positions (i,k) for i=0..2?
    #   Then sum_j u_j \otimes v_j \otimes w_j = sum_j sum_i A_{i,j} B_{j,k} C_{i,k}.
    #   This gives C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #   So for fixed k, we need 3 terms (one for each j).
    #   So for k=3, we need 3 terms.
    #   For k=4, we need 3 terms.
    #   Total 6 terms.
    
    # Let's construct these 6 terms.
    # Term index col
    for k in [3, 4]:
        for j in range(m):
            # u: has 1 at (i, j) for all i?
            # No, u is a vector of length n*m.
            # u[(i, j)] = 1 for all i?
            # If u[(i,j)]=1, then u \otimes v \otimes w contributes to all i.
            # v[(j, k)] = 1.
            # w[(i, k)] = 1 for all i.
            # Then product is sum_i 1 * 1 * A_{i,j} B_{j,k} ?
            # No, the tensor element T[a,b,c] is 1 if a=(i,j), b=(j,k), c=(i,k).
            # The reconstruction is sum_r U[a,r] V[b,r] W[c,r].
            # If we set U[(i,j), r] = 1, V[(j,k), r] = 1, W[(i',k'), r] = 1 if i'=i, k'=k?
            # No, W is a vector.
            # For a specific term r, U is a vector, V is a vector, W is a vector.
            # We want U[a,r] V[b,r] W[c,r] to be non-zero only when a=(i,j), b=(j,k), c=(i,k).
            # So for a fixed j and k:
            # Let r be the term index.
            # V[(j,k), r] = 1.
            # For all i:
            #   U[(i,j), r] = 1.
            #   W[(i,k), r] = 1.
            # Then for a=(i,j), b=(j,k), c=(i,k):
            # U[a,r] V[b,r] W[c,r] = 1 * 1 * 1 = 1.
            # For other c'=(i',k'):
            # If i' != i or k' != k, W[c',r] = 0.
            # So this works.
            
            # So for each (j, k) pair, we have one rank-1 term.
            # There are 3 j's and 2 k's. Total 6 terms.
            
            V[j * p + k][col] = 1
            for i in range(n):
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
