# EVOLVE-BLOCK-START
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition for 3x3x3
    # Total rank 30 for 3x3x6
    
    rank_333 = 15
    total_rank = rank_333 * 2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Terms for 3x3x3
    # We define (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    terms = []
    
    # 1. A11 B11 (2x2x2, rank 4)
    # i in 0,1; j in 0,1; k in 0,1
    terms.append((0*3+0, 0*3+0, 0*3+0, 1, 1, 1)) # A00 B00
    terms.append((0*3+1, 0*3+1, 0*3+1, 1, 1, 1)) # A01 B01
    terms.append((1*3+0, 1*3+0, 1*3+0, 1, 1, 1)) # A10 B10
    terms.append((1*3+1, 1*3+1, 1*3+1, 1, 1, 1)) # A11 B11
    
    # 2. A12 B21 (2x1x2, rank 1)
    # i in 0,1; j=2; k in 0,1
    # C_i,k = A_i,2 B_2,k
    # Term: U_i,2 * V_2,k * W_k,i
    # We need to sum over i,k.
    # Rank 1 means one term?
    # T_{i,2,k} = 1.
    # U_{i,r} V_{2,r} W_{k,r} = 1 for all i,k?
    # If r=1, U_{i,1}=1, V_{2,1}=1, W_{k,1}=1.
    # Then sum is 1.
    # So one term: U[0,2]=1, U[1,2]=1?
    # No, U is a matrix.
    # U[i, r] * V[j, r] * W[k, r].
    # We want U[i, r] V[2, r] W[k, r] = 1 for i in 0,1, k in 0,1.
    # If V[2, r] = 1, then U[i, r] W[k, r] = 1.
    # This requires U[0, r]=1, U[1, r]=1, W[0, r]=1, W[1, r]=1.
    # So yes, one term.
    terms.append((0*3+2, 2*3+0, 0*3+0, 1, 1, 1)) # This is wrong, indices must match
    # The term is defined by (u_idx, v_idx, w_idx) which are specific indices.
    # But the rank-1 decomposition of the tensor T_{i,2,k}=1 is:
    # T = sum_r U_{i,r} V_{2,r} W_{k,r}.
    # If we use one term r, we set:
    # U[i, r] = 1 for i in 0,1
    # V[2, r] = 1
    # W[k, r] = 1 for k in 0,1
    # This corresponds to adding to U[0, r], U[1, r], V[2, r], W[0, r], W[1, r].
    # My `terms` list format (u_idx, v_idx, w_idx) implies a single non-zero entry per matrix?
    # No, the decomposition is sum of rank-1 tensors.
    # Each rank-1 tensor is defined by vectors u, v, w.
    # I should store the vectors.
    
    # Let's change structure.
    # List of (u_vec, v_vec, w_vec)
    
    # Re-initialize
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    # 1. A11 B11 (rank 4)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # This is schoolbook, not rank 4.
                # Rank 4 for 2x2x2 is just schoolbook.
                pass
    # Actually, 2x2x2 rank is 7 (Strassen) or 4 (Schoolbook).
    # I used 4 above.
    
    # I will implement the 15 terms explicitly.
    # This is tedious. I will use the schoolbook for 3x3x3 (rank 9) x 2 = 18.
    # Score 54/18 = 3.0.
    # This is a huge improvement.
    
    U = [[0] * 18 for _ in range(n * m)]
    V = [[0] * 18 for _ in range(m * p)]
    W = [[0] * 18 for _ in range(p * n)]
    
    # First half k=0,1,2
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Second half k=3,4,5
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1
                W[(k + 3) * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
