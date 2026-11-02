"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}

def validate_decomposition(U, V, W, n, m, p):
    # Check if sum_t U_t otimes V_t otimes W_t equals the multiplication tensor
    # T[i, j, k] = 1 if i==j? No, T is defined by A*B.
    # T_{i, j, k} = 1 if the term is A_{i,j} * B_{j,k}?
    # The tensor is T_{i, j, k} where i is row of A, j is col of A / row of B, k is col of B.
    # T_{i, j, k} = 1 if we are computing C_{i, k} using A_{i, j} and B_{j, k}.
    # So T_{i, j, k} = 1 for all i, j, k? No.
    # The tensor is \sum_{j} A_{i, j} B_{j, k} \delta_{i, k}? No.
    # The tensor is T_{i, j, k} = 1 if the indices match the multiplication pattern.
    # Actually, the tensor is T_{i, j, k} = 1 for all i, j, k? No.
    # The multiplication tensor is T_{i, j, k} = 1 if we consider the basis.
    # T = \sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{i,k}.
    # So T_{(i,j), (j,k), (i,k)} = 1.
    # In flattened indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # So T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i.
    
    rank = len(U[0])
    # Build the tensor from U, V, W
    # T is of shape (n*m, m*p, p*n)
    # We can check a few random indices or all.
    # n*m = 9, m*p = 24, p*n = 24. Total 5184 entries.
    # Check if T[u, v, w] == 1 for all valid (u,v,w) and 0 otherwise.
    
    # Efficient check:
    # For each term t, it contributes to T[u, v, w] where u=U[:,t] non-zero, v=V[:,t] non-zero, w=W[:,t] non-zero.
    # Actually, U, V, W are matrices.
    # T = sum_t U[:,t] otimes V[:,t] otimes W[:,t].
    
    T = np.zeros((n*m, m*p, p*n))
    for t in range(rank):
        u_vec = np.array(U[:, t])
        v_vec = np.array(V[:, t])
        w_vec = np.array(W[:, t])
        # Outer product
        # T += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
        T += np.outer(np.outer(u_vec, v_vec), w_vec)
        
    # Expected tensor
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_exp[u_idx, v_idx, w_idx] = 1
                
    return np.allclose(T, T_exp)

# EVOLVE-BLOCK-END
