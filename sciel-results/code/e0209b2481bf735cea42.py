import numpy as np
import time

def construct():
    # Dimensions
    n, m, p = 2, 3, 5
    
    # Rank 28 decomposition
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val
        col += 1

    # Block 1: A[:, 0:2] * B[0:2, 0:2] (2x2 * 2x2) -> Strassen Rank 7
    # i1=0, i2=1, j1=0, j2=1, k1=0, k2=1
    i1, i2, j1, j2, k1, k2 = 0, 1, 0, 1, 0, 1
    # M1
    add_term(i1*m+j1, j1*p+k1, k1*n+i1)
    add_term(i2*m+j2, j2*p+k2, k2*n+i2)
    # M2
    add_term(i2*m+j1, j1*p+k1, k1*n+i2)
    add_term(i2*m+j2, j2*p+k2, k2*n+i2, -1)
    # M3
    add_term(i1*m+j1, j1*p+k2, k2*n+i1)
    add_term(i1*m+j1, j2*p+k2, k2*n+i1, -1)
    # M4
    add_term(i2*m+j2, j2*p+k1, k1*n+i1)
    add_term(i2*m+j2, j1*p+k1, k1*n+i1, -1)
    # M5
    add_term(i1*m+j1, j2*p+k2, k1*n+i1, -1)
    add_term(i1*m+j2, j2*p+k2, k2*n+i1)
    # M6
    add_term(i2*m+j1, j1*p+k1, k2*n+i2)
    add_term(i2*m+j1, j1*p+k2, k2*n+i2)
    add_term(i1*m+j1, j1*p+k1, k2*n+i2, -1)
    add_term(i1*m+j1, j1*p+k2, k2*n+i2, -1)
    # M7
    add_term(i1*m+j2, j2*p+k1, k1*n+i1)
    add_term(i1*m+j2, j2*p+k2, k1*n+i1)
    add_term(i2*m+j2, j2*p+k1, k1*n+i1, -1)
    add_term(i2*m+j2, j2*p+k2, k1*n+i1, -1)
    
    # Wait, the manual Strassen expansion above is messy and likely wrong in indices.
    # Let's use the helper functions from the original code which were verified.
    
    # Reset
    col = 0
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A[i1,j1] + A[i2,j2])(B[j1,k1] + B[j2,k2])
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = (A[i2,j1] + A[i2,j2]) B[j1,k1]
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = A[i1,j1] (B[j1,k2] - B[j2,k2])
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M4 = A[i2,j2] (B[j2,k1] - B[j1,k1])
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A[i1,j1] + A[i1,j2]) B[j2,k2]
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A[i2,j1] - A[i1,j1])(B[j1,k1] + B[j1,k2])
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A[i1,j2] - A[i2,j2])(B[j2,k1] + B[j2,k2])
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[:, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [0, 1], [4])
    
    # Block 4: A[:, 2:3] * B[2:3, 0:2] -> C[0:2, 0:2] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [0, 1])
    
    # Block 5: A[:, 2:3] * B[2:3, 2:4] -> C[0:2, 2:4] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [2, 3])
    
    # Block 6: A[:, 2:3] * B[2:3, 4:5] -> C[0:2, 4:5] (Schoolbook, rank 2)
    add_schoolbook([0, 1], [2], [4])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
