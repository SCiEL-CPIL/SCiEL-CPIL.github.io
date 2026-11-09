import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    rank = 84  # n*m*p, schoolbook rank

    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Schoolbook decomposition:
    # The tensor for matrix multiplication M_{i,j}^{(n x m)} * N_{j,k}^{(m x p)} = P_{i,k}^{(n x p)}
    # is decomposed as sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] (with appropriate index mapping)
    # For schoolbook: c = (i, j, k) triple
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1  
    # W[k*n + i, c] = 1
    # where c = i*m*p + j*p + k

    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
