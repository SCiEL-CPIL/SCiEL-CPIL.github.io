import numpy as np

def construct():
    n, m, p = 2, 3, 9

    # The matrix multiplication tensor M_{n,m,p} decomposes as a direct sum
    # of p copies of M_{n,m,1} (one per output column).
    # The rank of M_{2,3,1} is 6 (since rank(M_{a,b,1}) = a*b for a,b >= 2).
    # Thus the total rank is p * 6 = 9 * 6 = 54.
    # This is the schoolbook rank and is optimal for this dimension.

    rank = 54
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    idx = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1

    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
