import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8

    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16

    rank = 44

    U = np.zeros((dim_U, rank), dtype=int)
    V = np.zeros((dim_V, rank), dtype=int)
    W = np.zeros((dim_W, rank), dtype=int)

    col = 0

    # Part 1: 2x1x8 (column j=0 of A, row j=0 of B)
    for k in range(p):
        # C[0,k] = A[0,0] * B[0,k]
        U[0, col] = 1
        V[0 * p + k, col] = 1
        W[k * n + 0, col] = 1
        col += 1

        # C[1,k] = A[1,0] * B[0,k]
        U[1 * m + 0, col] = 1
        V[0 * p + k, col] = 1
        W[k * n + 1, col] = 1
        col += 1

    # Part 2: 2x2x8 (columns j=1,2 of A, rows j=1,2 of B)
    # Split into 4 blocks of 2x2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        # P1 = (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1, col] = 1
        U[1 * m + 2, col] = 1
        V[1 * p + k0, col] = 1
        V[2 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # P2 = (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1, col] = 1
        U[1 * m + 2, col] = 1
        V[1 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        W[k1 * n + 1, col] = -1
        col += 1

        # P3 = A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1, col] = 1
        V[1 * p + k1, col] = 1
        V[2 * p + k1, col] = -1
        W[k1 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # P4 = A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2, col] = 1
        V[2 * p + k0, col] = 1
        V[1 * p + k0, col] = -1
        W[k0 * n + 0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1

        # P5 = (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1, col] = 1
        U[0 * m + 2, col] = 1
        V[2 * p + k1, col] = 1
        W[k0 * n + 0, col] = -1
        W[k1 * n + 0, col] = 1
        col += 1

        # P6 = (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1, col] = 1
        U[0 * m + 1, col] = -1
        V[1 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # P7 = (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2, col] = 1
        U[1 * m + 2, col] = -1
        V[2 * p + k0, col] = 1
        V[2 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        col += 1

    # Convert to list of lists of ints
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
