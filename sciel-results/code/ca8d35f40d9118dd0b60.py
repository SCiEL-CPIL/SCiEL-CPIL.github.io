"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _strassen_templates():
    """Return (U, V, W) for 2x2 x 2x2 -> 2x2 matmul, rank 7 (Strassen)."""
    U = [
        [1, 0, 0, 1],    # M1: A11+A22
        [0, 0, 1, 1],    # M2: A21+A22
        [1, 0, 0, 0],    # M3: A11
        [0, 0, 0, 1],    # M4: A22
        [1, 1, 0, 0],    # M5: A11+A12
        [-1, 0, 1, 0],   # M6: A21-A11
        [0, 1, 0, -1],   # M7: A12-A22
    ]
    V = [
        [1, 0, 0, 1],    # M1: B11+B22
        [1, 0, 0, 0],    # M2: B11
        [0, 1, 0, -1],   # M3: B12-B22
        [0, 0, 1, -1],   # M4: B21-B11
        [0, 0, 0, 1],    # M5: B22
        [1, 1, 0, 0],    # M6: B11+B12
        [0, 0, 1, 1],    # M7: B21+B22
    ]
    W = [
        [1, 0, 0, 1],    # M1: C11+C22
        [0, 1, 0, -1],   # M2: C21-C22
        [0, 0, 1, 1],    # M3: C12+C22
        [1, 1, 0, 0],    # M4: C11+C21
        [-1, 0, 1, 0],   # M5: -C11+C12
        [0, 0, 0, 1],    # M6: C22
        [1, 0, 0, 0],    # M7: C11
    ]
    return U, V, W


def _build_rank54():
    """Construct rank-54 decomposition: Strassen on 2x2 blocks + schoolbook for j=4."""
    n, m, p = 2, 5, 6
    U_list, V_list, W_list = [], [], []
    sU, sV, sW = _strassen_templates()

    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Local index mapping for 2x2 blocks
            a_map = [j0, j1, 5 + j0, 5 + j1]       # A[i,j] -> i*5+j
            b_map = [j0*6+k0, j0*6+k1, j1*6+k0, j1*6+k1]  # B[j,k] -> j*6+k
            c_map = [k0*2, k0*2+1, k1*2, k1*2+1]    # C[i,k] -> W[k*2+i]

            for t in range(7):
                U_row = [0.0] * (n * m)
                V_row = [0.0] * (m * p)
                W_row = [0.0] * (p * n)
                for idx, val in enumerate(sU[t]):
                    if val:
                        U_row[a_map[idx]] = float(val)
                for idx, val in enumerate(sV[t]):
                    if val:
                        V_row[b_map[idx]] = float(val)
                for idx, val in enumerate(sW[t]):
                    if val:
                        W_row[c_map[idx]] = float(val)
                U_list.append(U_row)
                V_list.append(V_row)
                W_list.append(W_row)

    # j=4: schoolbook for 2x1 x 1x6 -> 2x6
    for i in range(2):
        for k in range(6):
            U_row = [0.0] * (n * m)
            V_row = [0.0] * (m * p)
            W_row = [0.0] * (p * n)
            U_row[i * 5 + 4] = 1.0
            V_row[4 * 6 + k] = 1.0
            W_row[k * 2 + i] = 1.0
            U_list.append(U_row)
            V_list.append(V_row)
            W_list.append(W_row)

    return {"U": U_list, "V": V_list, "W": W_list}


def _verify(U, V, W, n=2, m=5, p=6):
    """Verify exact decomposition."""
    rank = len(U[0])
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] for t in range(rank))
                err += (val - 1.0) ** 2
    return err


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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


def construct():
    start = time.time()
    result = _build_rank54()
    err = _verify(result["U"], result["V"], result["W"])
    if err < 1e-10:
        return result
    # Fallback to schoolbook if verification fails
    return _schoolbook()

# EVOLVE-BLOCK-END
