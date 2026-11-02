import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 110

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def build_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            # M1
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M2
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = -1
            col += 1
            # M3
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M4
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = -1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M5
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            col += 1
            # M6
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (0 + j_off)][col] = -1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M7
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = -1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            col += 1

        # Block 1: j=0 (schoolbook: 2*5 = 10 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
        # Block 2: j=1,2, k=0,1 (Strassen: 7 terms)
        add_strassen(0, 1, 0)
        
        # Block 3: j=1,2, k=2 (schoolbook: 2*2 = 4 terms)
        for i in range(n):
            for j in [1, 2]:
                U[i * m + j][col] = 1
                V[j * p + 2][col] = 1
                W[2 * n + i][col] = 1
                col += 1
                
        # Block 4: j=1,2, k=3,4 (Strassen: 7 terms)
        add_strassen(0, 1, 3)
        
        assert col == rank, f"Column count mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    res_28 = build_rank_28()
    if verify(res_28["U"], res_28["V"], res_28["W"], 28):
        return res_28

    # Fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
