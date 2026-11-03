import time
import sys

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Define the 6 Strassen blocks
    # I = {0, 1}
    # J pairs: {0, 1}, {2, 3}
    # K pairs: {0, 1}, {2, 3}, {4, 5}
    
    I = [0, 1]
    J_pairs = [(0, 1), (2, 3)]
    K_pairs = [(0, 1), (2, 3), (4, 5)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        # M1: (x00+x11)(y00+y11) -> z00, z11
        u1 = [0] * (n * m)
        u1[i0 * m + j0] = 1
        u1[i1 * m + j1] = 1
        v1 = [0] * (m * p)
        v1[j0 * p + k0] = 1
        v1[j1 * p + k1] = 1
        w1 = [0] * (p * n)
        w1[k0 * n + i0] = 1
        w1[k1 * n + i1] = 1
        U_cols.append(u1)
        V_cols.append(v1)
        W_cols.append(w1)
        
        # M2: (x10+x11)y00 -> z10, z11(-1)
        u2 = [0] * (n * m)
        u2[i1 * m + j0] = 1
        u2[i1 * m + j1] = 1
        v2 = [0] * (m * p)
        v2[j0 * p + k0] = 1
        w2 = [0] * (p * n)
        w2[k0 * n + i1] = 1
        w2[k1 * n + i1] = -1
        U_cols.append(u2)
        V_cols.append(v2)
        W_cols.append(w2)
        
        # M3: x00(y10-y11) -> z01, z11
        u3 = [0] * (n * m)
        u3[i0 * m + j0] = 1
        v3 = [0] * (m * p)
        v3[j1 * p + k0] = 1
        v3[j1 * p + k1] = -1
        w3 = [0] * (p * n)
        w3[k1 * n + i0] = 1
        w3[k1 * n + i1] = 1
        U_cols.append(u3)
        V_cols.append(v3)
        W_cols.append(w3)
        
        # M4: x11(y01-y00) -> z00, z10
        u4 = [0] * (n * m)
        u4[i1 * m + j1] = 1
        v4 = [0] * (m * p)
        v4[j0 * p + k1] = 1
        v4[j0 * p + k0] = -1
        w4 = [0] * (p * n)
        w4[k0 * n + i0] = 1
        w4[k0 * n + i1] = 1
        U_cols.append(u4)
        V_cols.append(v4)
        W_cols.append(w4)
        
        # M5: (x00+x10)y11 -> z00(-1), z01
        u5 = [0] * (n * m)
        u5[i0 * m + j0] = 1
        u5[i1 * m + j0] = 1
        v5 = [0] * (m * p)
        v5[j1 * p + k1] = 1
        w5 = [0] * (p * n)
        w5[k0 * n + i0] = -1
        w5[k1 * n + i0] = 1
        U_cols.append(u5)
        V_cols.append(v5)
        W_cols.append(w5)
        
        # M6: (x11-x00)(y00+y10) -> z11
        u6 = [0] * (n * m)
        u6[i1 * m + j1] = 1
        u6[i0 * m + j0] = -1
        v6 = [0] * (m * p)
        v6[j0 * p + k0] = 1
        v6[j1 * p + k0] = 1
        w6 = [0] * (p * n)
        w6[k1 * n + i1] = 1
        U_cols.append(u6)
        V_cols.append(v6)
        W_cols.append(w6)
        
        # M7: (x01-x11)(y11+y01) -> z00
        u7 = [0] * (n * m)
        u7[i0 * m + j1] = 1
        u7[i1 * m + j1] = -1
        v7 = [0] * (m * p)
        v7[j1 * p + k1] = 1
        v7[j0 * p + k1] = 1
        w7 = [0] * (p * n)
        w7[k0 * n + i0] = 1
        U_cols.append(u7)
        V_cols.append(v7)
        W_cols.append(w7)

    # Add the 6 Strassen blocks
    for j_pair in J_pairs:
        for k_pair in K_pairs:
            add_strassen_block(I[0], I[1], j_pair[0], j_pair[1], k_pair[0], k_pair[1])

    # Add the remaining schoolbook terms
    covered = set()
    for j_pair in J_pairs:
        for k_pair in K_pairs:
            for i in I:
                for j in j_pair:
                    for k in k_pair:
                        covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = [0] * (n * m)
                    u[i * m + j] = 1
                    v = [0] * (m * p)
                    v[j * p + k] = 1
                    w = [0] * (p * n)
                    w[k * n + i] = 1
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)

    # Transpose to get matrices of shape (rows, rank)
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            if U_cols[r][i] != 0:
                U[i][r] = U_cols[r][i]
        for i in range(m * p):
            if V_cols[r][i] != 0:
                V[i][r] = V_cols[r][i]
        for i in range(p * n):
            if W_cols[r][i] != 0:
                W[i][r] = W_cols[r][i]

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    print(f"Time: {time.time() - start:.2f}s")
