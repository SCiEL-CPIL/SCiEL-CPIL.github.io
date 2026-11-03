import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    # n: [0, 1], [2]
    # m: [0, 1], [2, 3], [4]
    # p: [0, 1], [2, 3], [4, 5], [6]
    
    n_bounds = [(0, 2), (2, 3)]
    m_bounds = [(0, 2), (2, 4), (4, 5)]
    p_bounds = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal rank
        U.append([0] * rank + [u_val])
        V.append([0] * rank + [v_val])
        W.append([0] * rank + [w_val])
        rank += 1

    def add_strassen(i_start, i_end, j_start, j_end, k_start, k_end):
        # Assumes size 2x2x2
        i0, i1 = i_start, i_end - 1
        j0, j1 = j_start, j_end - 1
        k0, k1 = k_start, k_end - 1
        
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # M1
        U[u00].append(1)
        U[u11].append(1)
        V[v00].append(1)
        V[v11].append(1)
        W[w00].append(1)
        W[w11].append(1)
        
        # M2
        U[u10].append(1)
        U[u11].append(1)
        V[v00].append(1)
        W[w10].append(1)
        W[w11].append(-1)
        
        # M3
        U[u00].append(1)
        V[v01].append(1)
        V[v11].append(-1)
        W[w01].append(1)
        W[w11].append(1)
        
        # M4
        U[u11].append(1)
        V[v10].append(1)
        V[v00].append(-1)
        W[w00].append(1)
        W[w10].append(1)
        
        # M5
        U[u00].append(1)
        U[u01].append(1)
        V[v11].append(1)
        W[w00].append(-1)
        W[w01].append(1)
        
        # M6
        U[u10].append(1)
        U[u00].append(-1)
        V[v00].append(1)
        V[v01].append(1)
        W[w11].append(1)
        
        # M7
        U[u01].append(1)
        U[u11].append(-1)
        V[v10].append(1)
        V[v11].append(1)
        W[w00].append(1)

    # Re-implementing the correct logic from the current program
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col] = 1
                    V[idx_v][col] = 1
                    W[idx_w][col] = 1
                    col += 1

    def add_strassen(i0, j0, k0):
        nonlocal col
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # M1
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M6
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        col += 1

    # Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 2, 4)
    
    # Schoolbook blocks
    add_schoolbook(0, 2, 0, 2, 6, 7)
    add_schoolbook(0, 2, 2, 4, 6, 7)
    add_schoolbook(0, 2, 4, 5, 0, 2)
    add_schoolbook(0, 2, 4, 5, 2, 4)
    add_schoolbook(0, 2, 4, 5, 4, 6)
    add_schoolbook(0, 2, 4, 5, 6, 7)
    add_schoolbook(2, 3, 0, 2, 0, 2)
    add_schoolbook(2, 3, 0, 2, 2, 4)
    add_schoolbook(2, 3, 0, 2, 4, 6)
    add_schoolbook(2, 3, 0, 2, 6, 7)
    add_schoolbook(2, 3, 2, 4, 0, 2)
    add_schoolbook(2, 3, 2, 4, 2, 4)
    add_schoolbook(2, 3, 2, 4, 4, 6)
    add_schoolbook(2, 3, 2, 4, 6, 7)
    add_schoolbook(2, 3, 4, 5, 0, 2)
    add_schoolbook(2, 3, 4, 5, 2, 4)
    add_schoolbook(2, 3, 4, 5, 4, 6)
    add_schoolbook(2, 3, 4, 5, 6, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    assert rank == 99, f"Expected rank 99, got {rank}"
    
    import random
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_decomp = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                ua = 0
                for jj in range(m):
                    ua += result['U'][i * m + jj][r] * A[i][jj]
                vb = 0
                for jj in range(m):
                    vb += result['V'][jj * p + k][r] * B[jj][k]
                val += result['W'][k * n + i][r] * ua * vb
            C_decomp[i][k] = val
            
    if C_decomp != C:
        print("Verification failed!", file=sys.stderr)
    else:
        print("Verification passed.", file=sys.stderr)
