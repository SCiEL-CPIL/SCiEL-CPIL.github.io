import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # We construct the rank 54 decomposition.
    # Split m=5 into {0,1}, {2,3}, {4}
    # Split p=6 into {0,1}, {2,3}, {4,5}
    # n=2 is whole.
    
    # Blocks:
    # 1. j in {0,1}, k in {0,1}, {2,3}, {4,5} -> 3 blocks of 2x2x2
    # 2. j in {2,3}, k in {0,1}, {2,3}, {4,5} -> 3 blocks of 2x2x2
    # 3. j in {4},   k in {0,1}, {2,3}, {4,5} -> 3 blocks of 2x1x2
    
    # Rank 2x2x2 = 7 (Strassen)
    # Rank 2x1x2 = 4 (Trivial)
    
    # Total Rank = 6 * 7 + 3 * 4 = 42 + 12 = 54.
    
    rank = 54
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    # Helper to add a standard term (rank 1)
    def add_standard(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1

    # Helper to add a Strassen block (2x2x2)
    def add_strassen(j1, j2, k1, k2):
        nonlocal col
        # i is always 0,1
        u00 = 0 * m + j1
        u01 = 0 * m + j2
        u10 = 1 * m + j1
        u11 = 1 * m + j2
        
        v00 = j1 * p + k1
        v01 = j1 * p + k2
        v10 = j2 * p + k1
        v11 = j2 * p + k2
        
        w00 = k1 * n + 0
        w01 = k2 * n + 0
        w10 = k1 * n + 1
        w11 = k2 * n + 1
        
        # P1 = (a00+a11)(b00+b11)
        U[u00, col] += 1; U[u11, col] += 1
        V[v00, col] += 1; V[v11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        # P2 = (a10+a11)b00
        U[u10, col] += 1; U[u11, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1; W[w11, col] += -1
        col += 1
        
        # P3 = a00(b01-b11)
        U[u00, col] += 1
        V[v01, col] += 1; V[v11, col] += -1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        # P4 = a11(b10-b00)
        U[u11, col] += 1
        V[v10, col] += 1; V[v00, col] += -1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        # P5 = (a00+a01)b11
        U[u00, col] += 1; U[u01, col] += 1
        V[v11, col] += 1
        W[w00, col] += -1; W[w01, col] += 1
        col += 1
        
        # P6 = (a10-a00)(b00+b01)
        U[u10, col] += 1; U[u00, col] += -1
        V[v00, col] += 1; V[v01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # P7 = (a01-a11)(b10+b11)
        U[u01, col] += 1; U[u11, col] += -1
        V[v10, col] += 1; V[v11, col] += 1
        W[w00, col] += 1
        col += 1

    # Apply Strassen to 2x2x2 blocks
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            add_strassen(j1, j2, k1, k2)
            
    # Add standard terms for j=4
    for k1, k2 in k_pairs:
        for k in [k1, k2]:
            add_standard(0, 4, k)
            add_standard(1, 4, k)
            
    # Verify quickly
    if time.time() < deadline - 5:
        # Check a few random points or all points
        # Since it's small, check all
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if result != 1:
                        # Fallback to trivial
                        rank_f = 60
                        U_f = np.zeros((n * m, rank_f), dtype=np.int8)
                        V_f = np.zeros((m * p, rank_f), dtype=np.int8)
                        W_f = np.zeros((p * n, rank_f), dtype=np.int8)
                        col_f = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    U_f[i * m + j, col_f] = 1
                                    V_f[j * p + k, col_f] = 1
                                    W_f[k * n + i, col_f] = 1
                                    col_f += 1
                        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
