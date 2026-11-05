"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strassen 2x2x2 factors
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # Partition m and p into [2, 2, 1]
    # We will create 4 blocks of 3x2x2
    # Each 3x2x2 block has rank 9 (7 from Strassen on top 2x2x2, 2 from schoolbook on bottom 1x2x2)
    
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4]]
    
    # We only use the first two groups for the Strassen blocks
    # The last group (index 4) will be handled by schoolbook
    
    rank_total = 0
    U = []
    V = []
    W = []
    
    # Initialize empty lists, we will build them block by block
    # It is easier to build a list of (row, col, val) tuples and then assemble
    
    u_entries = {}
    v_entries = {}
    w_entries = {}
    
    col_offset = 0
    
    # Define the 4 blocks
    blocks = [
        ([0, 1], [0, 1]),
        ([2, 3], [0, 1]),
        ([0, 1], [2, 3]),
        ([2, 3], [2, 3])
    ]
    
    for j_list, k_list in blocks:
        j0, j1 = j_list
        k0, k1 = k_list
        
        # Strassen part for i in {0, 1}
        # Local i' in {0, 1} maps to global I in {0, 1}
        # Local j' in {0, 1} maps to global J in {j0, j1}
        # Local k' in {0, 1} maps to global K in {k0, k1}
        
        u_rows_s = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
        v_rows_s = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
        w_rows_s = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
        
        for c in range(7):
            for s in range(4):
                u_entries[(u_rows_s[s], col_offset + c)] = u_entries.get((u_rows_s[s], col_offset + c), 0) + U_strassen[s][c]
                v_entries[(v_rows_s[s], col_offset + c)] = v_entries.get((v_rows_s[s], col_offset + c), 0) + V_strassen[s][c]
                w_entries[(w_rows_s[s], col_offset + c)] = w_entries.get((w_rows_s[s], col_offset + c), 0) + W_strassen[s][c]
        
        col_offset += 7
        
        # Schoolbook part for i = 2
        # Terms: (2, j0, k0), (2, j0, k1), (2, j1, k0), (2, j1, k1)
        
        # Term 1: (2, j0, k0)
        u_entries[(2 * m + j0, col_offset)] = u_entries.get((2 * m + j0, col_offset), 0) + 1
        v_entries[(j0 * p + k0, col_offset)] = v_entries.get((j0 * p + k0, col_offset), 0) + 1
        w_entries[(k0 * n + 2, col_offset)] = w_entries.get((k0 * n + 2, col_offset), 0) + 1
        col_offset += 1
        
        # Term 2: (2, j0, k1)
        u_entries[(2 * m + j0, col_offset)] = u_entries.get((2 * m + j0, col_offset), 0) + 1
        v_entries[(j0 * p + k1, col_offset)] = v_entries.get((j0 * p + k1, col_offset), 0) + 1
        w_entries[(k1 * n + 2, col_offset)] = w_entries.get((k1 * n + 2, col_offset), 0) + 1
        col_offset += 1
        
        # Term 3: (2, j1, k0)
        u_entries[(2 * m + j1, col_offset)] = u_entries.get((2 * m + j1, col_offset), 0) + 1
        v_entries[(j1 * p + k0, col_offset)] = v_entries.get((j1 * p + k0, col_offset), 0) + 1
        w_entries[(k0 * n + 2, col_offset)] = w_entries.get((k0 * n + 2, col_offset), 0) + 1
        col_offset += 1
        
        # Term 4: (2, j1, k1)
        u_entries[(2 * m + j1, col_offset)] = u_entries.get((2 * m + j1, col_offset), 0) + 1
        v_entries[(j1 * p + k1, col_offset)] = v_entries.get((j1 * p + k1, col_offset), 0) + 1
        w_entries[(k1 * n + 2, col_offset)] = w_entries.get((k1 * n + 2, col_offset), 0) + 1
        col_offset += 1

    # Remaining terms:
    # 1. j=4, k in {0, 1, 2, 3, 4}
    # 2. k=4, j in {0, 1, 2, 3}
    
    covered = set()
    for j_list, k_list in blocks:
        for j in j_list:
            for k in k_list:
                for i in range(n):
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                
                u_entries[(i * m + j, col_offset)] = u_entries.get((i * m + j, col_offset), 0) + 1
                v_entries[(j * p + k, col_offset)] = v_entries.get((j * p + k, col_offset), 0) + 1
                w_entries[(k * n + i, col_offset)] = w_entries.get((k * n + i, col_offset), 0) + 1
                col_offset += 1

    rank_total = col_offset
    
    # Assemble matrices
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    for (row, col), val in u_entries.items():
        U[row][col] = val
    for (row, col), val in v_entries.items():
        V[row][col] = val
    for (row, col), val in w_entries.items():
        W[row][col] = val

    # Verify correctness
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        U = np.array(U_mat)
        V = np.array(V_mat)
        W = np.array(W_mat)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
