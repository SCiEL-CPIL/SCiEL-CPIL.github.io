"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 120  # seconds, with safety margin

def construct():
    """
    Rank 50 decomposition for 2x3 matrix times 3x9 matrix.
    Split middle dimension: 3 = 2 + 1
    A = [A_1 | A_2], A_1 is 2x2, A_2 is 2x1
    B = [B_1; B_2], B_1 is 2x9, B_2 is 1x9
    C = A_1*B_1 + A_2*B_2
    
    For A_1*B_1 (2x2x9): split 9 = 2+2+2+2+1
    Each 2x2x2 uses Strassen (rank 7), last 2x2x1 uses schoolbook (rank 4)
    Total: 4*7 + 4 = 32
    
    For A_2*B_2 (2x1x9): schoolbook rank = 18
    Total rank = 50
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 27 x rank
    W = [[0] * rank for _ in range(p * n)]  # 18 x rank
    
    col = 0
    
    # Part 1: A_1 * B_1 where A_1 is 2x2 (j=0,1), B_1 is 2x9 (j=0,1, k=0..8)
    # Split k into groups of 2, last group has 1
    # k groups: [0,1], [2,3], [4,5], [6,7], [8]
    
    # Strassen for 2x2x2: 7 multiplications
    # For matrices A (2x2), B (2x2), C = A*B (2x2):
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A10) B11
    # M6 = (A11 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [1, 0, 1, 0],  # row 0 (i=0): A00, A01, A10, A11 -> but we use (i,j) indexing
        [1, 1, 0, 1],
    ]
    # Actually let me be more careful. A is 2x2 with A[i,j] for i,j in 0,1
    # In our U, index is i*m + j = i*3 + j, but for A_1 part, j is local (0,1)
    # Global j = local_j (since A_1 uses global j=0,1)
    # So U index for A_1[i, local_j] = i*3 + local_j
    
    # Strassen coefficients for A (2x2):
    # M1: A00 + A11 -> [1, 0, 0, 1] for (A00, A01, A10, A11)
    # M2: A10 + A11 -> [0, 0, 1, 1]
    # M3: A00 -> [1, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 1]
    # M5: A00 + A10 -> [1, 0, 1, 0]
    # M6: A11 - A00 -> [-1, 0, 0, 1]
    # M7: A01 - A11 -> [0, 1, 0, -1]
    
    strassen_A = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 0, 1, 0],   # M5
        [-1, 0, 0, 1],  # M6
        [0, 1, 0, -1],  # M7
    ]
    
    # Strassen coefficients for B (2x2): B[j,k] for j,k in 0,1
    # M1: B00 + B11 -> [1, 0, 0, 1] for (B00, B01, B10, B11)
    # M2: B00 -> [1, 0, 0, 0]
    # M3: B01 - B11 -> [0, 1, 0, -1]
    # M4: B10 - B00 -> [-1, 0, 1, 0]
    # M5: B11 -> [0, 0, 0, 1]
    # M6: B00 + B01 -> [1, 1, 0, 0]
    # M7: B10 + B11 -> [0, 0, 1, 1]
    
    strassen_B = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    
    # Strassen coefficients for C (2x2): C[i,k] for i,k in 0,1
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So for each M term, coefficient in C:
    strassen_C = [
        [1, 0, 0, 0],   # M1: C00
        [0, 0, 1, 0],   # M2: C10
        [0, 1, 0, 0],   # M3: C01
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 1],   # M7: C00 + C11
    ]
    
    # k groups for B_1 (2x9): [0,1], [2,3], [4,5], [6,7], [8]
    k_groups = [(0, 1), (2, 3), (4, 5), (6, 7), (8,)]
    
    for group in k_groups:
        if len(group) == 2:
            # 2x2x2 block: Strassen rank 7
            k_start, k_end = group
            for r in range(7):
                # A coefficients
                for i in range(2):
                    for j in range(2):
                        # Local j = j, global j = j (since A_1 uses global j=0,1)
                        u_idx = i * 3 + j
                        U[u_idx][col] += strassen_A[r][i * 2 + j]
                
                # B coefficients: B[j, k] for j in 0,1 (global j), k in k_start..k_end (local)
                for j in range(2):
                    for lk in range(2):
                        k_global = k_start + lk
                        v_idx = j * 9 + k_global
                        V[v_idx][col] += strassen_B[r][j * 2 + lk]
                
                # C coefficients: C[i, k] for i in 0,1, k local
                for i in range(2):
                    for lk in range(2):
                        k_global = k_start + lk
                        w_idx = k_global * 2 + i
                        W[w_idx][col] += strassen_C[r][i * 2 + lk]
                
                col += 1
        else:
            # 2x2x1 block: schoolbook rank 4
            k_start = group[0]
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j][col] = 1
                    V[j * 9 + k_start][col] = 1
                    W[k_start * 2 + i][col] = 1
                    col += 1
    
    # Part 2: A_2 * B_2 where A_2 is 2x1 (global j=2), B_2 is 1x9 (global j=2)
    # Schoolbook: rank = 2*1*9 = 18
    for i in range(2):
        for k in range(9):
            U[i * 3 + 2][col] = 1
            V[2 * 9 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

