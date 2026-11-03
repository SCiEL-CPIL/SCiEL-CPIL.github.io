import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank 46
    # Part 1: A[0:2, 0:2] * B[0:2, :] (2x2 * 2x5) -> Rank 18
    # Part 2: A[0:2, 2:5] * B[2:5, :] (2x3 * 3x5) -> Rank 28
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2 * 2x2 block
    # I_A: rows of A (size 2)
    # J_A: cols of A (size 2)
    # J_B: rows of B (size 2) - must be same as J_A
    # K_B: cols of B (size 2)
    def add_strassen_2x2(I_A, J_A, J_B, K_B):
        nonlocal col
        # Strassen coefficients for A (2x2)
        # A = [[a, b], [c, d]]
        # M1 = (a+d)(e+h) -> A: [[1, 0], [1, 1]]? No.
        # M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
        # A vec: [1, 0, 1, 1] (flattened row-major: a, b, c, d)
        # M1 A: [1, 0, 1, 1]
        # M1 B: [1, 0, 0, 1]
        
        # M2 = (c+d)e -> A: [0, 0, 1, 1], B: [1, 0, 0, 0]
        # M3 = a(f-h) -> A: [1, 0, 0, 0], B: [0, 1, 0, -1]
        # M4 = d(g-e) -> A: [0, 0, 0, 1], B: [0, -1, 1, 0]
        # M5 = (a+b)h -> A: [1, 1, 0, 0], B: [0, 0, 0, 1]
        # M6 = (c-a)(e+f) -> A: [-1, 0, 1, 0], B: [1, 1, 0, 0]
        # M7 = (b-d)(g+h) -> A: [0, 1, 0, -1], B: [0, 0, 1, 1]
        
        # C reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        strassen_A = [
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        strassen_B = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, -1, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        strassen_C = [
            [1, 1, 0, -1, 1, 0, 0],   # C00 coeffs for M1..M7
            [0, 0, 1, 0, 1, 0, 0],    # C01
            [0, 1, 0, 1, 0, 0, 0],    # C10
            [1, -1, 1, 0, 0, 1, 0]    # C11
        ]
        
        for r in range(7):
            c_idx = col
            # A coefficients
            for ia in range(2):
                for ja in range(2):
                    i = I_A[ia]
                    j = J_A[ja]
                    U[i * m + j][c_idx] = strassen_A[r][ia * 2 + ja]
            # B coefficients
            for jb in range(2):
                for kb in range(2):
                    j = J_B[jb]
                    k = K_B[kb]
                    V[j * p + k][c_idx] = strassen_B[r][jb * 2 + kb]
            # C coefficients
            for ic in range(2):
                for kc in range(2):
                    i = I_A[ic] # Output rows correspond to A rows
                    k = K_B[kc] # Output cols correspond to B cols
                    W[k * n + i][c_idx] = strassen_C[ic * 2 + kc][r]
            col += 1

    # Helper to add a Schoolbook block
    # I_A: list of rows
    # J_A: list of cols
    # J_B: list of rows (inner)
    # K_B: list of cols
    def add_schoolbook(I_A, J_A, J_B, K_B):
        nonlocal col
        for i in I_A:
            for j in J_A:
                # j must be in J_B
                for k in K_B:
                    # The inner index is j. J_A and J_B must share this index.
                    # In our decomposition, J_A == J_B for the block.
                    c_idx = col
                    U[i * m + j][c_idx] = 1
                    V[j * p + k][c_idx] = 1
                    W[k * n + i][c_idx] = 1
                    col += 1

    # Part 1: A[0:2, 0:2] * B[0:2, :]
    # B[0:2, :] is 2x5. Split into B[0:2, 0:2] and B[0:2, 2:5]
    
    # 1.1: A[0:2, 0:2] * B[0:2, 0:2] (2x2 * 2x2) -> Strassen
    add_strassen_2x2([0, 1], [0, 1], [0, 1], [0, 1])
    
    # 1.2: A[0:2, 0:2] * B[0:2, 2:5] (2x2 * 2x3)
    # Split B cols 2,3,4 into col 2 and cols 3,4
    # 1.2.1: A[0:2, 0:2] * B[0:2, [2]] (2x2 * 2x1) -> Schoolbook
    add_schoolbook([0, 1], [0, 1], [0, 1], [2])
    
    # 1.2.2: A[0:2, 0:2] * B[0:2, [3,4]] (2x2 * 2x2) -> Strassen
    add_strassen_2x2([0, 1], [0, 1], [0, 1], [3, 4])

    # Part 2: A[0:2, 2:5] * B[2:5, :]
    # Split A cols 2,3,4 into col 2 and cols 3,4
    # Split B rows 2,3,4 into row 2 and rows 3,4
    # Sum = A[2]B[2] + A[3:5]B[3:5] (cross terms are zero in this partitioning strategy? 
    # Wait. The multiplication is sum over k.
    # C = sum_{k=0}^4 A[:,k] B[k,:]
    # We grouped k=0,1 in Part 1.
    # Part 2 is k=2,3,4.
    # We split k=2 and k=3,4.
    # Term k=2: A[:,2] (2x1) * B[2,:] (1x5). Rank 10.
    # Term k=3,4: A[:,3:5] (2x2) * B[3:5,:] (2x5). Rank 18.
    
    # 2.1: k=2
    add_schoolbook([0, 1], [2], [2], [0, 1, 2, 3, 4])
    
    # 2.2: k=3,4
    # A cols 3,4. B rows 3,4.
    # B[3:5, :] is 2x5. Split into B[3:5, 0:2] and B[3:5, 2:5]
    # 2.2.1: A[0:2, 3:5] * B[3:5, 0:2] (2x2 * 2x2) -> Strassen
    add_strassen_2x2([0, 1], [3, 4], [3, 4], [0, 1])
    
    # 2.2.2: A[0:2, 3:5] * B[3:5, 2:5] (2x2 * 2x3)
    # Split B cols 2,3,4 into col 2 and cols 3,4
    # 2.2.2.1: A[0:2, 3:5] * B[3:5, [2]] (2x2 * 2x1) -> Schoolbook
    add_schoolbook([0, 1], [3, 4], [3, 4], [2])
    
    # 2.2.2.2: A[0:2, 3:5] * B[3:5, [3,4]] (2x2 * 2x2) -> Strassen
    add_strassen_2x2([0, 1], [3, 4], [3, 4], [3, 4])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
