import time

DIMENSIONS = [2, 5, 5]

def construct():
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 140.0
    
    n, m, p = DIMENSIONS
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_block(U_coeffs, V_coeffs, W_coeffs):
        """
        U_coeffs: list of (row_idx, coeff)
        V_coeffs: list of (row_idx, coeff)
        W_coeffs: list of (row_idx, coeff)
        """
        nonlocal rank
        U_row = [0] * rank
        V_row = [0] * rank
        W_row = [0] * rank
        for idx, coeff in U_coeffs:
            U_row[rank] = coeff
        for idx, coeff in V_coeffs:
            V_row[rank] = coeff
        for idx, coeff in W_coeffs:
            W_row[rank] = coeff
        U_rows.append(U_row)
        V_rows.append(V_row)
        W_rows.append(W_row)
        rank += 1

    # Helper to add a standard 1x1x1 term
    def add_term(i, j, k, coeff=1):
        add_block([(i*m+j, coeff)], [(j*p+k, coeff)], [(k*n+i, coeff)])

    # 1) First 2x2x2 block: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    add_block([(0*m+0, 1), (1*m+1, 1)], [(0*p+0, 1), (1*p+1, 1)], [(0*n+0, 1), (0*n+1, 1), (1*n+0, 1), (1*n+1, 1)])
    # M2 = (A10 + A11)(B01)
    add_block([(1*m+0, 1), (1*m+1, 1)], [(0*p+1, 1)], [(1*n+1, 1)])
    # M3 = (A00)(B10 + B11)
    add_block([(0*m+0, 1)], [(1*p+0, 1), (1*p+1, 1)], [(0*n+0, 1), (0*n+1, 1)])
    # M4 = (A10)(B00)
    add_block([(1*m+0, 1)], [(0*p+0, 1)], [(1*n+0, 1)])
    # M5 = (A00 + A10)(B11 - B01)
    add_block([(0*m+0, 1), (1*m+0, 1)], [(1*p+1, 1), (0*p+1, -1)], [(1*n+0, 1), (0*n+1, 1)])
    # M6 = (A11 - A00)(B00 + B01)
    add_block([(1*m+1, 1), (0*m+0, -1)], [(0*p+0, 1), (0*p+1, 1)], [(0*n+0, 1), (1*n+1, 1)])
    # M7 = (A01 - A11)(B10 + B11)
    add_block([(0*m+1, 1), (1*m+1, -1)], [(1*p+0, 1), (1*p+1, 1)], [(0*n+1, 1), (1*n+0, 1)])

    # 2) Next 2x2x2 block: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # j in 2,3; k in 0,1
    # Same structure, just different indices
    add_block([(0*m+2, 1), (1*m+3, 1)], [(2*p+0, 1), (3*p+1, 1)], [(0*n+0, 1), (0*n+1, 1), (1*n+0, 1), (1*n+1, 1)])
    add_block([(1*m+2, 1), (1*m+3, 1)], [(2*p+1, 1)], [(1*n+1, 1)])
    add_block([(0*m+2, 1)], [(3*p+0, 1), (3*p+1, 1)], [(0*n+0, 1), (0*n+1, 1)])
    add_block([(1*m+2, 1)], [(2*p+0, 1)], [(1*n+0, 1)])
    add_block([(0*m+2, 1), (1*m+2, 1)], [(3*p+1, 1), (2*p+1, -1)], [(1*n+0, 1), (0*n+1, 1)])
    add_block([(1*m+3, 1), (0*m+2, -1)], [(2*p+0, 1), (2*p+1, 1)], [(0*n+0, 1), (1*n+1, 1)])
    add_block([(0*m+3, 1), (1*m+3, -1)], [(3*p+0, 1), (3*p+1, 1)], [(0*n+1, 1), (1*n+0, 1)])

    # 3) Block: A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2]
    # 2x1x2 block. Rank 4.
    # C00 = A04 B40, C01 = A04 B41, C10 = A14 B40, C11 = A14 B41
    add_term(0, 4, 0)
    add_term(0, 4, 1)
    add_term(1, 4, 0)
    add_term(1, 4, 1)

    # 4) Block: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # j in 0,1; k in 2,3
    add_block([(0*m+0, 1), (1*m+1, 1)], [(0*p+2, 1), (1*p+3, 1)], [(2*n+0, 1), (3*n+0, 1), (2*n+1, 1), (3*n+1, 1)])
    add_block([(1*m+0, 1), (1*m+1, 1)], [(0*p+3, 1)], [(3*n+1, 1)])
    add_block([(0*m+0, 1)], [(1*p+2, 1), (1*p+3, 1)], [(2*n+0, 1), (3*n+0, 1)])
    add_block([(1*m+0, 1)], [(0*p+2, 1)], [(2*n+1, 1)])
    add_block([(0*m+0, 1), (1*m+0, 1)], [(1*p+3, 1), (0*p+3, -1)], [(2*n+1, 1), (3*n+0, 1)])
    add_block([(1*m+1, 1), (0*m+0, -1)], [(0*p+2, 1), (0*p+3, 1)], [(2*n+0, 1), (3*n+1, 1)])
    add_block([(0*m+1, 1), (1*m+1, -1)], [(1*p+2, 1), (1*p+3, 1)], [(3*n+0, 1), (2*n+1, 1)])

    # 5) Block: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # j in 2,3; k in 2,3
    add_block([(0*m+2, 1), (1*m+3, 1)], [(2*p+2, 1), (3*p+3, 1)], [(2*n+0, 1), (3*n+0, 1), (2*n+1, 1), (3*n+1, 1)])
    add_block([(1*m+2, 1), (1*m+3, 1)], [(2*p+3, 1)], [(3*n+1, 1)])
    add_block([(0*m+2, 1)], [(3*p+2, 1), (3*p+3, 1)], [(2*n+0, 1), (3*n+0, 1)])
    add_block([(1*m+2, 1)], [(2*p+2, 1)], [(2*n+1, 1)])
    add_block([(0*m+2, 1), (1*m+2, 1)], [(3*p+3, 1), (2*p+3, -1)], [(2*n+1, 1), (3*n+0, 1)])
    add_block([(1*m+3, 1), (0*m+2, -1)], [(2*p+2, 1), (2*p+3, 1)], [(2*n+0, 1), (3*n+1, 1)])
    add_block([(0*m+3, 1), (1*m+3, -1)], [(3*p+2, 1), (3*p+3, 1)], [(3*n+0, 1), (2*n+1, 1)])

    # 6) Block: A[0:2, 4:5] * B[4:5, 2:4] -> C[0:2, 2:4]
    # 2x1x2 block. Rank 4.
    add_term(0, 4, 2)
    add_term(0, 4, 3)
    add_term(1, 4, 2)
    add_term(1, 4, 3)

    # 7) Block: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # j in 0,1; k in 4
    # 2x2x1 block. Rank 4.
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    add_term(0, 0, 4)
    add_term(0, 1, 4)
    add_term(1, 0, 4)
    add_term(1, 1, 4)

    # 8) Block: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    # j in 2,3; k in 4
    # 2x2x1 block. Rank 4.
    add_term(0, 2, 4)
    add_term(0, 3, 4)
    add_term(1, 2, 4)
    add_term(1, 3, 4)

    # 9) Block: A[0:2, 4:5] * B[4:5, 4:5] -> C[0:2, 4:5]
    # 2x1x1 block. Rank 2.
    add_term(0, 4, 4)
    add_term(1, 4, 4)

    # Total rank: 7 + 7 + 4 + 7 + 7 + 4 + 4 + 4 + 2 = 46
    # Wait, let's verify the count.
    # 1: 7
    # 2: 7
    # 3: 4
    # 4: 7
    # 5: 7
    # 6: 4
    # 7: 4
    # 8: 4
    # 9: 2
    # Sum: 7+7+4+7+7+4+4+4+2 = 46.
    
    # Pad U, V, W to correct dimensions
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            if U_rows[r][i] != 0:
                U[i][r] = U_rows[r][i]
        for j in range(m * p):
            if V_rows[r][j] != 0:
                V[j][r] = V_rows[r][j]
        for k in range(p * n):
            if W_rows[r][k] != 0:
                W[k][r] = W_rows[r][k]
                
    return {"U": U, "V": V, "W": W}
