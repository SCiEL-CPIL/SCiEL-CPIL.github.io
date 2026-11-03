import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strassen decomposition for 2x2x2 matrix multiplication
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6

    # U coefficients: for each Strassen term s, dict of (i_loc, j_loc) -> coeff
    U_strassen = [
        {(0, 0): 1, (1, 1): 1},       # P1: X00 + X11
        {(1, 0): 1, (1, 1): 1},       # P2: X10 + X11
        {(0, 0): 1},                   # P3: X00
        {(1, 1): 1},                   # P4: X11
        {(0, 0): 1, (0, 1): 1},       # P5: X00 + X01
        {(1, 0): 1, (0, 0): -1},      # P6: X10 - X00
        {(0, 1): 1, (1, 1): -1}       # P7: X01 - X11
    ]

    # V coefficients: for each Strassen term s, dict of (j_loc, k_loc) -> coeff
    V_strassen = [
        {(0, 0): 1, (1, 1): 1},       # P1: Y00 + Y11
        {(0, 0): 1},                   # P2: Y00
        {(0, 1): 1, (1, 1): -1},      # P3: Y01 - Y11
        {(1, 0): 1, (0, 0): -1},      # P4: Y10 - Y00
        {(1, 1): 1},                   # P5: Y11
        {(0, 0): 1, (0, 1): 1},       # P6: Y00 + Y01
        {(1, 0): 1, (1, 1): 1}        # P7: Y10 + Y11
    ]

    # W contributions: for each Strassen term s, dict of (i_loc, k_loc) -> coeff
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    W_strassen = [
        {(0, 0): 1, (1, 1): 1},       # P1: Z00 + Z11
        {(1, 0): 1, (1, 1): -1},      # P2: Z10 - Z11
        {(0, 1): 1, (1, 1): 1},       # P3: Z01 + Z11
        {(0, 0): 1, (1, 0): 1},       # P4: Z00 + Z10
        {(0, 0): -1, (0, 1): 1},      # P5: -Z00 + Z01
        {(1, 1): 1},                   # P6: Z11
        {(0, 0): 1}                    # P7: Z00
    ]

    # Build the full decomposition
    # A is 2x4: split into A_L (cols 0-1) and A_R (cols 2-3)
    # B is 4x6: split into 3 blocks of 4x2 (B_1, B_2, B_3)
    # Each B_j: top (rows 0-1) and bottom (rows 2-3), each 2x2
    # C_{i, b*2+k_loc} = (A_L * B_{b,top})_{i,k_loc} + (A_R * B_{b,bot})_{i,k_loc}
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    col = 0
    for a in range(2):  # A block: 0 for cols 0-1, 1 for cols 2-3
        for b in range(3):  # B column block: 0, 1, 2
            for s in range(7):  # Strassen term
                # Update U: A indices are (i, j) where j = a*2 + j_loc
                for (i_loc, j_loc), coeff in U_strassen[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j  # i*4 + j
                    U[u_idx][col] += coeff

                # Update V: B indices are (j, k) where j depends on a, k = b*2 + k_loc
                # For a=0 (A_L), we use B_{b,top} (rows 0-1), so j = j_loc
                # For a=1 (A_R), we use B_{b,bot} (rows 2-3), so j = 2 + j_loc
                for (j_loc, k_loc), coeff in V_strassen[s].items():
                    if a == 0:
                        j = j_loc  # top rows 0-1
                    else:
                        j = 2 + j_loc  # bottom rows 2-3
                    k = b * 2 + k_loc
                    v_idx = j * p + k  # j*6 + k
                    V[v_idx][col] += coeff

                # Update W: C indices are (k, i) -> W[k*n + i] = W[k*2 + i]
                for (i_loc, k_loc), coeff in W_strassen[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i  # k*2 + i
                    W[w_idx][col] += coeff

                col += 1

    # Verify the decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    # Let's verify with random matrices
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(42)
        A_test = rng.integers(-5, 5, size=(n, m))
        B_test = rng.integers(-5, 5, size=(m, p))
        C_expected = A_test @ B_test
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute C from decomposition
        C_decomp = np.zeros((n, p))
        for s in range(rank):
            u_col = U_arr[:, s]  # n*m = 8
            v_col = V_arr[:, s]  # m*p = 24
            w_col = W_arr[:, s]  # p*n = 12
            
            # u_col reshaped to (n, m)
            u_mat = u_col.reshape(n, m)
            # v_col reshaped to (m, p)
            v_mat = v_col.reshape(m, p)
            # w_col reshaped to (p, n)
            w_mat = w_col.reshape(p, n)
            
            # Contribution: sum_{i,j,k} u_mat[i,j] * v_mat[j,k] * w_mat[k,i] * E_{i,k}
            # This is: for each i,k, add sum_j u_mat[i,j]*v_mat[j,k]*w_mat[k,i]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += np.sum(u_mat[i, :] * v_mat[:, k] * w_mat[k, i])
        
        if not np.allclose(C_decomp, C_expected, atol=1e-9):
            # Fallback to schoolbook
            schoolbook_rank = n * m * p
            U = [[0] * schoolbook_rank for _ in range(n * m)]
            V = [[0] * schoolbook_rank for _ in range(m * p)]
            W = [[0] * schoolbook_rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
