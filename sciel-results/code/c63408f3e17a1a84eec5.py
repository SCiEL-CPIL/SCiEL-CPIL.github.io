import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    # U: n*m = 6 x rank
    # V: m*p = 12 x rank
    # W: p*n = 8 x rank
    
    # Rank 22 decomposition:
    # 1. Strassen (Rank 7) for A[0:2, 0:2] * B[0:2, 0:2]
    # 2. Strassen (Rank 7) for A[0:2, 0:2] * B[0:2, 2:4]
    # 3. Standard (Rank 8) for A[0:2, 2:3] * B[2:3, 0:4]
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # --- Block 1: A[0:2, 0:2] * B[0:2, 0:2] ---
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=4, B11=5
    # C indices: C00=0, C01=2, C10=1, C11=3 (W[k*n+i, r] -> W[col*n+row, rank])
    # W index for C[i,j] is j*n + i.
    # C00 -> 0*2+0=0
    # C01 -> 1*2+0=2
    # C10 -> 0*2+1=1
    # C11 -> 1*2+1=3

    # M1: (A00+A11)(B00+B11) -> C00+C11
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    # M2: (A10+A11)B00 -> C10-C11
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    # M3: A00(B01-B11) -> C01+C11
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    # M4: A11(B10-B00) -> C00+C10
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    # M5: (A00+A01)B11 -> -C00+C01
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    # M6: (A10-A00)(B00+B01) -> C11
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    # M7: (A01-A11)(B10+B11) -> C00
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # --- Block 2: A[0:2, 0:2] * B[0:2, 2:4] ---
    # B indices: B02=2, B03=3, B12=6, B13=7
    # C indices: 
    # C02 -> 2*2+0=4
    # C03 -> 3*2+0=6
    # C12 -> 2*2+1=5
    # C13 -> 3*2+1=7

    # M1': (A00+A11)(B02+B13) -> C02+C13
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    # M2': (A10+A11)B02 -> C12-C13
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    # M3': A00(B03-B13) -> C03+C13
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    # M4': A11(B12-B02) -> C02+C12
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    # M5': (A00+A01)B13 -> -C02+C03
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    # M6': (A10-A00)(B02+B03) -> C13
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    # M7': (A01-A11)(B12+B13) -> C02
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # --- Block 3: A[0:2, 2:3] * B[2:3, 0:4] ---
    # A indices: A02=2, A12=5
    # B indices: B20=8, B21=9, B22=10, B23=11
    # C indices:
    # C00=0, C01=2, C02=4, C03=6
    # C10=1, C11=3, C12=5, C13=7

    # A02 * B20 -> C00
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    # A02 * B21 -> C01
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    # A02 * B22 -> C02
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    # A02 * B23 -> C03
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    
    # A12 * B20 -> C10
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    # A12 * B21 -> C11
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    # A12 * B22 -> C12
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    # A12 * B23 -> C13
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    # Verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Build target tensor
    num_u = n * m
    num_v = m * p
    num_w = p * n
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    T_comp = np.einsum('ac,bc,cc->abc', U_np, V_np, W_np)
    
    if not np.allclose(T_comp, T, atol=1e-9):
        # Fallback to schoolbook if verification fails (should not happen)
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

    return {"U": U, "V": V, "W": W}
