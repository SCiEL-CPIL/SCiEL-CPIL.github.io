import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    # We return a known rank 22 decomposition.
    # Structure:
    # 1. Strassen decomposition for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Rank 7)
    # 2. Strassen decomposition for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Rank 7)
    # 3. Outer products for A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4] (Rank 8)
    # Total Rank: 7 + 7 + 8 = 22
    
    rank = 22
    # U shape: (n*m, rank) = (6, 22)
    # V shape: (m*p, rank) = (12, 22)
    # W shape: (p*n, rank) = (8, 22)
    
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

    # Indices mapping:
    # A (2x3): 
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2
    #   A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B (3x4):
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    #   B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    #   B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # C (2x4) output W[k*n+i, r] where C[i,k] = sum_j A[i,j]B[j,k]
    #   W[0*2+0]=W[0] -> C[0,0]
    #   W[1*2+0]=W[2] -> C[0,1]
    #   W[2*2+0]=W[4] -> C[0,2]
    #   W[3*2+0]=W[6] -> C[0,3]
    #   W[0*2+1]=W[1] -> C[1,0]
    #   W[1*2+1]=W[3] -> C[1,1]
    #   W[2*2+1]=W[5] -> C[1,2]
    #   W[3*2+1]=W[7] -> C[1,3]

    # --- Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] ---
    # Uses indices:
    # A: 0,1,3,4
    # B: 0,1,4,5
    # C: 0,2,1,3 (W indices)
    
    # M1: (A00+A11)(B00+B11) -> C00+C11
    # A00=0, A11=4, B00=0, B11=5, C00=W[0], C11=W[3]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    
    # M2: (A10+A11)B00 -> C10-C11
    # A10=3, A11=4, B00=0, C10=W[1], C11=W[3]
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    
    # M3: A00(B01-B11) -> C01+C11
    # A00=0, B01=1, B11=5, C01=W[2], C11=W[3]
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    
    # M4: A11(B10-B00) -> C00+C10
    # A11=4, B10=4, B00=0, C00=W[0], C10=W[1]
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    
    # M5: (A00+A01)B11 -> -C00+C01
    # A00=0, A01=1, B11=5, C00=W[0], C01=W[2]
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    
    # M6: (A10-A00)(B00+B01) -> C11
    # A10=3, A00=0, B00=0, B01=1, C11=W[3]
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    
    # M7: (A01-A11)(B10+B11) -> C00
    # A01=1, A11=4, B10=4, B11=5, C00=W[0]
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # --- Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] ---
    # Uses indices:
    # A: 0,1,3,4
    # B: 2,3,6,7
    # C: 4,6,5,7 (W indices for C[0,2], C[0,3], C[1,2], C[1,3])
    
    # M1': (A00+A11)(B02+B13) -> C02+C13
    # A00=0, A11=4, B02=2, B13=7, C02=W[4], C13=W[7]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    
    # M2': (A10+A11)B02 -> C12-C13
    # A10=3, A11=4, B02=2, C12=W[5], C13=W[7]
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    
    # M3': A00(B03-B13) -> C03+C13
    # A00=0, B03=3, B13=7, C03=W[6], C13=W[7]
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    
    # M4': A11(B12-B02) -> C02+C12
    # A11=4, B12=6, B02=2, C02=W[4], C12=W[5]
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    
    # M5': (A00+A01)B13 -> -C02+C03
    # A00=0, A01=1, B13=7, C02=W[4], C03=W[6]
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    
    # M6': (A10-A00)(B02+B03) -> C13
    # A10=3, A00=0, B02=2, B03=3, C13=W[7]
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    
    # M7': (A01-A11)(B12+B13) -> C02
    # A01=1, A11=4, B12=6, B13=7, C02=W[4]
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # --- Block 3: Outer Products for A[0:2, 2:3] * B[2:3, 0:4] ---
    # A indices: 2 (A[0,2]), 5 (A[1,2])
    # B indices: 8,9,10,11 (B[2,0], B[2,1], B[2,2], B[2,3])
    # C indices: 
    #   Row 0: W[0] (C00), W[2] (C01), W[4] (C02), W[6] (C03)
    #   Row 1: W[1] (C10), W[3] (C11), W[5] (C12), W[7] (C13)
    
    # A[0,2] * B[2,0] -> C[0,0] (W[0])
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    # A[0,2] * B[2,1] -> C[0,1] (W[2])
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    # A[0,2] * B[2,2] -> C[0,2] (W[4])
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    # A[0,2] * B[2,3] -> C[0,3] (W[6])
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    
    # A[1,2] * B[2,0] -> C[1,0] (W[1])
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    # A[1,2] * B[2,1] -> C[1,1] (W[3])
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    # A[1,2] * B[2,2] -> C[1,2] (W[5])
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    # A[1,2] * B[2,3] -> C[1,3] (W[7])
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    # Verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Construct the target tensor T
    # T[c_idx, a_idx, b_idx] = 1 if c corresponds to a*b term
    # c_idx = k*n + i
    # a_idx = i*m + j
    # b_idx = j*p + k
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0

    # Reconstruct T from decomposition
    # T_recon[c, a, b] = sum_r U[a, r] V[b, r] W[c, r]
    T_recon = np.einsum('ar,br,cr->cab', U_np, V_np, W_np)
    
    if not np.allclose(T_recon, T, atol=1e-8):
        raise ValueError("Decomposition verification failed")

    # Check for integer/half-integer entries
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if val is integer or half-integer
                # val * 2 should be integer
                if val * 2 != int(val * 2):
                    raise ValueError("Non-half-integer entry found: " + str(val))

    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        raise TimeoutError("Exceeded time limit")

    return {"U": U, "V": V, "W": W}
