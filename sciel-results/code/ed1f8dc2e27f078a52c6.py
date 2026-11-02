"""Matrix multiplication 2x3x6 tensor decomposition; Strassen block decomposition."""
DIMENSIONS = [2, 3, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = 2, 3, 6

    # Block partition:
    # A = [A1 | A2], A1 is 2x1 (col 0), A2 is 2x2 (cols 1,2)
    # B = [B1; B2], B1 is 1x6 (row 0), B2 is 2x6 (rows 1,2)
    # C = A1*B1 + A2*B2
    # A1*B1: rank 1 (outer product of column vector and row vector)
    # A2*B2: partition B2's 6 cols into 3 pairs, each 2x2x2 via Strassen (rank 7)
    # Total rank: 1 + 3*7 = 22

    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 18 x 22
    W = [[0] * rank for _ in range(p * n)]  # 12 x 22

    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        """u_coeffs: dict {index: coeff}, v_coeffs, w_coeffs"""
        for idx, c in u_coeffs.items():
            U[idx][col] += c
        for idx, c in v_coeffs.items():
            V[idx][col] += c
        for idx, c in w_coeffs.items():
            W[idx][col] += c

    col = 0

    # Term 1: A1*B1 (rank 1)
    # A1[i,0] * B1[0,k] for i in {0,1}, k in {0..5}
    # U: A[i,0] for i=0,1 -> indices 0, 1
    # V: B[0,k] for k=0..5 -> indices 0..5
    # W: C[i,k] -> W[k*n+i] = W[k*2+i] for i=0,1, k=0..5
    # This is a single rank-1 term? No! It's 12 separate products.
    # Wait: A1[i,0]*B1[0,k] - the tensor entry T[i,0,k] = 1.
    # This factors as (sum_i A[i,0]) * (sum_k B[0,k])? No.
    # T[i,0,k] = 1 for all i,k. This is u_i * v_0 * w_k with u_i=1, v_0=1, w_k=1.
    # So it IS rank 1: U[0]=1, U[1]=1, V[0]=1, W[k*2+0]=1, W[k*2+1]=1 for all k.
    # Wait no. The rank-1 term means: U[i*m+j] * V[j*p+k] * W[k*n+i] = T[i,j,k]
    # For j=0: U[i*3+0] * V[0*6+k] * W[k*2+i] should equal T[i,0,k] = 1
    # So U[i*3] * V[k] * W[k*2+i] = 1 for all i,k
    # Set U[0]=1, U[3]=1, V[k]=1 for all k, W[k*2+i]=1 for all i,k
    # Check: U[i*3] * V[k] * W[k*2+i] = 1 * 1 * 1 = 1. Yes!
    u1 = {0: 1, 3: 1}  # A[0,0], A[1,0]
    v1 = {k: 1 for k in range(6)}  # B[0,0]..B[0,5]
    w1 = {k * 2 + i: 1 for k in range(6) for i in range(2)}  # C[i,k]
    add_term(u1, v1, w1, col)
    col += 1

    # A2*B2: 3 Strassen blocks
    # B2 rows: j=1,2; cols partitioned into {0,1}, {2,3}, {4,5}
    for block in range(3):
        k0 = block * 2
        k1 = block * 2 + 1
        # Local: A2[i, j_local] where j_local=0->j=1, j_local=1->j=2
        #         B2[j_local, k_local] where k_local=0->k0, k_local=1->k1
        # A indices: i*3 + (1+j_local)
        # B indices: (1+j_local)*6 + (k0 + k_local)
        # C indices: (k0+k_local)*2 + i

        # Strassen M1-M7
        # M1 = (A00+A11)(B00+B11)
        u = {0*3+1: 1, 1*3+2: 1}  # A[0,1]+A[1,2]
        v = {1*6+k0: 1, 2*6+k1: 1}  # B[1,k0]+B[2,k1]
        w = {(k0+0)*2+0: 1, (k0+1)*2+1: 1}  # C[0,k0]+C[1,k1]
        add_term(u, v, w, col); col += 1

        # M2 = (A10+A11)B00
        u = {1*3+1: 1, 1*3+2: 1}  # A[1,1]+A[1,2]
        v = {1*6+k0: 1}  # B[1,k0]
        w = {(k0+0)*2+1: 1}  # C[1,k0]
        add_term(u, v, w, col); col += 1

        # M3 = A00(B01-B11)
        u = {0*3+1: 1}  # A[0,1]
        v = {1*6+k1: 1, 2*6+k1: -1}  # B[1,k1]-B[2,k1]
        w = {(k0+1)*2+0: 1}  # C[0,k1]
        add_term(u, v, w, col); col += 1

        # M4 = A11(B10-B00)
        u = {1*3+2: 1}  # A[1,2]
        v = {2*6+k0: 1, 1*6+k0: -1}  # B[2,k0]-B[1,k0]
        w = {(k0+0)*2+1: 1, (k0+1)*2+0: 1}  # C[1,k0]+C[0,k1]
        add_term(u, v, w, col); col += 1

        # M5 = (A00+A01)B11
        u = {0*3+1: 1, 0*3+2: 1}  # A[0,1]+A[0,2]
        v = {2*6+k1: 1}  # B[2,k1]
        w = {(k0+0)*2+0: -1, (k0+1)*2+0: 1}  # -C[0,k0]+C[0,k1]
        add_term(u, v, w, col); col += 1

        # M6 = (A10-A00)(B00+B10)
        u = {1*3+1: 1, 0*3+1: -1}  # A[1,1]-A[0,1]
        v = {1*6+k0: 1, 2*6+k0: 1}  # B[1,k0]+B[2,k0]
        w = {(k0+1)*2+1: 1}  # C[1,k1]
        add_term(u, v, w, col); col += 1

        # M7 = (A01-A11)(B10+B11)
        u = {0*3+2: 1, 1*3+2: -1}  # A[0,2]-A[1,2]
        v = {2*6+k0: 1, 2*6+k1: 1}  # B[2,k0]+B[2,k1]
        w = {(k0+0)*2+0: 1}  # C[0,k0]
        add_term(u, v, w, col); col += 1

        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
