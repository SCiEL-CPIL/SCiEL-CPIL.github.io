import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = 2, 4, 5

    # Build rank-32 decomposition:
    # C (2x5) = A (2x4) * B (4x5)
    # Split B columns: cols 0-1, cols 2-3, col 4
    # C[:,0:2] = A * B[:,0:2]  -> 2x4x2, rank 12
    # C[:,2:4] = A * B[:,2:4]  -> 2x4x2, rank 12  
    # C[:,4:5]  = A * B[:,4:5]  -> 2x4x1, rank 4
    # Total: 12 + 12 + 4 = 28? But R(2,4,2)=12 requires specific construction.
    #
    # Let me use the known 12-term decomposition for 2x4x2:
    # The 2x4x2 tensor T[i,j,k] = delta_{i,k} for the "identity" part isn't right.
    # Actually T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor.
    #
    # For 2x4x2: T has shape (2,4,2). Entry T[i,j,k] = 1.
    # We need: C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # 12-term construction for 2x4x2:
    # Split into two 2x2x2 blocks (rows 0-1 and rows 2-3 of middle dim)
    # Each 2x2x2 has rank 7 (Strassen), total 14.
    # Save 2 terms by sharing: the terms corresponding to A[i,0]*B[0,k] + A[i,2]*B[2,k]
    # and A[i,1]*B[1,k] + A[i,3]*B[3,k] can be partially shared.
    #
    # Known 12-term scheme for 2x4x2:
    # Let a0=A[0,0], a1=A[0,1], a2=A[0,2], a3=A[0,3]
    #     b0=B[0,0], b1=B[1,0], b2=B[2,0], b3=B[3,0]  (for k=0)
    #     b4=B[0,1], b5=B[1,1], b6=B[2,1], b7=B[3,1]  (for k=1)
    #
    # Actually let me just use 14+14+4=32 (two Strassen + one standard)
    # and verify it works. That's better than 36.

    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 20 rows
    W = [[0] * rank for _ in range(p * n)]  # 10 rows

    col = 0

    # Strassen 7 terms for 2x2x2 (indices 0-1 in each dim)
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    # 4 terms for 2x2x1 (standard)
    standard_2x2x1 = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({0:1}, {0:1}, {1:1}),
        ({1:1}, {1:1}, {1:1}),
    ]

    # Block 1: cols 0-1 of B, using rows 0-3 of A
    # Split into rows 0-1 and rows 2-3, each 2x2x2
    for row_offset in [0, 2]:
        for u_t, v_t, w_t in strassen:
            for idx_u, val_u in u_t.items():
                li = idx_u // 2
                lj = idx_u % 2
                U[li * 4 + (row_offset + lj)][col] += val_u
            for idx_v, val_v in v_t.items():
                lj = idx_v // 2
                lk = idx_v % 2
                V[(row_offset + lj) * 5 + lk][col] += val_v
            for idx_w, val_w in w_t.items():
                li = idx_w // 2
                lk = idx_w % 2
                W[lk * 2 + li][col] += val_w
            col += 1

    # Block 2: cols 2-3 of B, same structure
    for row_offset in [0, 2]:
        for u_t, v_t, w_t in strassen:
            for idx_u, val_u in u_t.items():
                li = idx_u // 2
                lj = idx_u % 2
                U[li * 4 + (row_offset + lj)][col] += val_u
            for idx_v, val_v in v_t.items():
                lj = idx_v // 2
                lk = idx_v % 2
                V[(row_offset + lj) * 5 + (2 + lk)][col] += val_v
            for idx_w, val_w in w_t.items():
                li = idx_w // 2
                lk = idx_w % 2
                W[(2 + lk) * 2 + li][col] += val_w
            col += 1

    # Block 3: col 4 of B, 2x4x1, standard 4 terms
    for i in range(2):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 2 + i][col] = 1
            col += 1

    # Verify
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: rank 40 schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][c] = 1
                    V_fb[j * p + k][c] = 1
                    W_fb[k * n + i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
