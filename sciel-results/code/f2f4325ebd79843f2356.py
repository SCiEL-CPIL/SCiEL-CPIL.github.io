import numpy as np
import time

def construct():
    t0 = time.time()
    n, m, p = 2, 4, 4
    rank = 28

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_block(col_offset, X, Y, Z):
        # Strassen 7-term decomposition for 2x2x2
        # M1 = (X0+X3)(Y0+Y3) -> Z0, Z3
        U[X[0]][col_offset + 0] += 1
        U[X[3]][col_offset + 0] += 1
        V[Y[0]][col_offset + 0] += 1
        V[Y[3]][col_offset + 0] += 1
        W[Z[0]][col_offset + 0] += 1
        W[Z[3]][col_offset + 0] += 1

        # M2 = (X2+X3)Y0 -> Z2, -Z3
        U[X[2]][col_offset + 1] += 1
        U[X[3]][col_offset + 1] += 1
        V[Y[0]][col_offset + 1] += 1
        W[Z[2]][col_offset + 1] += 1
        W[Z[3]][col_offset + 1] -= 1

        # M3 = X0(Y1-Y3) -> Z1, Z3
        U[X[0]][col_offset + 2] += 1
        V[Y[1]][col_offset + 2] += 1
        V[Y[3]][col_offset + 2] -= 1
        W[Z[1]][col_offset + 2] += 1
        W[Z[3]][col_offset + 2] += 1

        # M4 = (Y2-Y0)X3 -> Z0, Z2
        U[X[3]][col_offset + 3] += 1
        V[Y[2]][col_offset + 3] += 1
        V[Y[0]][col_offset + 3] -= 1
        W[Z[0]][col_offset + 3] += 1
        W[Z[2]][col_offset + 3] += 1

        # M5 = (X0+X1)Y3 -> -Z0, Z1
        U[X[0]][col_offset + 4] += 1
        U[X[1]][col_offset + 4] += 1
        V[Y[3]][col_offset + 4] += 1
        W[Z[0]][col_offset + 4] -= 1
        W[Z[1]][col_offset + 4] += 1

        # M6 = (Y0+Y1)(X2-X0) -> Z3
        U[X[2]][col_offset + 5] += 1
        U[X[0]][col_offset + 5] -= 1
        V[Y[0]][col_offset + 5] += 1
        V[Y[1]][col_offset + 5] += 1
        W[Z[3]][col_offset + 5] += 1

        # M7 = (Y2+Y3)(X1-X3) -> Z0
        U[X[1]][col_offset + 6] += 1
        U[X[3]][col_offset + 6] -= 1
        V[Y[2]][col_offset + 6] += 1
        V[Y[3]][col_offset + 6] += 1
        W[Z[0]][col_offset + 6] += 1

    # A is 2x4, row-major: A[i,k] at index i*m+k
    # A_L (cols 0,1): indices 0,1,4,5
    # A_R (cols 2,3): indices 2,3,6,7
    # B is 4x4, row-major: B[k,j] at index k*p+j
    # B_TL (rows 0,1, cols 0,1): 0,1,4,5
    # B_TR (rows 0,1, cols 2,3): 2,3,6,7
    # B_BL (rows 2,3, cols 0,1): 8,9,12,13
    # B_BR (rows 2,3, cols 2,3): 10,11,14,15
    # C is 2x4, trace convention: C[i,j] at W[j*n+i]
    # C_L (cols 0,1): C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5 -> W indices 0,1,4,5
    # C_R (cols 2,3): C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7 -> W indices 2,3,6,7

    # For Strassen on 2x2x2, the local indices X=[x0,x1,x2,x3] map to:
    # x0=A[0,0], x1=A[0,1], x2=A[1,0], x3=A[1,1]
    # Similarly Y and Z.
    # Z[0]=C[0,0], Z[1]=C[0,1], Z[2]=C[1,0], Z[3]=C[1,1]

    # Block 1: A_L * B_TL -> C_LL (C[0,0],C[0,1],C[1,0],C[1,1])
    # A_L: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B_TL: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C_LL: C[0,0] at W[0], C[0,1] at W[1], C[1,0] at W[4], C[1,1] at W[5]
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 1, 4, 5]
    add_block(0, X1, Y1, Z1)

    # Block 2: A_L * B_TR -> C_LR (C[0,2],C[0,3],C[1,2],C[1,3])
    # B_TR: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C_LR: C[0,2] at W[2], C[0,3] at W[3], C[1,2] at W[6], C[1,3] at W[7]
    X2 = [0, 1, 4, 5]
    Y2 = [2, 3, 6, 7]
    Z2 = [2, 3, 6, 7]
    add_block(7, X2, Y2, Z2)

    # Block 3: A_R * B_BL -> C_RL (C[0,0],C[0,1],C[1,0],C[1,1])
    # A_R: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B_BL: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    X3 = [2, 3, 6, 7]
    Y3 = [8, 9, 12, 13]
    Z3 = [0, 1, 4, 5]
    add_block(14, X3, Y3, Z3)

    # Block 4: A_R * B_BR -> C_RR (C[0,2],C[0,3],C[1,2],C[1,3])
    # B_BR: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [2, 3, 6, 7]
    add_block(21, X4, Y4, Z4)

    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build expected tensor
    # T[a,b,c] = 1 iff a=i*m+k, b=k*p+j, c=j*n+i
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                val = np.sum(U_np[a_idx] * V_np[b_idx] * W_np[c_idx])
                if abs(val - 1) > 1e-10:
                    raise ValueError(f"Tensor mismatch at ({a_idx},{b_idx},{c_idx}): {val}")

    return {"U": U, "V": V, "W": W}
