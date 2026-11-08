import numpy as np

def construct():
    n, m, p = 2, 3, 9
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    # Blocks 0-6: Strassen 7-term per 2x3 sub-block (3*7=21)
    for b in range(3):
        k0 = 3 * b
        k1 = 3 * b + 1
        k2 = 3 * b + 2
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1

        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        W[w_10][col+1] += 1

        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        W[w_01][col+2] += 1

        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        W[w_00][col+3] += 1
        W[w_10][col+3] += 1

        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        W[w_00][col+4] -= 1
        W[w_01][col+4] += 1

        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        W[w_11][col+5] += 1

        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        W[w_00][col+6] += 1
        W[w_11][col+1] -= 1

        col += 7

    # Cross terms for the 3rd column of A (col=2) with rows 0,1 of B
    # A[:,2] * B[0,:] and A[:,2] * B[1,:]
    # C[i, j] = A[i,2] * B[0,j] + A[i,2] * B[1,j] = A[i,2] * (B[0,j] + B[1,j])
    u_c2 = 2  # A[0,2]
    u_c3 = 5  # A[1,2]
    for j in range(9):
        v_j0 = 0 * p + j
        v_j1 = 1 * p + j
        w_j0 = j * n + 0
        w_j1 = j * n + 1

        # C[0,j] = A[0,2]*(B[0,j]+B[1,j])
        U[u_c2][col] += 1
        V[v_j0][col] += 1
        V[v_j1][col] += 1
        W[w_j0][col] += 1

        # C[1,j] = A[1,2]*(B[0,j]+B[1,j])
        U[u_c3][col+1] += 1
        V[v_j0][col+1] += 1
        V[v_j1][col+1] += 1
        W[w_j1][col+1] += 1
        col += 2

    # Cross terms for A[:,0:2] with B[2,:]
    # C[i,j] = A[i,0]*B[2,j] + A[i,1]*B[2,j] = (A[i,0]+A[i,1]) * B[2,j]
    u_a0 = 0  # A[0,0]
    u_a1 = 1  # A[0,1]
    u_b0 = 3  # A[1,0]
    u_b1 = 4  # A[1,1]
    for j in range(9):
        v_2j = 2 * p + j
        w_j0 = j * n + 0
        w_j1 = j * n + 1

        # C[0,j] = (A[0,0]+A[0,1])*B[2,j]
        U[u_a0][col] += 1
        U[u_a1][col] += 1
        V[v_2j][col] += 1
        W[w_j0][col] += 1

        # C[1,j] = (A[1,0]+A[1,1])*B[2,j]
        U[u_b0][col+1] += 1
        U[u_b1][col+1] += 1
        V[v_2j][col+1] += 1
        W[w_j1][col+1] += 1
        col += 2

    return {"U": U, "V": V, "W": W}
