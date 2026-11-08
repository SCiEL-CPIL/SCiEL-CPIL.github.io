import time

_DEADLINE = 115.0
_START = None


def construct():
    global _START
    if _START is None:
        _START = time.time()

    n, m, p = 2, 3, 9
    # Target rank 48: use 3 blocks of 2x2x3 (rank 10 each) + 2x1x9 (rank 18)
    # 3*10 + 18 = 48

    RANK = 48
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]

    col = 0

    # 2x2x3 block: computes C_{i,k} = A_{i,0}*B_{0,k} + A_{i,1}*B_{1,k} for i in {0,1}, k in {k0, k0+1, k0+2}
    # Rank 10 construction
    def block_2x2x3(k0, col_start):
        # U indices: 0=A_{0,0}, 1=A_{1,0}, 2=A_{0,1}, 3=A_{1,1}
        # V indices: 4=B_{0,k0}, 5=B_{0,k0+1}, 6=B_{0,k0+2}, 7=B_{1,k0}, 8=B_{1,k0+1}, 9=B_{1,k0+2}
        # W indices: 10=C_{0,k0}, 11=C_{0,k0+1}, 12=C_{0,k0+2}, 13=C_{1,k0}, 14=C_{1,k0+1}, 15=C_{1,k0+2}
        
        u_a0 = 0  # A_{0,0}
        u_a1 = 2  # A_{0,1}
        u_b0 = 1  # A_{1,0}
        u_b1 = 3  # A_{1,1}
        
        v_b0k0 = 4   # B_{0,k0}
        v_b0k1 = 5   # B_{0,k0+1}
        v_b0k2 = 6   # B_{0,k0+2}
        v_b1k0 = 7   # B_{1,k0}
        v_b1k1 = 8   # B_{1,k0+1}
        v_b1k2 = 9   # B_{1,k0+2}
        
        # We need to compute:
        # C_{0,k0} = A_{0,0}*B_{0,k0} + A_{0,1}*B_{1,k0}
        # C_{0,k0+1} = A_{0,0}*B_{0,k0+1} + A_{0,1}*B_{1,k0+1}
        # C_{0,k0+2} = A_{0,0}*B_{0,k0+2} + A_{0,1}*B_{1,k0+2}
        # C_{1,k0} = A_{1,0}*B_{0,k0} + A_{1,1}*B_{1,k0}
        # C_{1,k0+1} = A_{1,0}*B_{0,k0+1} + A_{1,1}*B_{1,k0+1}
        # C_{1,k0+2} = A_{1,0}*B_{0,k0+2} + A_{1,1}*B_{1,k0+2}
        
        # Use 10 rank-1 terms:
        # M1 = (A_{0,0} + A_{1,0} + A_{0,1} + A_{1,1}) * (B_{0,k0} + B_{1,k0} + B_{0,k0+1} + B_{1,k0+1} + B_{0,k0+2} + B_{1,k0+2})
        # This is getting complex. Let me use a simpler approach.
        
        # Direct approach with some sharing:
        # M1 = A_{0,0} * (B_{0,k0} + B_{0,k0+1} + B_{0,k0+2})
        # M2 = A_{0,1} * (B_{1,k0} + B_{1,k0+1} + B_{1,k0+2})
        # M3 = A_{1,0} * (B_{0,k0} + B_{0,k0+1} + B_{0,k0+2})
        # M4 = A_{1,1} * (B_{1,k0} + B_{1,k0+1} + B_{1,k0+2})
        # That's only 4 terms but doesn't give correct results.
        
        # Actually, the direct rank for 2x2x3 is 12 (each of 6 outputs needs 2 terms, but they share).
        # Let me just use rank 12 for safety, giving total 3*12 + 18 = 54. That's worse.
        
        # Let me use Strassen-like: split into 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
        # Total: 3*11 + 18 = 51. Still worse than 50.
        
        # Hmm, let me just keep the rank 50 solution but verify it's correct.
        pass

    # Let me just output the verified rank-50 solution
    U = [[0] * 50 for _ in range(n * m)]
    V = [[0] * 50 for _ in range(m * p)]
    W = [[0] * 50 for _ in range(p * n)]

    col = 0

    # Strassen block for 2x2x2
    def strassen_block(k0, k1, col_start):
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1

        u_a = 0
        u_b = 1
        u_c = 3
        u_d = 4

        # M1 = (a+d)(e+h)
        U[u_a][col_start] += 1
        U[u_d][col_start] += 1
        V[v_e][col_start] += 1
        V[v_h][col_start] += 1

        # M2 = (c+d)e
        U[u_c][col_start+1] += 1
        U[u_d][col_start+1] += 1
        V[v_e][col_start+1] += 1

        # M3 = a(f-h)
        U[u_a][col_start+2] += 1
        V[v_f][col_start+2] += 1
        V[v_h][col_start+2] -= 1

        # M4 = d(g-e)
        U[u_d][col_start+3] += 1
        V[v_g][col_start+3] += 1
        V[v_e][col_start+3] -= 1

        # M5 = (a+b)h
        U[u_a][col_start+4] += 1
        U[u_b][col_start+4] += 1
        V[v_h][col_start+4] += 1

        # M6 = (c-a)(e+f)
        U[u_c][col_start+5] += 1
        U[u_a][col_start+5] -= 1
        V[v_e][col_start+5] += 1
        V[v_f][col_start+5] += 1

        # M7 = (b-d)(g+h)
        U[u_b][col_start+6] += 1
        U[u_d][col_start+6] -= 1
        V[v_g][col_start+6] += 1
        V[v_h][col_start+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # C00 = M1 + M4 - M5 + M7
        W[w_00][col_start] += 1
        W[w_00][col_start+3] += 1
        W[w_00][col_start+4] -= 1
        W[w_00][col_start+6] += 1

        # C01 = M3 + M5
        W[w_01][col_start+2] += 1
        W[w_01][col_start+4] += 1

        # C10 = M2 + M4
        W[w_10][col_start+1] += 1
        W[w_10][col_start+3] += 1

        # C11 = M1 - M2 + M3 + M6
        W[w_11][col_start] += 1
        W[w_11][col_start+1] -= 1
        W[w_11][col_start+2] += 1
        W[w_11][col_start+5] += 1

    # 4 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        strassen_block(k0, k1, col)
        col += 7

    # 2x2x1 block for column 8
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1

    col += 4

    # 2x1x9 block
    for k in range(9):
        v_b = 2 * p + k
        U[2][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1

        U[5][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1

        col += 2

    return {"U": U, "V": V, "W": W}
