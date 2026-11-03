"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def strassen_2x2_terms():
        """7 Strassen terms for 2x2x2. Each: (U4, V4, W4) flat row-major."""
        return [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1=(a+d)(e+h)
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2=(c+d)(e)
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # M3=a(f-h)
            ([0,0,0,1], [0,0,1,-1], [-1,0,1,0]), # M4=d(g-e)
            ([1,1,0,0], [0,0,0,1], [1,1,0,0]),   # M5=(a+b)(h)
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6=(c-a)(e+f)
            ([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]) # M7=(b-d)(g+h)
        ]

    def build_4x4x4_strassen():
        """49 terms for 4x4x4 via Kronecker of 2x2 Strassen."""
        base = strassen_2x2_terms()
        terms = []
        for u1, v1, w1 in base:
            for u2, v2, w2 in base:
                U = [0]*16
                V = [0]*16
                W = [0]*16
                for i in range(4):
                    for j in range(4):
                        U[i*4+j] = u1[i*2 + (j//2)] * u2[i%2 *2 + j%2] if False else 0
                # Proper Kronecker: U[i*4+j] = u1[i*2 + j_hi] * u2[i_lo * 2 + j_lo]
                # where i = i_hi*2 + i_lo, j = j_hi*2 + j_lo
                U = [0]*16
                V = [0]*16
                W = [0]*16
                for i_hi in range(2):
                    for i_lo in range(2):
                        i = i_hi*2 + i_lo
                        for j_hi in range(2):
                            for j_lo in range(2):
                                j = j_hi*2 + j_lo
                                U[i*4+j] = u1[i_hi*2+j_hi] * u2[i_lo*2+j_lo]
                                V[i*4+j] = v1[i_hi*2+j_hi] * v2[i_lo*2+j_lo]
                                W[i*4+j] = w1[i_hi*2+j_hi] * w2[i_lo*2+j_lo]
                terms.append((U, V, W))
        return terms

    U_list = []
    V_list = []
    W_list = []

    # Part 1: 4x4x4 (first 4 cols of B), rank 49
    for U4, V4, W4 in build_4x4x4_strassen():
        Uf = [0]*16
        Vf = [0]*28
        Wf = [0]*28
        for i in range(4):
            for j in range(4):
                Uf[i*4+j] = U4[i*4+j]
                for k in range(4):
                    Vf[j*7+k] = V4[j*4+k]
                    Wf[k*4+i] = W4[k*4+i]
        U_list.append(Uf)
        V_list.append(Vf)
        W_list.append(Wf)

    # Part 2: 4x4x3 (last 3 cols of B), schoolbook rank 48
    for i in range(4):
        for j in range(4):
            for k in range(3):
                Uf = [0]*16
                Vf = [0]*28
                Wf = [0]*28
                Uf[i*4+j] = 1
                Vf[j*7+(4+k)] = 1
                Wf[(4+k)*4+i] = 1
                U_list.append(Uf)
                V_list.append(Vf)
                W_list.append(Wf)

    rank = len(U_list)

    # Verify
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
