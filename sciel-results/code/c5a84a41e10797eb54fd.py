"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

def _make_2x3x9_rank_50():
    # Returns U (6x50), V (27x50), W (18x50) as lists of lists.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=2, m=3, p=9.
    n, m, p = 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0

    def add_term(U_entries, V_entries, W_entries, c_idx):
        for (u_idx, u_val) in U_entries:
            U[u_idx][c_idx] += u_val
        for (v_idx, v_val) in V_entries:
            V[v_idx][c_idx] += v_val
        for (w_idx, w_val) in W_entries:
            W[w_idx][c_idx] += w_val

    # --- Part A: 2x2x9 (rank 32) ---
    # Part A1: 4 blocks of 2x2x2 (Strassen, rank 7 each) -> 28
    # For each group g in 0..3, k0 = 2g, k1 = 2g+1
    # A_top = [[a00, a01], [a10, a11]]
    # B_top = [[b0k0, b0k1], [b1k0, b1k1]]
    # C_top = [[c0k0, c0k1], [c1k0, c1k1]]
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # Strassen terms for 2x2 matrix multiplication
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        # M2 = (a10 + a11) * b0k0
        # M3 = a00 * (b1k1 - b0k1)
        # M4 = a11 * (b0k0 - b1k0)
        # M5 = (a00 + a01) * b1k1
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        add_term(
            [(0*m+0, 1), (1*m+1, 1)],  # a00 + a11
            [(0*p+k0, 1), (1*p+k1, 1)], # b0k0 + b1k1
            [(k0*n+0, 1), (k1*n+1, -1)], # C00 +=, C11 -=
            c
        )
        c += 1
        
        # Term 2: M2
        add_term(
            [(1*m+0, 1), (1*m+1, 1)],  # a10 + a11
            [(0*p+k0, 1)],             # b0k0
            [(k0*n+1, 1), (k1*n+1, -1)], # C10 +=, C11 -=
            c
        )
        c += 1
        
        # Term 3: M3
        add_term(
            [(0*m+0, 1)],              # a00
            [(1*p+k1, 1), (0*p+k1, -1)], # b1k1 - b0k1
            [(k1*n+0, 1), (k1*n+1, 1)],  # C01 +=, C11 +=
            c
        )
        c += 1
        
        # Term 4: M4
        add_term(
            [(1*m+1, 1)],              # a11
            [(0*p+k0, 1), (1*p+k0, -1)], # b0k0 - b1k0
            [(k0*n+0, -1), (k0*n+1, 1)],  # C00 -=, C10 +=
            c
        )
        c += 1
        
        # Term 5: M5
        add_term(
            [(0*m+0, 1), (0*m+1, 1)],  # a00 + a01
            [(1*p+k1, 1)],             # b1k1
            [(k0*n+0, -1), (k1*n+0, 1)], # C00 -=, C01 +=
            c
        )
        c += 1
        
        # Term 6: M6
        add_term(
            [(1*m+0, 1), (0*m+0, -1)],  # a10 - a00
            [(0*p+k0, 1), (0*p+k1, 1)], # b0k0 + b0k1
            [(k1*n+1, 1)],              # C11 +=
            c
        )
        c += 1
        
        # Term 7: M7
        add_term(
            [(0*m+1, 1), (1*m+1, -1)],  # a01 - a11
            [(1*p+k0, 1), (1*p+k1, 1)], # b1k0 + b1k1
            [(k0*n+0, 1)],              # C00 +=
            c
        )
        c += 1

    # Part A2: 2x2x1 for k=8 (Schoolbook, rank 4)
    k8 = 8
    # C08 = a00*b08 + a01*b18
    # C18 = a10*b08 + a11*b18
    
    add_term([(0*m+0, 1)], [(0*p+k8, 1)], [(k8*n+0, 1)], c); c += 1
    add_term([(0*m+1, 1)], [(1*p+k8, 1)], [(k8*n+0, 1)], c); c += 1
    add_term([(1*m+0, 1)], [(0*p+k8, 1)], [(k8*n+1, 1)], c); c += 1
    add_term([(1*m+1, 1)], [(1*p+k8, 1)], [(k8*n+1, 1)], c); c += 1

    # --- Part B: 2x1x9 (rank 18) ---
    # C_{i,k} += a_{i2} * b_{2k}
    for i in range(2):
        for k in range(9):
            add_term(
                [(i*m+2, 1)], # a_{i2}
                [(2*p+k, 1)], # b_{2k}
                [(k*n+i, 1)], # C_{ik}
                c
            )
            c += 1

    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    try:
        U, V, W = _make_2x3x9_rank_50()
        if time.time() > deadline:
            raise TimeoutError()
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook if something goes wrong
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
