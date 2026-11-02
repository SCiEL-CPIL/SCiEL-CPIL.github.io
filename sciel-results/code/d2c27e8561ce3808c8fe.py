"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition using Strassen for 2x2x2 subblocks and trivial for 1x2x2
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Partition j (0..3) into J0={0,1}, J1={2,3}
    # Partition k (0..3) into K0={0,1}, K1={2,3}
    
    for a in range(2):
        for b in range(2):
            j0 = 2 * a
            j1 = 2 * a + 1
            k0 = 2 * b
            k1 = 2 * b + 1
            
            base_col = (a * 2 + b) * 11
            
            # --- Strassen Terms (7) for rows 0,1 ---
            
            # M1 = (A0j0 + A1j1)(Bj0k0 + Bj1k1)
            col = base_col + 0
            U[0 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] += 1
            W[k0 * n + 1][col] += 1
            W[k1 * n + 1][col] += 1
            
            # M2 = (A1j0 + A1j1)Bj0k0
            col = base_col + 1
            U[1 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + 1][col] += 1
            W[k1 * n + 1][col] += -1
            
            # M3 = A0j0(Bj1k0 - Bj1k1)
            col = base_col + 2
            U[0 * m + j0][col] += 1
            V[j1 * p + k0][col] += 1
            V[j1 * p + k1][col] += -1
            W[k1 * n + 0][col] += 1
            W[k1 * n + 1][col] += 1
            
            # M4 = A1j1(Bj0k1 - Bj0k0)
            col = base_col + 3
            U[1 * m + j1][col] += 1
            V[j0 * p + k1][col] += 1
            V[j0 * p + k0][col] += -1
            W[k0 * n + 0][col] += 1
            W[k0 * n + 1][col] += 1
            
            # M5 = (A0j0 + A0j1)Bj1k1
            col = base_col + 4
            U[0 * m + j0][col] += 1
            U[0 * m + j1][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] += -1
            W[k1 * n + 0][col] += 1
            
            # M6 = (A1j0 - A0j0)(Bj0k0 + Bj1k0)
            col = base_col + 5
            U[1 * m + j0][col] += 1
            U[0 * m + j0][col] += -1
            V[j0 * p + k0][col] += 1
            V[j1 * p + k0][col] += 1
            W[k1 * n + 1][col] += 1
            
            # M7 = (A0j1 - A1j1)(Bj0k1 + Bj1k1)
            col = base_col + 6
            U[0 * m + j1][col] += 1
            U[1 * m + j1][col] += -1
            V[j0 * p + k1][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] += 1

            # --- Trivial Terms (4) for row 2 ---
            
            # T1 = A2j0 Bj0k0
            col = base_col + 7
            U[2 * m + j0][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + 2][col] += 1
            
            # T2 = A2j1 Bj1k0
            col = base_col + 8
            U[2 * m + j1][col] += 1
            V[j1 * p + k0][col] += 1
            W[k0 * n + 2][col] += 1
            
            # T3 = A2j0 Bj0k1
            col = base_col + 9
            U[2 * m + j0][col] += 1
            V[j0 * p + k1][col] += 1
            W[k1 * n + 2][col] += 1
            
            # T4 = A2j1 Bj1k1
            col = base_col + 10
            U[2 * m + j1][col] += 1
            V[j1 * p + k1][col] += 1
            W[k1 * n + 2][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
