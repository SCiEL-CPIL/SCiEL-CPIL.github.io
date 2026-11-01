"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    # Rank: 12 (2x1x6) + 2 * 3 * 7 (2x2x6 using 3 Strassen blocks) = 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: j=0, 2x1x6 = 12 terms
    for i in range(n):
        for k in range(p):
            U[i * m][col] = 1
            V[k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1..4, two 2x2x6 blocks
    for block in range(2):
        j_base = 1 + block * 2
        for s in range(3):
            k_base = s * 2
            strassen = [
                ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
                ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
                ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
                ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
                ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
            ]
            for u_list, v_list, w_list in strassen:
                for (li, lj, val) in u_list:
                    U[li * m + (j_base + lj)][col] += val
                for (lj, lk, val) in v_list:
                    V[(j_base + lj) * p + (k_base + lk)][col] += val
                for (li, lk, val) in w_list:
                    W[(k_base + lk) * n + li][col] += val
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
